"""Opt-in integration checks; GLINKER_TEST_DSN must name a throwaway *_test DB."""

import importlib.util
import os
from pathlib import Path
import uuid

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import parse_dsn
import pytest

from glinker.l2.component import PostgresLayer
from glinker.l2.models import DatabaseRecord, LayerConfig


@pytest.fixture
def pg():
    dsn = os.environ.get("GLINKER_TEST_DSN")
    if not dsn:
        pytest.skip("Set GLINKER_TEST_DSN explicitly to a throwaway *_test database")
    conn = psycopg2.connect(dsn)
    if not conn.info.dbname.endswith("_test"):
        conn.close()
        pytest.fail("Refusing a non-test database")
    schema = "glinker_test_" + uuid.uuid4().hex
    path = Path(__file__).parents[2] / "scripts" / "init_postgres.py"
    spec = importlib.util.spec_from_file_location("init_postgres", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.initialize(conn, schema)
    cfg = parse_dsn(dsn)
    cfg["database"] = cfg.pop("dbname")
    cfg["schema"] = schema
    layer = PostgresLayer(LayerConfig(type="postgres", priority=0, config=cfg))
    try:
        yield layer
    finally:
        layer.conn.close()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
        conn.close()


def test_normalization_description_alias_fuzzy_and_embeddings(pg):
    record = DatabaseRecord(
        entity_id="Q1",
        label="Straße São Ｐａｕｌｏ",
        aliases=["Recep Tayyip Erdoğan"],
        description="unique description marker",
    )
    pg.load_bulk([record])
    assert pg.search("STRASSE SAO Paulo")[0].entity_id == "Q1"
    assert pg.search("Sa\u0303o")[0].entity_id == "Q1"
    assert pg.search("Erdogan")[0].entity_id == "Q1"
    assert pg.search("unique description")[0].entity_id == "Q1"
    assert pg.search_fuzzy("Recep Tayyip Erdogna")[0].entity_id == "Q1"
    assert not pg.search("%")
    assert not pg.search("_")
    assert not pg.search("  ")
    pg.update_embeddings(["Q1"], [[1.0, 2.0]], "model")
    assert pg.search("Erdogan")[0].embedding == [1.0, 2.0]
    record.aliases = []
    pg.load_bulk([record], overwrite=True)
    assert not pg.search("Erdogan")


def test_query_error_rolls_back_connection(pg):
    with pytest.raises(psycopg2.Error):
        with pg.conn:
            with pg.conn.cursor() as cur:
                cur.execute("SELECT 1/0")
    assert pg.search("nothing") == []


def test_word_boundaries_and_edit_distance_reject_popular_noise(pg):
    pg.load_bulk(
        [
            DatabaseRecord(entity_id="target", label="Deliverance", aliases=["Acre"], popularity=1),
            DatabaseRecord(
                entity_id="noise", label="trance", description="massacre", popularity=1000
            ),
        ]
    )
    assert [r.entity_id for r in pg.search("Acre")] == ["target"]
    assert [r.entity_id for r in pg.search_fuzzy("Delivrance")] == ["target"]
