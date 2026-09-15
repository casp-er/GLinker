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
    layer = PostgresLayer(
        LayerConfig(type="postgres", priority=0, search_mode=["exact", "fuzzy"], config=cfg)
    )
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
    record.description = "changed description"
    pg.load_bulk([record], overwrite=True)
    assert pg.search("Erdogan")[0].embedding is None
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


def test_search_many_matches_sequential_search_and_search_fuzzy(pg):
    pg.load_bulk(
        [
            DatabaseRecord(entity_id="Q1", label="Straße São Ｐａｕｌｏ", popularity=5),
            DatabaseRecord(entity_id="Q2", label="Deliverance", aliases=["Acre"], popularity=1),
            DatabaseRecord(entity_id="Q3", label="trance", description="massacre", popularity=1000),
        ]
    )

    mentions = ["STRASSE SAO Paulo", "Acre", "Delivrance", "no such entity", "Acre"]
    batched = pg.search_many(mentions)

    sequential = []
    for mention in mentions:
        exact = pg.search(mention)
        sequential.append(exact if exact else pg.search_fuzzy(mention))

    assert [[r.entity_id for r in result] for result in batched] == [
        [r.entity_id for r in result] for result in sequential
    ]
    assert [r.entity_id for r in batched[0]] == ["Q1"]
    assert [r.entity_id for r in batched[1]] == ["Q2"]
    assert [r.entity_id for r in batched[2]] == ["Q2"]  # fuzzy fallback
    assert batched[3] == []
    assert batched[1] == batched[4]  # duplicate mention, same result


class _CursorCountingConn:
    """Wraps a real psycopg2 connection to count `cursor()` open calls.

    psycopg2.extensions.connection is an immutable C type: it has no
    `__dict__` (dictoffset 0) and rejects attribute assignment on both the
    instance and the class ("cannot set 'cursor' attribute of immutable
    type"), confirmed against psycopg2-binary 2.9.13 / Python 3.14 in this
    environment. `unittest.mock.patch.object(pg.conn, "cursor", ...)`
    therefore cannot spy on a real connection at all -- it fails during
    __enter__/__exit__ with AttributeError before any query runs. This
    proxy gets the same spy behavior by wrapping `pg.conn` itself (a plain
    Python attribute on PostgresLayer, which has no such restriction).
    """

    def __init__(self, real_conn):
        self._real_conn = real_conn
        self.call_count = 0

    def cursor(self, *args, **kwargs):
        self.call_count += 1
        return self._real_conn.cursor(*args, **kwargs)

    def __enter__(self):
        return self._real_conn.__enter__()

    def __exit__(self, *exc_info):
        return self._real_conn.__exit__(*exc_info)

    def __getattr__(self, name):
        return getattr(self._real_conn, name)


def test_search_many_uses_a_constant_number_of_round_trips(pg):
    pg.load_bulk(
        [DatabaseRecord(entity_id=f"Q{i}", label=f"Entity{i}", popularity=i) for i in range(20)]
    )
    mentions = [f"Entity{i}" for i in range(20)]

    spy = _CursorCountingConn(pg.conn)
    pg.conn = spy
    pg.search_many(mentions)
    # Bounded by mention count is what we're fixing away from: 20 mentions
    # should not need anywhere near 20 cursor()-openings. Normalize (1) +
    # equality chunks (ceil(20 / (timeout_ms // _ASSUMED_MS_PER_MENTION))
    # = ceil(20 / 5) = 4) is 5; nothing here needs the phrase or fuzzy
    # phases since every mention is an exact label match.
    assert spy.call_count <= 6


def test_direct_equality_resolves_short_aliases_before_trigram_scan(pg):
    pg.load_bulk(
        [
            DatabaseRecord(
                entity_id="Qfinancialtimes",
                label="Financial Times",
                aliases=["FT"],
                popularity=900,
            ),
            DatabaseRecord(
                entity_id="Qfoot",
                label="Foot",
                aliases=["ft"],
                popularity=10,
            ),
        ]
    )

    # "ft" is below the trigram minimum, so only the equality pass runs;
    # both alias matches must come back, more popular first.
    results = pg.search_many(["FT"])
    assert [r.entity_id for r in results[0]] == [
        "Qfinancialtimes",
        "Qfoot",
    ]

    detailed = pg.search_many_detailed(["FT"])
    assert detailed[0].status == "matched"


def test_search_many_detailed_reports_closed_backend_as_failure(pg):
    pg.load_bulk([DatabaseRecord(entity_id="Q1", label="Findable")])
    pg.conn.close()

    detailed = pg.search_many_detailed(["findable"])

    assert detailed[0].status == "backend_error"
    assert detailed[0].failures
    assert all(failure.layer == "PostgresLayer" for failure in detailed[0].failures)
    # Candidate view keeps its shape; the failure distinction is the point.
    assert pg.search_many(["findable"]) == [[]]
