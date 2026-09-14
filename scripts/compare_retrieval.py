"""Compare actual ES/PG candidate retrieval on a shared JSONL corpus.

Only runs against an explicitly named throwaway database ending in _test.
Creates a UUID schema/index, loads the same corpus, reports raw and pipeline
(top three by popularity) results, then removes only those UUID resources.
The mention file is a JSON list of {mention, expected_id, category} objects.
ES settings must be exported from the application's init_es.INDEX_SETTINGS.
"""

import argparse
import json
import time
import uuid
from pathlib import Path

import psycopg2
from psycopg2 import sql
from elasticsearch import Elasticsearch
from glinker.l2.component import ElasticsearchLayer, PostgresLayer
from glinker.l2.models import DatabaseRecord, LayerConfig
from init_postgres import initialize


def compare(args):
    conn = psycopg2.connect(args.pg_dsn)
    if not conn.info.dbname.endswith("_test"):
        conn.close()
        raise ValueError("Comparison requires a throwaway database ending in _test")
    suffix = uuid.uuid4().hex
    schema, index = "glinker_test_" + suffix, "glinker-test-" + suffix
    es_client = Elasticsearch(args.es_host, request_timeout=60)
    es_client.cluster.health(wait_for_status="yellow", timeout="60s")
    pg = None
    try:
        initialize(conn, schema)
        settings = json.loads(Path(args.es_settings).read_text())
        es_client.indices.create(index=index, **settings)
        es_client.cluster.health(index=index, wait_for_status="green", timeout="60s")
        es = ElasticsearchLayer(
            LayerConfig(
                type="elasticsearch",
                priority=0,
                search_mode=["exact", "fuzzy"],
                config={"hosts": [args.es_host], "index_name": index, "popularity_boost": True},
            )
        )
        es.client = es.client.options(request_timeout=60)
        config = {"dsn": args.pg_dsn, "schema": schema}
        pg = PostgresLayer(
            LayerConfig(type="postgres", priority=0, search_mode=["exact", "fuzzy"], config=config)
        )
        count = 0
        batch = []
        with open(args.corpus) as source:
            for line in source:
                batch.append(DatabaseRecord(**json.loads(line)))
                if len(batch) == 1000:
                    pg.load_bulk(batch)
                    assert es.load_bulk(batch) == len(batch)
                    count += len(batch)
                    batch = []
            if batch:
                pg.load_bulk(batch)
                assert es.load_bulk(batch) == len(batch)
                count += len(batch)
        es_client.indices.refresh(index=index)
        with pg.conn:
            with pg.conn.cursor() as cur:
                cur.execute("ANALYZE entities")
                cur.execute("ANALYZE aliases")
        cases = json.loads(Path(args.mentions).read_text())
        report = {"corpus_size": count, "cases": []}
        for case in cases:
            item = dict(case)
            for name, layer in [("es", es), ("pg", pg)]:
                start = time.perf_counter()
                records = layer.search_many([case["mention"]])[0]
                item[name + "_ms"] = round((time.perf_counter() - start) * 1000, 2)
                item[name + "_raw"] = [r.entity_id for r in records]
                item[name + "_top3"] = [
                    r.entity_id
                    for r in sorted(records, key=lambda r: r.popularity, reverse=True)[:3]
                ]
                item[name + "_hit3"] = case["expected_id"] in item[name + "_top3"]
            report["cases"].append(item)
        n = len(cases)
        report["summary"] = {
            "mentions": n,
            "es_hits3": sum(c["es_hit3"] for c in report["cases"]),
            "pg_hits3": sum(c["pg_hit3"] for c in report["cases"]),
            "pg_regressions": [
                c["mention"] for c in report["cases"] if c["es_hit3"] and not c["pg_hit3"]
            ],
            "top3_set_agreement": sum(
                set(c["es_top3"]) == set(c["pg_top3"]) for c in report["cases"]
            ),
        }
        # Verify the planner can use each trigram index, including fuzzy ops.
        report["index_plans"] = []
        with pg.conn:
            with pg.conn.cursor() as cur:
                cur.execute("SET LOCAL enable_seqscan = off")
                for table, column in [
                    ("entities", "label_folded"),
                    ("entities", "description_folded"),
                    ("aliases", "alias_folded"),
                ]:
                    for predicate in [f"{column} ~ '\\merdogan\\M'", f"{column} % 'erdogan'"]:
                        cur.execute(
                            f"EXPLAIN (FORMAT JSON) SELECT entity_id FROM {table} WHERE {predicate}"
                        )
                        report["index_plans"].append(cur.fetchone()[0])
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report["summary"], indent=2))
    finally:
        if pg:
            pg.conn.close()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
                )
        conn.close()
        es_client.indices.delete(index=index, ignore_unavailable=True)
        es_client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ["pg-dsn", "es-host", "es-settings", "corpus", "mentions", "output"]:
        parser.add_argument("--" + option, required=True)
    compare(parser.parse_args())
