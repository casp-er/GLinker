# PostgreSQL retrieval and selective upstream reconciliation

This branch starts at `4b86fbe` (the Broadside pin), not fork main. The verified
merge-base with upstream/main (`a781ca5`) was `0e66584`; upstream had 20 commits,
and fork main was four commits behind the pin. A merge-tree check reproduced
10 conflict sections in L2 component/processor and L3 processor.

## Reconciliation decision

Selected the Redis embedding changes from upstream `4d5f051`, and the required
rapidfuzz dependency from `b8e942c`. This is a selective integration, **not a
merge of all upstream/main**. L3 modernization and the sparse-model changes
are excluded; the existing L3 fixes remain unchanged.

Upstream's chain `batch_search` runs exact and fuzzy modes for every mention,
concatenates exact before fuzzy, and deduplicates by entity ID (first wins).
It returns the entire batch when *any* mention has results, so misses never
reach lower layers. Batch cache writeback is a TODO. Its processor also skips
all search pipeline steps after unconditionally starting a batch search.

Retain `search_many`: unique normalized ES queries, fuzzy requests only for
exact misses, independent per-mention layer fallback, cache writeback, and
custom pipeline handling. Add upstream-compatible `batch_search` entry points:
ES exposes explicit exact/fuzzy batches; the chain delegates to search_many.
The two policies are not recall-equivalent: unioning fuzzy candidates can find
entities absent from exact hits, but adds noise and work. This branch preserves
the existing exact-first policy deliberately, not because it was conflict-free.

Redis embeddings live once at `entity:emb:{id}`. Normal search now hydrates those
keys (missing from the upstream change), preserving the L3 embedding cache path.
Legacy inline embeddings remain readable. Cache keys expire naturally; no live
Redis migration or cleanup was run.

## PostgreSQL schema and semantics

Use PostgreSQL 14+ with UTF-8 and extensions `pg_trgm`, `unaccent`, and
`fuzzystrmatch` in public. Explicit setup creates a dedicated schema with
entities/aliases, foreign keys, alias lookup primary key, normalized columns,
normalization triggers, and three `gin_trgm_ops` GIN indexes. Connection setup
runs no DDL and rejects a missing KB schema. Supply an explicit DSN or database.

Normalization is NFKC → unaccent → lowercase → trim, applied by the **same SQL
function** to inserted/updated fields and queries. It handles tested accents,
combining marks, ligatures, and full-width characters; it is not a claim of
complete ICU folding equivalence. If the folding rules change, stored columns
must be rebuilt explicitly. Existing legacy tables are not migrated in place.

Exact retrieval uses escaped, word-bounded phrases over label, aliases, and
description, with weights 2 / 1.5 / 1. Fuzzy retrieval uses the indexable `%`
operator, then bounded Levenshtein filtering (`max_distance`, default 2). This
filters unrelated popular trigram matches before GLiNKER's popularity sort.
Fuzzy comparison is whole-field and limited to 255 characters, unlike ES's
per-token fuzziness. Exact description search supports longer text. Three
independent indexable branches are unioned and deduplicated before fetching
aliases for the top 50 entities. Search transactions roll back on errors;
statement timeout defaults to five seconds. Short/common mentions may remain
expensive. No seven-million-entity performance claim is made.

References: [pg_trgm index support](https://www.postgresql.org/docs/16/pgtrgm.html),
[unaccent](https://www.postgresql.org/docs/16/unaccent.html),
[bounded Levenshtein](https://www.postgresql.org/docs/16/fuzzystrmatch.html).

## Explicit provisioning/import (operator action)

These commands write the selected database. Obtain separate explicit approval
before each invocation against any real local/development/production database.
Do not point validation at the application's `.env` or `DATABASE_URL`.

```sh
python scripts/init_postgres.py --dsn 'service=glinker-kb' --schema glinker
python scripts/load_postgres.py --dsn 'service=glinker-kb' --schema glinker --input /path/to/wikidata_production.jsonl
```

The loader streams batches and analyzes both tables. It does not create a
schema, delete other schemas, or clear the database. `--overwrite` updates
existing records and replaces aliases, including clearing an empty alias list.
Provisioning credentials need extension/schema privileges; runtime credentials
need SELECT, plus UPDATE if embedding writeback is enabled.

## Retrieval comparison, 2026-09-14

Two disposable containers: PostgreSQL 16 and Elasticsearch 8.15.0 with ICU.
No application database was used. Each run created and cleaned a UUID schema
and index. The ES mappings match Broadside-ML's init_es.py and use the server's
`popularity_boost=True`, exact/fuzzy fallback, max 50 candidates and final
popularity sort / top three.

Corpus: first 50,000 rows of the available Wikidata production JSONL plus 19
missing expected entities, identical in both databases. Corpus SHA-256:
`16233aca884aec08de20253b91ca5659fe7417d0c57501be63a6f7459a774951`.
The 291 committed mentions contain 31 news/Unicode/ambiguity cases, three
real description-only phrases, and 100 deterministically sampled entities
ordered by SHA-256(entity_id), tested as labels, middle-character deletion
variants, and first aliases where available. This is a sampled retrieval test,
not a blinded benchmark, full-KB test, or end-to-end disambiguation evaluation.

| Result | Elasticsearch | PostgreSQL |
|---|---:|---:|
| Expected ID in final top three | 93/291 | 290/291 |
| Median per-mention time | 4.40 ms | 85.41 ms |
| p95 per-mention time | 8.83 ms | 132.76 ms |
| Maximum per-mention time | 17.88 ms | 471.84 ms |

No ES-hit case regressed. Top-three sets agree in only 45/291 cases. The remaining
PostgreSQL miss is the ambiguous alias `Heart` (Q21071553); ES also missed it.
These low ES scores include intentionally adversarial synthetic typos and
low-popularity entities, not a measured news-production accuracy rate.

The first version used substring matching and unconstrained trigram similarity:
248/291 PG hits with seven ES-hit regressions, because unrelated popular
candidates displaced the correct entity. Word boundaries and bounded edit
filtering fixed that mechanism; a dedicated regression test covers it.
`tests/l2/retrieval/report.json` contains all IDs, ranks, timings, and index plans.
Index plans with sequential scans disabled establish index eligibility only,
not full-scale planner choices. The final report checks regex and fuzzy predicates.
Timings are single-query observations on a shared development host immediately
after loading, not a controlled benchmark: an earlier run of the same retrieval
code measured PostgreSQL median 9.99 ms and p95 34.63 ms. Index pending-list and
cache state were not controlled. PostgreSQL was slower than ES in both runs.

```sh
python scripts/prepare_retrieval_corpus.py --input /path/to/wikidata_production.jsonl --mentions tests/l2/retrieval/mentions.json --output /tmp/kb-comparison.jsonl
python scripts/compare_retrieval.py --pg-dsn 'host=127.0.0.1 port=55439 dbname=glinker_retrieval_test user=postgres password=TEST_PASSWORD' --es-host http://127.0.0.1:59239 --es-settings tests/l2/retrieval/es-settings.json --corpus /tmp/kb-comparison.jsonl --mentions tests/l2/retrieval/mentions.json --output /tmp/retrieval-report.json
```

The comparison requires an explicitly selected database ending in `_test`.
Use disposable services, never a production ES endpoint. Full-KB index size,
load time, latency/concurrency under the intended RAM budget, and downstream
L3 quality remain deployment acceptance checks.

## Code validation

59 L2 tests passed, including isolated PostgreSQL integration, Redis embedding
hydration/update complexity, per-mention fallback, ES batching, and DSN builder
coverage. Broadside-ML's 10 tests and Compose configuration validation passed.
The existing config-builder suite had 47 passes and two failures, reproduced
against the original `4b86fbe`: stale DictLayer similarity default expectation
(0.6 versus existing 0.75), and an outdated requirement for mandatory L1.
No L3 models were downloaded or inference run for this retrieval-only change.
