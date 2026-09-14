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
python scripts/compare_retrieval.py --pg-dsn 'host=127.0.0.1 port=55439 dbname=glinker_retrieval_test user=postgres password=TEST_PASSWORD' --es-host http://127.0.0.1:59239 --es-settings tests/l2/retrieval/es-settings.json --corpus /tmp/kb-comparison.jsonl --mentions tests/l2/retrieval/mentions.json --redis-host 127.0.0.1 --redis-port 63790 --redis-ttl 604800 --output /tmp/retrieval-report.json
```

The comparison requires an explicitly selected database ending in `_test`.
Use disposable services, never a production ES endpoint. Full-KB index size,
load time, latency/concurrency under the intended RAM budget, and downstream
L3 quality remain deployment acceptance checks.

## Batching and cache re-validation, 2026-09-14

The 2026-09-14 comparison above measured `search_many` called with exactly
one mention per case — the same shape as the pre-fix code path, since
PostgresLayer had no batching override and search_many degraded to the
base class's per-mention loop. This section re-runs the same corpus and
methodology through the fixed `PostgresLayer.search_many` (native batching,
`compare_retrieval.py` unchanged in structure, extended with two new
measurements) and adds a Redis cache layer measurement that the original
comparison didn't have at all. An earlier attempt at this measurement called
`search_many` with all 291 mentions in one request and silently got back
empty results for every mention because Postgres's 5-second
`statement_timeout` canceled the query — that's what motivated Task 5b's
`_batch_retrieve_chunked` fix. The batched-throughput measurement below
calls `search_many` in 15-mention chunks (matching broadside-ml's
`GLINKER_MAX_MENTIONS`), which is below `_chunk_size()`'s default of 25 and
so never actually exercises `_batch_retrieve_chunked`'s internal
chunk-splitting — it just confirms that a request shape a real pipeline run
actually produces stays well within the timeout on its own. It's the cache
pass below, which calls `chain.search_many` with the full 291-mention list
in one request, that exercises the chunking fix: internally that splits
into roughly a dozen 25-mention chunks, and the cold/warm pass completing
with zero regressions is the confirmation that Task 5b's fix works.

Same disposable-container methodology, same corpus (SHA-256 as above),
same 291 mentions.

| Result | Value |
|---|---:|
| Corpus recall (top three), sequential per-mention | 290/291 |
| Batched throughput, whole corpus in 15-mention `search_many` chunks | 3050.39 ms total, 10.48 ms/mention |
| Cold cache pass (Redis empty, populates via cache writeback) | 11.22 ms/mention |
| Warm cache pass (same 291 mentions repeated) | 0.33 ms/mention |

Regressions vs. the sequential per-mention baseline: none. Both
`batched.regressions_vs_sequential` and `cache.regressions_vs_sequential`
were empty lists, and `summary.pg_regressions` was also empty.

Carrying forward the "not a controlled benchmark" caveat from the section
above: the batched measurement runs immediately after a full sequential
pass over the same 291 mentions, in the same script invocation, so
Postgres's OS/shared-buffer cache is already warm by the time batching is
measured — the same kind of host/cache-state variance the section above
documented as an ~8.5x swing (85.41 ms vs. 9.99 ms median) on *identical*
unbatched code. The drop from that section's 85.41 ms median to this
section's 10.48 ms/mention batched figure reflects both batching's
round-trip amortization and this warm-cache effect; this comparison does
not isolate how much of the improvement is attributable to each.

The batched-throughput number is nonetheless the one that matters for a
real pipeline run: NER surfaces multiple mentions per article
(`GLINKER_MAX_MENTIONS=15` in broadside-ml), and the pre-fix per-mention
round-trip cost compounded linearly with mention count regardless of cache
state. The Redis timing used a loopback connection inside the disposable
test environment, so the 0.33 ms warm result is not a claim about latency
over Tailscale or another deployment network. `RedisLayer.search_many` now
pipelines all query-key reads and all unique embedding reads, and cache
writeback pipelines the full result set, bounding network round trips for a
request. Measure the real deployment route separately before setting its
latency budget.

### Redis TTL and cache invalidation

`RedisLayer.write_cache` previously ignored `LayerConfig.ttl`'s documented
"0 = no expiry" contract entirely — every write used `SETEX`, which Redis
rejects outright for `ttl<=0`. That's now implemented (`RedisLayer._cache_set`
uses plain `SET` when `ttl<=0`), so a genuinely non-expiring cache is
possible. broadside-ml's wiring does not use it: the KB only changes on an
explicit operator-approved `scripts/load_postgres.py --overwrite` reload,
but a non-expiring cache means any reload that isn't paired with a manual
flush leaves stale candidates (outdated popularity/aliases/descriptions)
cached indefinitely with no automatic recovery. broadside-ml instead uses a
long, finite default TTL (`GLINKER_REDIS_TTL`, default 7 days — longer than
Elasticsearch's 86400s default cache TTL, since the KB changes far less
often than that comparison implies) as a safety net: a missed manual flush
self-heals within a week instead of never.

**Cache invalidation on KB reload:** after any `scripts/load_postgres.py
--overwrite` run against broadside-ml's Postgres KB, flush the Redis cache
layer before restarting (or alongside restarting) the encoder — either
`DatabaseChainComponent.get_layer("redis").clear()` from a Python shell
against the running pipeline, or `redis-cli -h <host> -p <port> -a
<password> FLUSHDB` against the dedicated Redis DB index broadside-ml uses
(not the whole Redis instance — broadside's own application data may share
that instance on a different DB index). This is a manual step, same as the
KB reload itself requiring explicit per-invocation approval.

## Code validation

`uv run --extra dev pytest tests/l2/ -v` (2026-09-14, this branch's tip):
66 collected, 61 passed, 5 skipped. The 5 skips are
`tests/l2/test_postgres_integration.py`, which requires a live
`GLINKER_TEST_DSN` (normalization/description/alias/fuzzy/embeddings,
query-error rollback, word-boundary + edit-distance filtering, and
`search_many` parity with sequential search — including a constant-round-trip
bound) and are expected to skip in this environment, not fail.

Beyond the original PostgreSQL integration, Redis embedding
hydration/update, per-mention fallback, ES batching, and DSN builder
coverage, `tests/l2/test_reconciliation.py` now also covers this branch's
batching and cache work specifically: `PostgresLayer.search_many`'s native
batched exact/fuzzy dispatch and its restriction of fuzzy retry to unique
misses; `_batch_retrieve_chunked`'s statement_timeout-safe chunking,
including that one chunk's failure doesn't lose another chunk's results and
that a mention whose exact-phase chunk failed is excluded from `search_many`'s
fuzzy retry rather than re-issued against a connection that may have just
timed out; `RedisLayer._cache_set`'s `ttl<=0` no-expiry contract (`SET` vs.
`SETEX`); batched Redis query, embedding, and writeback pipelines; overwrite
invalidation of stale precomputed embeddings; and the config builder's
Postgres DSN/credential preservation and Redis password forwarding. L0 and
L3 regression tests cover punctuation-ending aliases and per-mention
popularity normalization respectively. No L3 models were downloaded or
inference run for these tests.
