"""Regressions for the selectively reconciled upstream L2 changes."""

import json
from unittest.mock import MagicMock, patch

from glinker.l2.component import (
    DatabaseChainComponent,
    RedisLayer,
    ElasticsearchLayer,
    PostgresLayer,
)
from glinker.l2.models import (
    DatabaseRecord,
    LayerConfig,
    L2Config,
    RetrievalFailure,
    MentionRetrieval,
)


def test_redis_embedding_update_does_not_scan_aliases():
    with patch("glinker.l2.component.redis.Redis") as client:
        layer = RedisLayer(LayerConfig(type="redis", priority=0))
    pipe = client.return_value.pipeline.return_value
    assert layer.update_embeddings(["Q1", "Q2"], [[1.0], [2.0]], "model") == 2
    assert pipe.setex.call_count == 2
    client.return_value.scan_iter.assert_not_called()
    keys = [c.args[0] for c in pipe.setex.call_args_list]
    assert keys == ["entity:emb:Q1", "entity:emb:Q2"]


def test_redis_cache_write_separates_and_search_restores_embedding():
    with patch("glinker.l2.component.redis.Redis") as client:
        layer = RedisLayer(LayerConfig(type="redis", priority=0))
    record = DatabaseRecord(entity_id="Q1", label="One", embedding=[1.0], embedding_model_id="m")
    layer.write_cache("alias", [record], 60)
    pipe = client.return_value.pipeline.return_value
    writes = {c.args[0]: json.loads(c.args[2]) for c in pipe.setex.call_args_list}
    assert "embedding" not in writes["entity:alias"][0]
    assert writes["entity:emb:Q1"]["embedding"] == [1.0]
    client.return_value.get.return_value = json.dumps(writes["entity:alias"])
    pipe.execute.return_value = [json.dumps(writes["entity:emb:Q1"])]
    assert layer.search("alias")[0].embedding == [1.0]
    assert record.embedding == [1.0]


def test_partial_layer_hits_continue_for_missing_mentions():
    chain = DatabaseChainComponent(L2Config(layers=[]))
    cached = DatabaseRecord(entity_id="Q1", label="one")
    fetched = DatabaseRecord(entity_id="Q2", label="two")
    first, second = MagicMock(), MagicMock()
    first.search_many_detailed.return_value = [
        MentionRetrieval(candidates=[cached])
    ]  # Short backend response.
    second.search_many_detailed.return_value = [MentionRetrieval(candidates=[fetched])]
    chain.layers = [first, second]
    chain._cache_write_many = MagicMock()
    result = chain.batch_search(["one", "two"])
    assert result == [[cached], [fetched]]
    second.search_many_detailed.assert_called_once_with(["two"])
    assert chain._cache_write_many.call_count == 2


def test_redis_search_many_batches_query_and_embedding_reads():
    with patch("glinker.l2.component.redis.Redis") as client:
        layer = RedisLayer(LayerConfig(type="redis", priority=1))
    pipe = client.return_value.pipeline.return_value
    record = DatabaseRecord(entity_id="Q1", label="One")
    cached_record = json.dumps([record.model_dump()])
    cached_embedding = json.dumps({"embedding": [1.0], "embedding_model_id": "model"})
    pipe.execute.side_effect = [
        [cached_record, None],
        [cached_embedding],
    ]

    results = layer.search_many(["One", "missing", "one"])

    assert [[record.entity_id for record in result] for result in results] == [["Q1"], [], ["Q1"]]
    assert results[0][0].embedding == [1.0]
    assert pipe.execute.call_count == 2


def test_chain_batches_cache_writeback_for_many_hits():
    chain = DatabaseChainComponent(L2Config(layers=[]))
    cache = MagicMock()
    cache.priority = 1
    cache.write = True
    cache.cache_policy = "always"
    cache.ttl = 60
    cache.is_available.return_value = True
    cache.search_many_detailed.return_value = [
        MentionRetrieval(),
        MentionRetrieval(),
    ]
    source = MagicMock()
    source.priority = 0
    source.write = False
    source.is_available.return_value = True
    records = [
        DatabaseRecord(entity_id="Q1", label="one"),
        DatabaseRecord(entity_id="Q2", label="two"),
    ]
    source.search_many_detailed.return_value = [
        MentionRetrieval(candidates=[records[0]]),
        MentionRetrieval(candidates=[records[1]]),
    ]
    chain.layers = [cache, source]

    assert chain.search_many(["one", "two"]) == [[records[0]], [records[1]]]
    cache.write_cache_many.assert_called_once_with(
        [("one", [records[0]]), ("two", [records[1]])], 60
    )


def test_redis_write_cache_many_uses_one_pipeline_execution():
    with patch("glinker.l2.component.redis.Redis") as client:
        layer = RedisLayer(LayerConfig(type="redis", priority=1))
    pipe = client.return_value.pipeline.return_value
    records = [
        DatabaseRecord(entity_id="Q1", label="one"),
        DatabaseRecord(entity_id="Q2", label="two"),
    ]

    layer.write_cache_many([("one", [records[0]]), ("two", [records[1]])], 60)

    assert pipe.execute.call_count == 1
    assert [call.args[0] for call in pipe.setex.call_args_list] == ["entity:one", "entity:two"]


def test_es_fuzzy_batch_only_for_missing_unique_mentions():
    with patch("glinker.l2.component.Elasticsearch"):
        layer = ElasticsearchLayer(
            LayerConfig(
                type="elasticsearch",
                priority=0,
                search_mode=["exact", "fuzzy"],
                config={"hosts": ["http://localhost:9200"], "index_name": "test"},
            )
        )
    one = DatabaseRecord(entity_id="Q1", label="one")
    two = DatabaseRecord(entity_id="Q2", label="two")
    layer._msearch = MagicMock(side_effect=[[[one], []], [[two]]])
    assert layer.search_many(["One", "two", "one"]) == [[one], [two], [one]]
    assert layer._msearch.call_args_list[1].args[0] == [("two", True)]


def test_postgres_fuzzy_batch_only_for_missing_unique_mentions():
    with patch.object(PostgresLayer, "_setup", return_value=None):
        layer = PostgresLayer(
            LayerConfig(
                type="postgres",
                priority=0,
                search_mode=["exact", "fuzzy"],
                config={"dsn": "service=test"},
            )
        )
    one = DatabaseRecord(entity_id="Q1", label="one")
    two = DatabaseRecord(entity_id="Q2", label="second")
    layer._normalize_many = MagicMock(return_value=["one", "second", "one"])
    layer._batch_retrieve_direct_chunked = MagicMock(return_value=([[one], []], {}))
    layer._batch_retrieve_chunked = MagicMock(side_effect=[([[]], {}), ([[two]], {})])

    assert layer.search_many(["One", "second", "one"]) == [[one], [two], [one]]

    layer._batch_retrieve_direct_chunked.assert_called_once_with(["one", "second"])
    phrase_call, fuzzy_call = layer._batch_retrieve_chunked.call_args_list
    assert phrase_call.args == (["second"],)
    assert phrase_call.kwargs == {"fuzzy": False, "phase": "phrase"}
    assert fuzzy_call.args == (["second"],)
    assert fuzzy_call.kwargs == {"fuzzy": True, "phase": "fuzzy"}


def test_postgres_short_mentions_only_use_direct_equality():
    with patch.object(PostgresLayer, "_setup", return_value=None):
        layer = PostgresLayer(
            LayerConfig(
                type="postgres",
                priority=0,
                search_mode=["exact", "fuzzy"],
                config={"dsn": "service=test"},
            )
        )
    us = DatabaseRecord(entity_id="Q30", label="United States", aliases=["US"])
    layer._normalize_many = MagicMock(return_value=["us", "xy"])
    layer._batch_retrieve_direct_chunked = MagicMock(return_value=([[us], []], {}))
    layer._batch_retrieve_chunked = MagicMock()

    assert layer.search_many(["US", "XY"]) == [[us], []]

    layer._batch_retrieve_direct_chunked.assert_called_once_with(["us", "xy"])
    layer._batch_retrieve_chunked.assert_not_called()


def test_postgres_search_many_detailed_reports_backend_failure():
    with patch.object(PostgresLayer, "_setup", return_value=None):
        layer = PostgresLayer(
            LayerConfig(
                type="postgres",
                priority=0,
                search_mode=["exact", "fuzzy"],
                config={"dsn": "service=test"},
            )
        )
    one = DatabaseRecord(entity_id="Q1", label="one")
    timeout_failure = RetrievalFailure(
        layer="PostgresLayer",
        phase="direct",
        kind="timeout",
        message="statement timeout",
    )
    layer._normalize_many = MagicMock(return_value=["one", "two"])
    layer._batch_retrieve_direct_chunked = MagicMock(
        return_value=([[one], []], {1: timeout_failure})
    )

    detailed = layer.search_many_detailed(["One", "two"])

    assert [retrieval.status for retrieval in detailed] == ["matched", "backend_error"]
    assert detailed[1].failures == [timeout_failure]
    # A failed direct phase must not be retried as phrase or fuzzy search.
    layer._batch_retrieve_chunked = MagicMock()
    layer.search_many_detailed(["One", "two"])
    layer._batch_retrieve_chunked.assert_not_called()

    # The plain candidate view keeps its shape: an operational failure looks
    # like an empty list there, which is exactly why the detailed API exists.
    layer._batch_retrieve_direct_chunked = MagicMock(return_value=([[one]], {1: timeout_failure}))
    assert layer.search_many(["One", "two"]) == [[one], []]


def test_postgres_normalize_failure_is_backend_error_not_miss():
    with patch.object(PostgresLayer, "_setup", return_value=None):
        layer = PostgresLayer(
            LayerConfig(
                type="postgres",
                priority=0,
                search_mode=["exact"],
                config={"dsn": "service=test"},
            )
        )
    layer._normalize_many = MagicMock(side_effect=RuntimeError("connection closed"))

    detailed = layer.search_many_detailed(["One", "Two"])

    assert [retrieval.status for retrieval in detailed] == ["backend_error", "backend_error"]
    assert all(
        failure.phase == "normalize" and failure.kind == "query_error"
        for retrieval in detailed
        for failure in retrieval.failures
    )


def test_chain_layer_outage_falls_through_and_stays_reported():
    class OutageLayer:
        priority = 1
        write = False

        def is_available(self):
            return True

        def search_many_detailed(self, queries):
            raise RuntimeError("cache down")

    class PostgresStubLayer:
        priority = 0
        write = False

        def is_available(self):
            return True

        def search_many_detailed(self, queries):
            return [
                MentionRetrieval(candidates=[DatabaseRecord(entity_id="Q1", label="one")])
                if query == "one"
                else MentionRetrieval()
                for query in queries
            ]

    chain = DatabaseChainComponent(L2Config(layers=[]))
    chain.layers = [OutageLayer(), PostgresStubLayer()]

    detailed = chain.search_many_detailed(["one", "two"])

    # "one" resolves via the fallback layer, so the outage failure is
    # superseded; "two" stays unknown and must carry the outage failure.
    assert [retrieval.status for retrieval in detailed] == ["matched", "backend_error"]
    assert detailed[0].failures == []
    assert [(f.layer, f.phase, f.kind) for f in detailed[1].failures] == [
        ("OutageLayer", "availability", "query_error")
    ]


def test_builder_preserves_postgres_dsn_without_overriding_credentials():
    from glinker.core.builders import ConfigBuilder

    builder = ConfigBuilder(name="test")
    builder.l2.add("postgres", dsn="service=kb", schema="kb", connect_timeout=5)
    assert builder._l2_layers[0]["config"] == {
        "dsn": "service=kb",
        "schema": "kb",
        "connect_timeout": 5,
    }


def test_redis_cache_set_uses_setex_for_positive_ttl_and_set_for_no_expiry():
    with patch("glinker.l2.component.redis.Redis") as client:
        layer = RedisLayer(LayerConfig(type="redis", priority=0))
    record = DatabaseRecord(entity_id="Q1", label="One")
    pipe = client.return_value.pipeline.return_value

    layer.write_cache("alias", [record], 3600)
    assert pipe.setex.call_args.args[0] == "entity:alias"
    assert pipe.set.call_count == 0

    layer.write_cache("alias", [record], 0)
    assert pipe.set.call_args.args[0] == "entity:alias"


def test_builder_forwards_redis_password():
    from glinker.core.builders import ConfigBuilder

    builder = ConfigBuilder(name="test")
    builder.l2.add("redis", host="cache.internal", port=6379, db=1, password="secret")
    assert builder._l2_layers[0]["config"] == {
        "host": "cache.internal",
        "port": 6379,
        "db": 1,
        "password": "secret",
    }


def test_postgres_chunked_batch_isolates_a_failing_chunk():
    from glinker.l2.component import PostgresLayer

    with patch.object(PostgresLayer, "_setup", return_value=None):
        layer = PostgresLayer(
            LayerConfig(
                type="postgres", priority=0, search_mode=["exact"], config={"dsn": "service=test"}
            )
        )
    layer.statement_timeout_ms = 2000  # -> _chunk_size() == 2

    one = DatabaseRecord(entity_id="Q1", label="one")
    two = DatabaseRecord(entity_id="Q2", label="two")
    layer._batch_retrieve = MagicMock(
        side_effect=[
            [[one], [two]],
            RuntimeError("simulated statement timeout"),
        ]
    )

    results, failed_indices = layer._batch_retrieve_chunked(["a", "b", "c", "d"], fuzzy=False)

    assert results == [[one], [two], [], []]
    assert set(failed_indices) == {2, 3}
    assert all(
        failure.phase == "phrase" and failure.kind == "query_error"
        for failure in failed_indices.values()
    )
    assert layer._batch_retrieve.call_count == 2
    assert layer._batch_retrieve.call_args_list[0].args == (["a", "b"],)
    assert layer._batch_retrieve.call_args_list[0].kwargs == {"fuzzy": False}
    assert layer._batch_retrieve.call_args_list[1].args == (["c", "d"],)


def test_postgres_search_many_skips_fuzzy_retry_for_failed_chunk_mentions():
    """A mention whose exact-phase chunk raised must not be re-issued as a
    fuzzy query: that mention's empty result is "unknown" (the chunk never
    actually ran), not a genuine miss, and retrying it with the strictly
    more expensive fuzzy path would double the cost against a connection
    that may have just timed out - the exact regression this fix targets.
    """
    with patch.object(PostgresLayer, "_setup", return_value=None):
        layer = PostgresLayer(
            LayerConfig(
                type="postgres",
                priority=0,
                search_mode=["exact", "fuzzy"],
                config={"dsn": "service=test"},
            )
        )
    layer._normalize_many = MagicMock(return_value=["one", "two"])

    one = DatabaseRecord(entity_id="Q1", label="one")
    # "one" resolves normally; "two" lands in a direct-search chunk that
    # failed, so index 1 is unknown rather than a genuine miss.
    layer._batch_retrieve_direct_chunked = MagicMock(
        return_value=(
            [[one], []],
            {1: RetrievalFailure(layer="PostgresLayer", phase="direct", kind="timeout")},
        )
    )
    layer._batch_retrieve_chunked = MagicMock()

    assert layer.search_many(["One", "two"]) == [[one], []]

    # Neither the phrase nor fuzzy phase should retry "two".
    layer._batch_retrieve_chunked.assert_not_called()


def test_postgres_overwrite_invalidates_changed_label_embeddings():
    with patch.object(PostgresLayer, "_setup", return_value=None):
        layer = PostgresLayer(
            LayerConfig(type="postgres", priority=0, config={"dsn": "service=test"})
        )
    layer.conn = MagicMock()
    cursor = layer.conn.cursor.return_value
    with patch("glinker.l2.component.execute_batch") as execute_batch:
        layer.load_bulk(
            [DatabaseRecord(entity_id="Q1", label="New", description="New description")],
            overwrite=True,
        )

    entity_upsert = execute_batch.call_args_list[0].args[1]
    assert "THEN NULL ELSE entities.embedding END" in entity_upsert
    assert "THEN NULL ELSE entities.embedding_model_id END" in entity_upsert
    cursor.close.assert_called_once()
