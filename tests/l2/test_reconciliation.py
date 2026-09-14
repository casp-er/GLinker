"""Regressions for the selectively reconciled upstream L2 changes."""

import json
from unittest.mock import MagicMock, patch

from glinker.l2.component import DatabaseChainComponent, RedisLayer, ElasticsearchLayer, PostgresLayer
from glinker.l2.models import DatabaseRecord, LayerConfig, L2Config


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
    first.search_many.return_value = [[cached]]  # Short backend response.
    second.search_many.return_value = [[fetched]]
    chain.layers = [first, second]
    chain._cache_write = MagicMock()
    result = chain.batch_search(["one", "two"])
    assert result == [[cached], [fetched]]
    second.search_many.assert_called_once_with(["two"])
    assert chain._cache_write.call_count == 2


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
    two = DatabaseRecord(entity_id="Q2", label="two")
    layer._normalize_many = MagicMock(return_value=["one", "two", "one"])
    layer._batch_retrieve = MagicMock(side_effect=[[[one], []], [[two]]])

    assert layer.search_many(["One", "two", "one"]) == [[one], [two], [one]]

    exact_call, fuzzy_call = layer._batch_retrieve.call_args_list
    assert exact_call.args == (["one", "two"],)
    assert exact_call.kwargs == {"fuzzy": False}
    assert fuzzy_call.args == (["two"],)
    assert fuzzy_call.kwargs == {"fuzzy": True}


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
            LayerConfig(type="postgres", priority=0, search_mode=["exact"], config={"dsn": "service=test"})
        )
    layer.statement_timeout_ms = 400  # -> _chunk_size() == 2

    one = DatabaseRecord(entity_id="Q1", label="one")
    two = DatabaseRecord(entity_id="Q2", label="two")
    layer._batch_retrieve = MagicMock(side_effect=[
        [[one], [two]],
        RuntimeError("simulated statement timeout"),
    ])

    result = layer._batch_retrieve_chunked(["a", "b", "c", "d"], fuzzy=False)

    assert result == [[one], [two], [], []]
    assert layer._batch_retrieve.call_count == 2
    assert layer._batch_retrieve.call_args_list[0].args == (["a", "b"],)
    assert layer._batch_retrieve.call_args_list[0].kwargs == {"fuzzy": False}
    assert layer._batch_retrieve.call_args_list[1].args == (["c", "d"],)
