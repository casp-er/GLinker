from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, Any, Set, Union
from pathlib import Path
import re
import redis
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk as es_bulk
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch

from glinker.core.base import BaseComponent
from .models import (
    L2Config,
    LayerConfig,
    FuzzyConfig,
    DatabaseRecord,
    RetrievalFailure,
    MentionRetrieval,
)


def _classify_retrieval_error(error: Exception) -> str:
    """Map a backend exception to a RetrievalFailure kind.

    QueryCanceled is a subclass of OperationalError, so it must be tested
    first; everything else that psycopg2 reports about the connection or
    server state is an availability problem rather than a query bug.
    """
    from psycopg2 import errors as pg_errors

    if isinstance(error, pg_errors.QueryCanceled):
        return "timeout"
    if isinstance(error, (pg_errors.InterfaceError, pg_errors.OperationalError)):
        return "unavailable"
    return "query_error"


class DatabaseLayer(ABC):
    """Base class for all database layers"""

    def __init__(self, config: LayerConfig):
        self.config = config
        self.priority = config.priority
        self.ttl = config.ttl
        self.write = config.write
        self.cache_policy = config.cache_policy
        self.field_mapping = config.field_mapping
        self.fuzzy_config = config.fuzzy or FuzzyConfig()
        self._setup()

    @abstractmethod
    def _setup(self):
        """Initialize layer resources"""
        pass

    def normalize_query(self, query: str) -> str:
        """Normalize query for search"""
        return query.lower().strip()

    @abstractmethod
    def search(self, query: str) -> List[DatabaseRecord]:
        """Exact search"""
        pass

    def search_many(self, queries: List[str]) -> List[List[DatabaseRecord]]:
        """Search multiple queries, preserving query order.

        Layers with a native bulk-search API should override this method.
        The default keeps existing layer implementations correct.
        """
        results = []
        for query in queries:
            found = self.search(query) if "exact" in self.config.search_mode else []
            if not found and "fuzzy" in self.config.search_mode and self.supports_fuzzy():
                found = self.search_fuzzy(query)
            results.append(found)
        return results

    def search_many_detailed(self, queries: List[str]) -> List[MentionRetrieval]:
        """Search multiple queries, keeping operational failures explicit.

        Default wraps search_many: candidates without failures. Layers that
        can distinguish "confirmed miss" from "backend outage" should
        override this and report RetrievalFailure entries.
        """
        return [MentionRetrieval(candidates=candidates) for candidates in self.search_many(queries)]

    @abstractmethod
    def search_fuzzy(self, query: str) -> List[DatabaseRecord]:
        """Fuzzy search"""
        pass

    def supports_fuzzy(self) -> bool:
        """Check if layer supports fuzzy search"""
        return self.fuzzy_config is not None

    @abstractmethod
    def write_cache(self, key: str, records: List[DatabaseRecord], ttl: int):
        """Write records to cache"""
        pass

    def write_cache_many(self, entries: List[tuple[str, List[DatabaseRecord]]], ttl: int):
        """Write multiple query results, with a correct per-entry fallback."""
        for key, records in entries:
            self.write_cache(key, records, ttl)

    @abstractmethod
    def is_available(self) -> bool:
        """Check if layer is available"""
        pass

    @abstractmethod
    def load_bulk(
        self, entities: List[DatabaseRecord], overwrite: bool = False, batch_size: int = 1000
    ) -> int:
        """Bulk load entities"""
        pass

    def clear(self):
        """Clear all data in layer"""
        pass

    def count(self) -> int:
        """Count entities in layer"""
        return 0

    def get_all_entities(self) -> List[DatabaseRecord]:
        """Get all entities from layer (for precompute)"""
        return []

    def update_embeddings(
        self, entity_ids: List[str], embeddings: List[List[float]], model_id: str
    ) -> int:
        """Update embeddings for entities"""
        return 0

    def map_to_record(self, raw_data: Dict[str, Any]) -> DatabaseRecord:
        """Map raw data to DatabaseRecord using field_mapping"""
        mapped = {}
        for standard_field, db_field in self.field_mapping.items():
            if db_field in raw_data:
                mapped[standard_field] = raw_data[db_field]

        # Handle embedding fields directly (not in field_mapping)
        if "embedding" in raw_data:
            mapped["embedding"] = raw_data["embedding"]
        if "embedding_model_id" in raw_data:
            mapped["embedding_model_id"] = raw_data["embedding_model_id"]

        mapped["source"] = self.config.type
        return DatabaseRecord(**mapped)


class DictLayer(DatabaseLayer):
    """Simple dict-based storage for small entity sets (<5000)"""

    def _setup(self):
        self._storage: Dict[str, DatabaseRecord] = {}
        self._label_index: Dict[str, str] = {}
        self._alias_index: Dict[str, Set[str]] = {}

    def search(self, query: str) -> List[DatabaseRecord]:
        """Fast O(1) exact search using indexes"""
        query_key = self.normalize_query(query)
        results = []
        seen = set()

        # Label lookup
        if query_key in self._label_index:
            eid = self._label_index[query_key]
            results.append(self._storage[eid])
            seen.add(eid)

        # Alias lookup
        if query_key in self._alias_index:
            for eid in self._alias_index[query_key]:
                if eid not in seen:
                    results.append(self._storage[eid])
                    seen.add(eid)

        return results

    def search_fuzzy(self, query: str) -> List[DatabaseRecord]:
        """Simple fuzzy search for small datasets (O(n) is fine for <5000 entities)"""
        try:
            from rapidfuzz import fuzz
        except ImportError:
            print("[WARN DictLayer] rapidfuzz not installed, fuzzy search disabled")
            return []

        query_key = self.normalize_query(query)
        results = []

        # Check prefix requirement
        if self.fuzzy_config.prefix_length > 0:
            prefix = query_key[: self.fuzzy_config.prefix_length]

        for entity in self._storage.values():
            # Check label
            label_key = entity.label.lower()

            if self.fuzzy_config.prefix_length > 0:
                if not label_key.startswith(prefix):
                    continue

            similarity = fuzz.ratio(query_key, label_key) / 100.0
            if similarity >= self.fuzzy_config.min_similarity:
                results.append((entity, similarity))
                continue

            # Check aliases
            for alias in entity.aliases:
                alias_key = alias.lower()
                if self.fuzzy_config.prefix_length > 0:
                    if not alias_key.startswith(prefix):
                        continue

                sim = fuzz.ratio(query_key, alias_key) / 100.0
                if sim >= self.fuzzy_config.min_similarity:
                    results.append((entity, sim))
                    break

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results]

    def write_cache(self, key: str, records: List[DatabaseRecord], ttl: int):
        """Write is same as load_bulk for dict layer"""
        self.load_bulk(records, overwrite=True)

    def load_bulk(
        self, entities: List[DatabaseRecord], overwrite: bool = False, batch_size: int = 1000
    ) -> int:
        """Bulk load entities with indexing"""
        count = 0
        for entity in entities:
            entity_id = entity.entity_id

            if not overwrite and entity_id in self._storage:
                continue

            # Store entity
            self._storage[entity_id] = entity

            # Index by label
            label_key = entity.label.lower()
            self._label_index[label_key] = entity_id

            # Index by aliases
            for alias in entity.aliases:
                alias_key = alias.lower()
                if alias_key not in self._alias_index:
                    self._alias_index[alias_key] = set()
                self._alias_index[alias_key].add(entity_id)

            count += 1
        return count

    def clear(self):
        """Clear all data"""
        self._storage.clear()
        self._label_index.clear()
        self._alias_index.clear()

    def count(self) -> int:
        """Count entities"""
        return len(self._storage)

    def get_all_entities(self) -> List[DatabaseRecord]:
        """Get all entities from storage"""
        return list(self._storage.values())

    def update_embeddings(
        self, entity_ids: List[str], embeddings: List[List[float]], model_id: str
    ) -> int:
        """Update embeddings for entities"""
        count = 0
        for eid, emb in zip(entity_ids, embeddings):
            if eid in self._storage:
                self._storage[eid].embedding = emb
                self._storage[eid].embedding_model_id = model_id
                count += 1
        return count

    def is_available(self) -> bool:
        """Dict layer is always available"""
        return True


class RedisLayer(DatabaseLayer):
    """Redis cache layer with optimized embedding storage.

    Storage structure:
    - entity:{label} -> entity data without embedding (fast lookup)
    - entity:{alias} -> entity data without embedding (fast lookup)
    - entity:emb:{entity_id} -> {embedding, embedding_model_id} (no duplication)

    Benefits:
    - Embeddings stored once per entity (not duplicated across aliases)
    - update_embeddings() is O(M) instead of O(N×M) where:
      - M = entities to update
      - N = total keys in Redis (entities × aliases)
    - Reduced memory usage for large entity databases
    - Faster batch embedding updates
    """

    def _setup(self):
        self.client = redis.Redis(
            host=self.config.config.get("host", "localhost"),
            port=self.config.config.get("port", 6379),
            db=self.config.config.get("db", 0),
            password=self.config.config.get("password"),
            decode_responses=False,
        )

    @staticmethod
    def _cache_set(pipe, key: str, ttl: int, data: str):
        """SETEX with a positive ttl; SET (no expiry) when ttl <= 0.

        LayerConfig.ttl documents 0 as "no expiry", but every call site here
        used to call SETEX unconditionally, which Redis rejects for ttl=0.
        """
        if ttl is not None and ttl > 0:
            pipe.setex(key, ttl, data)
        else:
            pipe.set(key, data)

    def supports_fuzzy(self) -> bool:
        return False

    def _search_records(self, query: str) -> List[DatabaseRecord]:
        query = self.normalize_query(query)
        key = f"entity:{query}"

        try:
            data = self.client.get(key)
            if data:
                if isinstance(data, bytes):
                    data = data.decode("utf-8")

                records_data = json.loads(data)

                if isinstance(records_data, list):
                    results = []
                    for r in records_data:
                        if isinstance(r, dict):
                            r["source"] = "redis"
                            results.append(DatabaseRecord(**r))
                        else:
                            results.append(r)
                    return results

                elif isinstance(records_data, dict):
                    records_data["source"] = "redis"
                    return [DatabaseRecord(**records_data)]

        except Exception as e:
            print(f"[ERROR Redis] Search error: {e}")

        return []

    @staticmethod
    def _decode_records(data) -> List[DatabaseRecord]:
        """Decode one cached query result without issuing Redis commands."""
        if not data:
            return []
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        records_data = json.loads(data)
        if isinstance(records_data, dict):
            records_data = [records_data]
        if not isinstance(records_data, list):
            return []

        records = []
        for record_data in records_data:
            if isinstance(record_data, dict):
                record_data["source"] = "redis"
                records.append(DatabaseRecord(**record_data))
            elif isinstance(record_data, DatabaseRecord):
                records.append(record_data)
        return records

    def search(self, query: str) -> List[DatabaseRecord]:
        records = self._search_records(query)
        embeddings = self.get_embeddings_batch([r.entity_id for r in records])
        for record in records:
            if record.entity_id in embeddings:
                data = embeddings[record.entity_id]
                record.embedding = data.get("embedding")
                record.embedding_model_id = data.get("embedding_model_id")
        return records

    def search_many(self, queries: List[str]) -> List[List[DatabaseRecord]]:
        """Fetch all query keys and embeddings in two Redis round trips."""
        if not queries:
            return []

        normalized = [self.normalize_query(query) for query in queries]
        unique = list(dict.fromkeys(normalized))
        try:
            pipe = self.client.pipeline()
            for query in unique:
                pipe.get(f"entity:{query}")
            payloads = pipe.execute()

            records_by_query = {}
            entity_ids = []
            seen_ids = set()
            for query, payload in zip(unique, payloads):
                records = self._decode_records(payload)
                records_by_query[query] = records
                for record in records:
                    if record.entity_id not in seen_ids:
                        seen_ids.add(record.entity_id)
                        entity_ids.append(record.entity_id)

            embeddings = self.get_embeddings_batch(entity_ids)
            for records in records_by_query.values():
                for record in records:
                    if record.entity_id in embeddings:
                        data = embeddings[record.entity_id]
                        record.embedding = data.get("embedding")
                        record.embedding_model_id = data.get("embedding_model_id")
            return [records_by_query.get(query, []) for query in normalized]
        except Exception as e:
            print(f"[ERROR Redis] Batched search error: {e}")
            return [[] for _ in queries]

    def search_fuzzy(self, query: str) -> List[DatabaseRecord]:
        return []

    def write_cache(self, key: str, records: List[DatabaseRecord], ttl: int):
        """Write search results to cache.

        Optimized: embeddings stored separately to avoid duplication.
        """
        key = self.normalize_query(key)
        cache_key = f"entity:{key}"

        try:
            # Store entity data WITHOUT embeddings
            records_data = []
            pipe = self.client.pipeline()

            for r in records:
                r_dict = r.dict()
                embedding = r_dict.pop("embedding", None)
                embedding_model_id = r_dict.pop("embedding_model_id", None)
                records_data.append(r_dict)

                # Store embedding separately if present
                if embedding is not None:
                    emb_key = f"entity:emb:{r.entity_id}"
                    emb_data = json.dumps(
                        {"embedding": embedding, "embedding_model_id": embedding_model_id}
                    )
                    self._cache_set(pipe, emb_key, ttl, emb_data)

            # Store main cache data
            data = json.dumps(records_data)
            self._cache_set(pipe, cache_key, ttl, data)
            pipe.execute()

        except Exception as e:
            print(f"[ERROR Redis] Write error: {e}")

    def write_cache_many(self, entries: List[tuple[str, List[DatabaseRecord]]], ttl: int):
        """Write all cache results in one Redis pipeline execution."""
        if not entries:
            return
        try:
            pipe = self.client.pipeline()
            for key, records in entries:
                cache_key = f"entity:{self.normalize_query(key)}"
                records_data = []
                for record in records:
                    record_data = record.dict()
                    embedding = record_data.pop("embedding", None)
                    embedding_model_id = record_data.pop("embedding_model_id", None)
                    records_data.append(record_data)
                    if embedding is not None:
                        self._cache_set(
                            pipe,
                            f"entity:emb:{record.entity_id}",
                            ttl,
                            json.dumps(
                                {
                                    "embedding": embedding,
                                    "embedding_model_id": embedding_model_id,
                                }
                            ),
                        )
                self._cache_set(pipe, cache_key, ttl, json.dumps(records_data))
            pipe.execute()
        except Exception as e:
            print(f"[ERROR Redis] Batched write error: {e}")

    def load_bulk(
        self, entities: List[DatabaseRecord], overwrite: bool = False, batch_size: int = 1000
    ) -> int:
        """Bulk load to Redis.

        Optimized: embeddings stored separately to avoid duplication across aliases.
        Structure:
        - entity:{label/alias} -> entity data WITHOUT embedding
        - entity:emb:{entity_id} -> {embedding, model_id}
        """
        count = 0
        pipe = self.client.pipeline()

        for entity in entities:
            # Prepare data WITHOUT embedding (store separately)
            entity_data = entity.dict()
            embedding = entity_data.pop("embedding", None)
            embedding_model_id = entity_data.pop("embedding_model_id", None)
            data_json = json.dumps(entity_data)

            # Store by label
            label_key = f"entity:{entity.label.lower()}"
            if overwrite or not self.client.exists(label_key):
                self._cache_set(pipe, label_key, self.ttl, data_json)
                count += 1

            # Store by aliases
            for alias in entity.aliases:
                alias_key = f"entity:{alias.lower()}"
                if overwrite or not self.client.exists(alias_key):
                    self._cache_set(pipe, alias_key, self.ttl, data_json)

            # Store embedding SEPARATELY (no duplication)
            if embedding is not None:
                emb_key = f"entity:emb:{entity.entity_id}"
                emb_data = json.dumps(
                    {"embedding": embedding, "embedding_model_id": embedding_model_id}
                )
                self._cache_set(pipe, emb_key, self.ttl, emb_data)

            # Execute in batches
            if len(pipe) >= batch_size:
                pipe.execute()
                pipe = self.client.pipeline()

        # Execute remaining
        if len(pipe) > 0:
            pipe.execute()

        return count

    def clear(self):
        """Clear all entity keys (including embeddings)."""
        for key in self.client.scan_iter(match="entity:*"):
            self.client.delete(key)
        # Note: entity:emb:* keys are also matched by entity:* pattern

    def count(self) -> int:
        """Count entity keys."""
        return sum(1 for _ in self.client.scan_iter(match="entity:*"))

    def get_all_entities(self, include_embeddings: bool = False) -> List[DatabaseRecord]:
        """Get all entities from Redis (scans all entity:* keys).

        Args:
            include_embeddings: If True, also fetch embeddings (slower)

        Note: Skips entity:emb:* keys as they're fetched separately if needed.
        """
        entities = []
        seen_ids = set()

        for key in self.client.scan_iter(match="entity:*"):
            # Skip embedding keys - they're handled separately
            if isinstance(key, bytes):
                key_str = key.decode("utf-8")
            else:
                key_str = key

            if key_str.startswith("entity:emb:"):
                continue

            try:
                data = self.client.get(key)
                if data:
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    record_data = json.loads(data)

                    if isinstance(record_data, dict):
                        if record_data.get("entity_id") not in seen_ids:
                            record_data["source"] = "redis"
                            entities.append(DatabaseRecord(**record_data))
                            seen_ids.add(record_data.get("entity_id"))
                    elif isinstance(record_data, list):
                        for r in record_data:
                            if r.get("entity_id") not in seen_ids:
                                r["source"] = "redis"
                                entities.append(DatabaseRecord(**r))
                                seen_ids.add(r.get("entity_id"))
            except Exception:
                continue

        # Optionally fetch embeddings in batch
        if include_embeddings and entities:
            entity_ids = [e.entity_id for e in entities]
            embeddings_map = self.get_embeddings_batch(entity_ids)

            for entity in entities:
                if entity.entity_id in embeddings_map:
                    emb_data = embeddings_map[entity.entity_id]
                    entity.embedding = emb_data.get("embedding")
                    entity.embedding_model_id = emb_data.get("embedding_model_id")

        return entities

    def update_embeddings(
        self, entity_ids: List[str], embeddings: List[List[float]], model_id: str
    ) -> int:
        """Update embeddings in Redis entities.

        Optimized: O(M) direct access instead of O(N×M) full scan.
        Embeddings stored separately at entity:emb:{entity_id}.
        """
        if not entity_ids or not embeddings:
            return 0

        pipe = self.client.pipeline()
        count = 0

        for entity_id, embedding in zip(entity_ids, embeddings):
            emb_key = f"entity:emb:{entity_id}"
            emb_data = json.dumps({"embedding": embedding, "embedding_model_id": model_id})
            self._cache_set(pipe, emb_key, self.ttl, emb_data)
            count += 1

            # Execute in batches of 1000
            if count % 1000 == 0:
                pipe.execute()
                pipe = self.client.pipeline()

        # Execute remaining
        if len(pipe) > 0:
            pipe.execute()

        return count

    def get_embeddings_batch(self, entity_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get embeddings for multiple entities in batch.

        Args:
            entity_ids: List of entity IDs

        Returns:
            Dict mapping entity_id -> {embedding: List[float], embedding_model_id: str}
            Missing entities are omitted from result.
        """
        if not entity_ids:
            return {}

        # Batch GET using pipeline
        pipe = self.client.pipeline()
        for entity_id in entity_ids:
            emb_key = f"entity:emb:{entity_id}"
            pipe.get(emb_key)

        results = pipe.execute()

        # Parse results
        embeddings_map = {}
        for entity_id, data in zip(entity_ids, results):
            if data:
                try:
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    emb_data = json.loads(data)
                    embeddings_map[entity_id] = emb_data
                except Exception:
                    continue

        return embeddings_map

    def is_available(self) -> bool:
        try:
            self.client.ping()
            return True
        except:
            return False


class ElasticsearchLayer(DatabaseLayer):
    """Elasticsearch full-text search layer"""

    def _setup(self):
        self.client = Elasticsearch(
            self.config.config["hosts"], api_key=self.config.config.get("api_key")
        )
        self.index_name = self.config.config["index_name"]
        self.popularity_boost = self.config.config.get("popularity_boost", False)

    def _build_query(self, match_query: dict) -> dict:
        """Wrap a match query with optional popularity boosting.

        When popularity_boost is enabled, wraps the query in a function_score
        that multiplies BM25 relevance by ln(2 + popularity). This brings ES
        in line with PostgresLayer, which uses ORDER BY popularity DESC.

        Uses ln2p (not ln1p) to avoid zeroing out entities with popularity=0,
        since ln(2+0)=0.69 while ln(1+0)=0.
        """
        if not self.popularity_boost:
            return {"query": match_query, "size": 50}

        return {
            "query": {
                "function_score": {
                    "query": match_query,
                    "field_value_factor": {"field": "popularity", "modifier": "ln2p", "missing": 1},
                    "boost_mode": "multiply",
                }
            },
            "size": 50,
        }

    def search(self, query: str) -> List[DatabaseRecord]:
        query = self.normalize_query(query)

        try:
            exact = self._search_exact(query)
            if exact or "fuzzy" not in self.config.search_mode:
                return exact
            return self._search_fuzzy(query)
        except Exception as e:
            print(f"[ERROR ES] Search error: {e}")
            return []

    def _match_query(self, query: str, fuzzy: bool = False) -> dict:
        match = {
            "multi_match": {
                "query": query,
                "fields": ["label^2", "aliases^1.5", "description"],
            }
        }
        if fuzzy:
            match["multi_match"].update(
                {
                    "fuzziness": self.fuzzy_config.max_distance,
                    "prefix_length": self.fuzzy_config.prefix_length,
                    "max_expansions": 50,
                }
            )
        else:
            match["multi_match"]["type"] = "best_fields"
        return match

    def _search_exact(self, query: str) -> List[DatabaseRecord]:
        response = self.client.search(
            index=self.index_name,
            body=self._build_query(self._match_query(query)),
        )
        return self._process_hits(response["hits"]["hits"])

    def _search_fuzzy(self, query: str) -> List[DatabaseRecord]:
        response = self.client.search(
            index=self.index_name,
            body=self._build_query(self._match_query(query, fuzzy=True)),
        )
        return self._process_hits(response["hits"]["hits"])

    def _msearch(self, queries: List[tuple[str, bool]]) -> List[List[DatabaseRecord]]:
        """Run exact or fuzzy searches in one Elasticsearch request."""
        if not queries:
            return []

        searches = []
        for query, fuzzy in queries:
            searches.extend([{}, self._build_query(self._match_query(query, fuzzy=fuzzy))])

        response = self.client.msearch(index=self.index_name, searches=searches)
        return [
            self._process_hits(result.get("hits", {}).get("hits", []))
            for result in response["responses"]
        ]

    def search_many(self, queries: List[str]) -> List[List[DatabaseRecord]]:
        """Search unique mentions in batched exact-then-fuzzy requests.

        Exact matches are preferred. Fuzzy lookup is only issued for mentions
        with no exact result, avoiding the previous two sequential requests per
        mention while preserving fuzzy fallback behavior.
        """
        if not queries:
            return []

        normalized = [self.normalize_query(query) for query in queries]
        unique = list(dict.fromkeys(normalized))
        results_by_query = {query: [] for query in unique}

        try:
            if "exact" in self.config.search_mode:
                exact_results = self._msearch([(query, False) for query in unique])
                for query, results in zip(unique, exact_results):
                    results_by_query[query] = results

            missing = [
                query
                for query in unique
                if not results_by_query[query] and "fuzzy" in self.config.search_mode
            ]
            if missing:
                fuzzy_results = self._msearch([(query, True) for query in missing])
                for query, results in zip(missing, fuzzy_results):
                    results_by_query[query] = results
        except Exception as e:
            print(f"[ERROR ES] Batched search error: {e}")
            return [[] for _ in queries]

        return [results_by_query[query] for query in normalized]

    def search_fuzzy(self, query: str) -> List[DatabaseRecord]:
        query = self.normalize_query(query)
        fuzzy_distance = self.fuzzy_config.max_distance

        try:
            match_query = {
                "multi_match": {
                    "query": query,
                    "fields": ["label^2", "aliases^1.5", "description"],
                    "fuzziness": fuzzy_distance,
                    "prefix_length": self.fuzzy_config.prefix_length,
                    "max_expansions": 50,
                }
            }
            body = self._build_query(match_query)
            response = self.client.search(index=self.index_name, body=body)
            return self._process_hits(response["hits"]["hits"])
        except Exception as e:
            print(f"[ERROR ES] Fuzzy error: {e}")
            return []

    def batch_search(self, queries: List[str], fuzzy: bool = False) -> List[List[DatabaseRecord]]:
        """Upstream-compatible explicit-mode batch API (no fallback)."""
        normalized = [self.normalize_query(q) for q in queries]
        return self._msearch([(q, fuzzy) for q in normalized])

    def _process_hits(self, hits: List[Dict]) -> List[DatabaseRecord]:
        records = []
        for hit in hits:
            source = hit["_source"]
            source["_id"] = hit["_id"]
            source["source"] = "elasticsearch"
            record = self.map_to_record(source)
            records.append(record)
        return records

    def write_cache(self, key: str, records: List[DatabaseRecord], ttl: int):
        if not records:
            return

        try:
            actions = []
            for record in records:
                doc = self._map_from_record(record)
                actions.append({"_index": self.index_name, "_id": record.entity_id, "_source": doc})

            if actions:
                es_bulk(self.client, actions)
                self.client.indices.refresh(index=self.index_name)
        except Exception as e:
            print(f"[ERROR ES] Write error: {e}")

    def load_bulk(
        self, entities: List[DatabaseRecord], overwrite: bool = False, batch_size: int = 1000
    ) -> int:
        """Bulk load to Elasticsearch"""
        actions = []
        for entity in entities:
            doc = self._map_from_record(entity)

            action = {"_index": self.index_name, "_id": entity.entity_id, "_source": doc}

            if overwrite:
                action["_op_type"] = "index"
            else:
                action["_op_type"] = "create"

            actions.append(action)

        success, failed = es_bulk(self.client, actions, raise_on_error=False, chunk_size=batch_size)

        self.client.indices.refresh(index=self.index_name)
        return success

    def _map_from_record(self, record: DatabaseRecord) -> dict:
        """Map DatabaseRecord -> ES document using field_mapping"""
        reverse_mapping = {v: k for k, v in self.field_mapping.items()}

        doc = {}
        for standard_field, value in record.dict().items():
            if standard_field == "source":
                continue

            es_field = reverse_mapping.get(standard_field, standard_field)
            doc[es_field] = value

        return doc

    def clear(self):
        """Delete all documents in index"""
        try:
            self.client.delete_by_query(index=self.index_name, body={"query": {"match_all": {}}})
            self.client.indices.refresh(index=self.index_name)
        except Exception as e:
            print(f"[ERROR ES] Clear error: {e}")

    def count(self) -> int:
        """Count documents in index"""
        try:
            result = self.client.count(index=self.index_name)
            return result["count"]
        except:
            return 0

    def get_all_entities(self) -> List[DatabaseRecord]:
        """Get all entities from Elasticsearch using scroll"""
        entities = []

        try:
            # Use scroll API for large datasets
            response = self.client.search(
                index=self.index_name, body={"query": {"match_all": {}}, "size": 1000}, scroll="2m"
            )

            scroll_id = response["_scroll_id"]
            hits = response["hits"]["hits"]

            while hits:
                entities.extend(self._process_hits(hits))

                response = self.client.scroll(scroll_id=scroll_id, scroll="2m")
                scroll_id = response["_scroll_id"]
                hits = response["hits"]["hits"]

            # Clear scroll
            self.client.clear_scroll(scroll_id=scroll_id)

        except Exception as e:
            print(f"[ERROR ES] get_all_entities error: {e}")

        return entities

    def update_embeddings(
        self, entity_ids: List[str], embeddings: List[List[float]], model_id: str
    ) -> int:
        """Update embeddings in Elasticsearch"""
        try:
            actions = []
            for eid, emb in zip(entity_ids, embeddings):
                actions.append(
                    {
                        "_op_type": "update",
                        "_index": self.index_name,
                        "_id": eid,
                        "doc": {"embedding": emb, "embedding_model_id": model_id},
                    }
                )

            success, failed = es_bulk(self.client, actions, raise_on_error=False, chunk_size=500)

            self.client.indices.refresh(index=self.index_name)
            return success

        except Exception as e:
            print(f"[ERROR ES] update_embeddings error: {e}")
            return 0

    def is_available(self) -> bool:
        try:
            return self.client.ping()
        except:
            return False


class PostgresLayer(DatabaseLayer):
    """PostgreSQL database layer"""

    # Class-level fallback matching _setup's config default, so
    # _chunk_size() has a safe value even if a caller (or a test that
    # patches out _setup) never runs _setup's config-derived assignment.
    statement_timeout_ms = 5000
    # pg_trgm cannot extract useful trigrams from very short mentions. The
    # word-boundary regex and similarity operators then devolve into broad
    # scans on a production-sized KB (for example, "us" scans every row).
    _MIN_TRIGRAM_QUERY_LENGTH = 4

    def _setup(self):
        from psycopg2 import sql

        cfg = self.config.config
        if not cfg.get("dsn") and not cfg.get("database"):
            raise ValueError("PostgresLayer requires an explicit dsn or database")
        self.conn = psycopg2.connect(
            cfg.get("dsn", ""),
            **{
                key: value
                for key, value in cfg.items()
                if key
                in {"host", "port", "database", "user", "password", "sslmode", "connect_timeout"}
            },
        )
        self.schema = cfg.get("schema", "glinker")
        self.statement_timeout_ms = int(cfg.get("statement_timeout_ms", 5000))
        # No DDL on connection. Provision explicitly with scripts/init_postgres.py.
        with self.conn:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.schema))
                )
                for name in ("entities", "aliases"):
                    cursor.execute(
                        "SELECT to_regclass(%s)",
                        (sql.Identifier(self.schema, name).as_string(self.conn),),
                    )
                    if cursor.fetchone()[0] is None:
                        raise ValueError(f"GLiNKER schema {self.schema!r} is not provisioned")

    def normalize_query(self, query: str) -> str:
        # The database function is shared by ingestion triggers and queries.
        with self.conn:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT glinker_fold(%s)", (query,))
                return cursor.fetchone()[0]

    def search(self, query: str) -> List[DatabaseRecord]:
        return self._retrieve(query, fuzzy=False)

    def search_fuzzy(self, query: str) -> List[DatabaseRecord]:
        return self._retrieve(query, fuzzy=True)

    def _retrieve(self, query: str, fuzzy: bool) -> List[DatabaseRecord]:
        query = self.normalize_query(query)
        if not query:
            return []
        # Separate indexed branches prevent an OR across the alias join from
        # turning candidate discovery into a full entity-table scan.
        if fuzzy:
            predicate = (
                "{field} %% %(query)s AND length({field}) <= 255 "
                "AND length(%(query)s) <= 255 "
                "AND abs(length({field}) - length(%(query)s)) <= %(distance)s "
                "AND levenshtein_less_equal(left({field}, 255), left(%(query)s, 255), "
                "%(distance)s) <= %(distance)s"
            )
            score = "similarity({field}, %(query)s)"
        else:
            predicate = "{field} ~ %(pattern)s"
            score = "1.0"
        branches = []
        for table, field, weight in [
            ("entities", "label_folded", 2.0),
            ("aliases", "alias_folded", 1.5),
            ("entities", "description_folded", 1.0),
        ]:
            branches.append(
                f"SELECT entity_id, {weight} * {score.format(field=field)} AS score "
                f"FROM {table} WHERE {predicate.format(field=field)}"
            )
        statement = """
            WITH matches AS ( %s ), ranked AS (
                SELECT entity_id, max(score) AS score FROM matches GROUP BY entity_id
            ), candidates AS (
                SELECT e.*, r.score FROM ranked r JOIN entities e USING (entity_id)
                ORDER BY r.score DESC, e.popularity DESC, e.entity_id LIMIT 50
            )
            SELECT c.*, ARRAY(SELECT a.alias FROM aliases a
                WHERE a.entity_id = c.entity_id ORDER BY a.alias) AS aliases
            FROM candidates c ORDER BY c.score DESC, c.popularity DESC, c.entity_id
        """ % " UNION ALL ".join(branches)
        pattern = self._exact_pattern(query)
        with self.conn:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (str(self.statement_timeout_ms),),
                )
                cursor.execute(
                    "SELECT set_config('pg_trgm.similarity_threshold', %s, true)",
                    (str(self.fuzzy_config.min_similarity),),
                )
                cursor.execute(
                    statement,
                    {
                        "query": query,
                        "pattern": pattern,
                        "distance": self.fuzzy_config.max_distance,
                    },
                )
                return self._process_rows(cursor.fetchall())

    def _exact_pattern(self, query: str) -> str:
        """Word-bounded, escaped regex pattern for exact retrieval."""
        pattern = (r"\m" if query[0].isalnum() else "") + re.escape(query)
        pattern += r"\M" if query[-1].isalnum() else ""
        return pattern

    def _normalize_many(self, queries: List[str]) -> List[str]:
        """Fold many queries with the shared DB function in one round trip."""
        with self.conn:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    "SELECT glinker_fold(q) FROM unnest(%(queries)s::text[]) "
                    "WITH ORDINALITY AS t(q, ord) ORDER BY ord",
                    {"queries": queries},
                )
                return [row[0] for row in cursor.fetchall()]

    def search_many(self, queries: List[str]) -> List[List[DatabaseRecord]]:
        """Resolve many mentions, returning only candidate lists."""
        return [retrieval.candidates for retrieval in self.search_many_detailed(queries)]

    def search_many_detailed(self, queries: List[str]) -> List[MentionRetrieval]:
        """Resolve many mentions in a bounded number of round trips.

        The base class default (DatabaseLayer.search_many) issues a full
        search()/search_fuzzy() round trip per mention — a full database
        round trip for every single entity mention, serially. This override
        normalizes, equality-searches, and fuzzy-searches (only for exact
        misses) each in a single batched statement, mirroring
        ElasticsearchLayer._msearch.

        Unlike a plain candidate list, the result distinguishes a confirmed
        miss (no candidates, no failures) from an operational failure (a
        phase that raised, e.g. a statement_timeout cancellation). Failed
        mentions are NOT retried in later phases — that would turn one
        failed chunk into a sequence of timeouts against a connection that
        is already saturated, exactly the overload scenario chunking exists
        to contain.
        """
        if not queries:
            return []

        try:
            normalized = self._normalize_many(queries)
            unique = list(dict.fromkeys(q for q in normalized if q))
            outcomes: Dict[str, MentionRetrieval] = {query: MentionRetrieval() for query in unique}
            failed: Set[str] = set()

            if unique and "exact" in self.config.search_mode:
                # Resolve the common case first with indexed equality over
                # labels and aliases. Besides being cheaper, this keeps short
                # mentions such as US/UK/FT away from regex scans that cannot
                # usefully exploit a trigram index.
                direct_results, direct_failures = self._batch_retrieve_direct_chunked(unique)
                for idx, (query, records) in enumerate(zip(unique, direct_results)):
                    outcomes[query].candidates = records
                    failure = direct_failures.get(idx)
                    if failure is not None:
                        failed.add(query)
                        outcomes[query].failures.append(failure)

                phrase_queries = [
                    query
                    for query in unique
                    if not outcomes[query].candidates
                    and query not in failed
                    and len(query) >= self._MIN_TRIGRAM_QUERY_LENGTH
                ]
                if phrase_queries:
                    phrase_results, phrase_failures = self._batch_retrieve_chunked(
                        phrase_queries, fuzzy=False, phase="phrase"
                    )
                    for idx, (query, records) in enumerate(zip(phrase_queries, phrase_results)):
                        outcomes[query].candidates = records
                        failure = phrase_failures.get(idx)
                        if failure is not None:
                            failed.add(query)
                            outcomes[query].failures.append(failure)

            missing = [
                query
                for query in unique
                if not outcomes[query].candidates
                and query not in failed
                and len(query) >= self._MIN_TRIGRAM_QUERY_LENGTH
                and "fuzzy" in self.config.search_mode
                and self.supports_fuzzy()
            ]
            if missing:
                fuzzy_results, fuzzy_failures = self._batch_retrieve_chunked(
                    missing, fuzzy=True, phase="fuzzy"
                )
                for idx, (query, records) in enumerate(zip(missing, fuzzy_results)):
                    outcomes[query].candidates = records
                    failure = fuzzy_failures.get(idx)
                    if failure is not None:
                        outcomes[query].failures.append(failure)
        except Exception as e:
            # Normalization (or anything outside the per-chunk guards) blew
            # up: every mention is unknown rather than a confirmed miss.
            print(f"[ERROR Postgres] Batched search error: {e}")
            failure = RetrievalFailure(
                layer=type(self).__name__,
                phase="normalize",
                kind=_classify_retrieval_error(e),
                message=str(e),
            )
            return [MentionRetrieval(failures=[failure]) for _ in queries]

        return [
            outcomes.get(query, MentionRetrieval()) if query else MentionRetrieval()
            for query in normalized
        ]

    # Conservative estimate of per-mention SQL cost, used only to size
    # batches defensively against statement_timeout. Real cost varies with
    # corpus size and popularity distribution; measured cost against a real
    # 50K-entity corpus was ~80-90ms/mention, so this leaves real margin.
    _ASSUMED_MS_PER_MENTION = 1000

    def _chunk_size(self) -> int:
        return max(1, self.statement_timeout_ms // self._ASSUMED_MS_PER_MENTION)

    def _batch_retrieve_chunked(
        self, queries: List[str], fuzzy: bool, phase: str | None = None
    ) -> "tuple[List[List[DatabaseRecord]], Dict[int, RetrievalFailure]]":
        """Splits a large query list into statement_timeout-safe chunks.

        A single _batch_retrieve call costs roughly N * per-mention SQL
        work; above the timeout-derived chunk size this can exceed
        statement_timeout and cancel the WHOLE batch, not just the slow
        part. Chunking keeps each statement inside a safe margin, and a
        chunk that still fails only empties that chunk's mentions rather
        than the entire search_many call.

        Returns (results, failures): results is a flat list aligned
        with `queries` (a failed chunk contributes `[]` per mention, same
        as before); failures maps the positions in `queries` whose
        chunk raised to the operational failure that emptied them.
        Callers must treat those positions as "unknown", not
        "confirmed no match" - e.g. search_many_detailed uses this to skip
        later phases for mentions whose earlier chunk already failed,
        rather than re-issuing a more expensive query against a connection
        that may just have timed out.
        """
        phase = phase or ("fuzzy" if fuzzy else "phrase")
        chunk_size = self._chunk_size()
        results: List[List[DatabaseRecord]] = []
        failures: Dict[int, RetrievalFailure] = {}
        for i in range(0, len(queries), chunk_size):
            chunk = queries[i : i + chunk_size]
            try:
                results.extend(self._batch_retrieve(chunk, fuzzy=fuzzy))
            except Exception as e:
                print(f"[ERROR Postgres] {phase} chunk search error ({len(chunk)} mentions): {e}")
                results.extend([[] for _ in chunk])
                for position in range(i, i + len(chunk)):
                    failures[position] = RetrievalFailure(
                        layer=type(self).__name__,
                        phase=phase,
                        kind=_classify_retrieval_error(e),
                        message=str(e),
                    )
        return results, failures

    def _batch_retrieve_direct_chunked(
        self, queries: List[str]
    ) -> "tuple[List[List[DatabaseRecord]], Dict[int, RetrievalFailure]]":
        """Run cheap label/alias equality lookups in timeout-sized chunks."""
        chunk_size = self._chunk_size()
        results: List[List[DatabaseRecord]] = []
        failures: Dict[int, RetrievalFailure] = {}
        for i in range(0, len(queries), chunk_size):
            chunk = queries[i : i + chunk_size]
            try:
                results.extend(self._batch_retrieve_direct(chunk))
            except Exception as e:
                print(f"[ERROR Postgres] direct chunk search error ({len(chunk)} mentions): {e}")
                results.extend([[] for _ in chunk])
                for position in range(i, i + len(chunk)):
                    failures[position] = RetrievalFailure(
                        layer=type(self).__name__,
                        phase="direct",
                        kind=_classify_retrieval_error(e),
                        message=str(e),
                    )
        return results, failures

    def _batch_retrieve_direct(self, queries: List[str]) -> List[List[DatabaseRecord]]:
        """Resolve exact label/alias equality matches without broad text scans."""
        statement = """
            WITH input AS (
                SELECT * FROM unnest(%(queries)s::text[])
                    WITH ORDINALITY AS t(query, idx)
            ),
            matches AS (
                SELECT i.idx, m.entity_id, m.score
                FROM input i, LATERAL (
                    SELECT entity_id, 2.0 AS score
                    FROM entities WHERE label_folded = i.query
                    UNION ALL
                    SELECT entity_id, 1.5 AS score
                    FROM aliases WHERE alias_folded = i.query
                ) m
            ),
            ranked AS (
                SELECT idx, entity_id, max(score) AS score FROM matches GROUP BY idx, entity_id
            ),
            candidates AS (
                SELECT idx, e.*, r.score,
                       row_number() OVER (
                           PARTITION BY idx ORDER BY r.score DESC, e.popularity DESC, e.entity_id
                       ) AS rn
                FROM ranked r JOIN entities e USING (entity_id)
            )
            SELECT c.*, ARRAY(
                SELECT a.alias FROM aliases a WHERE a.entity_id = c.entity_id ORDER BY a.alias
            ) AS aliases
            FROM candidates c WHERE c.rn <= 50
            ORDER BY c.idx, c.score DESC, c.popularity DESC, c.entity_id
        """

        rows_by_idx: Dict[int, List[Dict]] = defaultdict(list)
        with self.conn, self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT set_config('statement_timeout', %s, true)",
                (str(self.statement_timeout_ms),),
            )
            cursor.execute(statement, {"queries": queries})
            for row in cursor.fetchall():
                rows_by_idx[row["idx"]].append(dict(row))

        return [self._process_rows(rows_by_idx.get(idx, [])) for idx in range(1, len(queries) + 1)]

    def _batch_retrieve(self, queries: List[str], fuzzy: bool) -> List[List[DatabaseRecord]]:
        """Resolve many already-normalized queries in one round trip.

        Mirrors _retrieve's three weighted branches (label/alias/description),
        but each branch is a LATERAL subquery correlated to the per-mention
        input row, so the planner can still use a parameterized index-nested
        loop per mention (same index eligibility as the single-query path)
        instead of one unparameterized scan.
        """
        if fuzzy:
            predicate = (
                "{field} %% i.query AND length({field}) <= 255 "
                "AND length(i.query) <= 255 "
                "AND abs(length({field}) - length(i.query)) <= %(distance)s "
                "AND levenshtein_less_equal(left({field}, 255), left(i.query, 255), "
                "%(distance)s) <= %(distance)s"
            )
            score = "similarity({field}, i.query)"
            patterns: List[Any] = [None] * len(queries)
        else:
            predicate = "{field} ~ i.pattern"
            score = "1.0"
            patterns = [self._exact_pattern(q) for q in queries]

        branches = []
        search_fields = [
            ("entities", "label_folded", 2.0),
            ("aliases", "alias_folded", 1.5),
        ]
        # Description phrases remain a useful exact fallback, but fuzzy
        # typo matching against millions of long descriptions produces huge,
        # noisy candidate sets and dominates cold-cache I/O.
        if not fuzzy:
            search_fields.append(("entities", "description_folded", 1.0))
        for table, field, weight in search_fields:
            # Each branch must return a BOUNDED top-k: the ranking CTE below
            # keeps only the best 50 per mention anyway, and an unbounded
            # scan that collects every trigram/Levenshtein match for a common
            # word makes the whole statement blow its statement_timeout on a
            # cold KB. Fuzzy keeps the most similar rows; phrase keeps the
            # most popular ones (its branch score is a constant weight), so
            # the global popularity-aware ranking still sees them.
            qualified = f"t.{field}"
            if fuzzy:
                branches.append(
                    f"(SELECT t.entity_id, {weight} * similarity({qualified}, i.query) AS score "
                    f"FROM {table} t "
                    f"WHERE {predicate.format(field=qualified)} "
                    f"ORDER BY score DESC LIMIT 100)"
                )
            elif table == "entities":
                branches.append(
                    f"(SELECT t.entity_id, {weight} * {score.format(field=qualified)} AS score "
                    f"FROM entities t "
                    f"WHERE {predicate.format(field=qualified)} "
                    f"ORDER BY t.popularity DESC LIMIT 100)"
                )
            else:
                # aliases has no popularity column: rank by the linked entity's.
                branches.append(
                    f"(SELECT s.entity_id, s.score FROM ("
                    f"SELECT t.entity_id, {weight} * {score.format(field=qualified)} AS score, "
                    f"e.popularity AS pop "
                    f"FROM aliases t LEFT JOIN entities e ON e.entity_id = t.entity_id "
                    f"WHERE {predicate.format(field=qualified)} "
                    f"ORDER BY e.popularity DESC LIMIT 100"
                    f") s)"
                )
        matches_lateral = " UNION ALL ".join(branches)

        statement = """
            WITH input AS (
                SELECT * FROM unnest(%(queries)s::text[], %(patterns)s::text[])
                    WITH ORDINALITY AS t(query, pattern, idx)
            ),
            matches AS (
                SELECT i.idx, m.entity_id, m.score
                FROM input i, LATERAL ( __MATCHES__ ) m
            ),
            ranked AS (
                SELECT idx, entity_id, max(score) AS score FROM matches GROUP BY idx, entity_id
            ),
            candidates AS (
                SELECT idx, e.*, r.score,
                       row_number() OVER (
                           PARTITION BY idx ORDER BY r.score DESC, e.popularity DESC, e.entity_id
                       ) AS rn
                FROM ranked r JOIN entities e USING (entity_id)
            )
            SELECT c.*, ARRAY(
                SELECT a.alias FROM aliases a WHERE a.entity_id = c.entity_id ORDER BY a.alias
            ) AS aliases
            FROM candidates c WHERE c.rn <= 50
            ORDER BY c.idx, c.score DESC, c.popularity DESC, c.entity_id
        """.replace("__MATCHES__", matches_lateral)

        rows_by_idx: Dict[int, List[Dict]] = defaultdict(list)
        with self.conn:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (str(self.statement_timeout_ms),),
                )
                cursor.execute(
                    "SELECT set_config('pg_trgm.similarity_threshold', %s, true)",
                    (str(self.fuzzy_config.min_similarity),),
                )
                cursor.execute(
                    statement,
                    {
                        "queries": queries,
                        "patterns": patterns,
                        "distance": self.fuzzy_config.max_distance,
                    },
                )
                for row in cursor.fetchall():
                    rows_by_idx[row["idx"]].append(dict(row))

        return [self._process_rows(rows_by_idx.get(idx, [])) for idx in range(1, len(queries) + 1)]

    def _process_rows(self, rows: List[Dict]) -> List[DatabaseRecord]:
        records = []
        for row in rows:
            row_dict = dict(row)
            row_dict["source"] = "postgres"
            if isinstance(row_dict.get("embedding"), (bytes, memoryview)):
                import pickle

                row_dict["embedding"] = pickle.loads(bytes(row_dict["embedding"]))
            record = self.map_to_record(row_dict)
            records.append(record)
        return records

    def write_cache(self, key: str, records: List[DatabaseRecord], ttl: int):
        pass

    def load_bulk(
        self, entities: List[DatabaseRecord], overwrite: bool = False, batch_size: int = 1000
    ) -> int:
        """Bulk load to Postgres"""
        cursor = self.conn.cursor()

        try:
            # Prepare entity data
            entity_values = [
                (e.entity_id, e.label, e.description, e.entity_type, e.popularity) for e in entities
            ]

            # Insert entities
            if overwrite:
                entity_query = """
                    INSERT INTO entities (entity_id, label, description, entity_type, popularity)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (entity_id) DO UPDATE SET
                        label = EXCLUDED.label,
                        description = EXCLUDED.description,
                        entity_type = EXCLUDED.entity_type,
                        popularity = EXCLUDED.popularity,
                        embedding = CASE
                            WHEN entities.label IS DISTINCT FROM EXCLUDED.label
                              OR entities.description IS DISTINCT FROM EXCLUDED.description
                            THEN NULL ELSE entities.embedding END,
                        embedding_model_id = CASE
                            WHEN entities.label IS DISTINCT FROM EXCLUDED.label
                              OR entities.description IS DISTINCT FROM EXCLUDED.description
                            THEN NULL ELSE entities.embedding_model_id END
                """
            else:
                entity_query = """
                    INSERT INTO entities (entity_id, label, description, entity_type, popularity)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (entity_id) DO NOTHING
                """

            execute_batch(cursor, entity_query, entity_values, page_size=batch_size)

            # Prepare alias data
            alias_values = []
            for entity in entities:
                for alias in entity.aliases:
                    alias_values.append((entity.entity_id, alias))

            # Delete old aliases if overwrite
            if overwrite and entities:
                entity_ids = [e.entity_id for e in entities]
                cursor.execute("DELETE FROM aliases WHERE entity_id = ANY(%s)", (entity_ids,))

            # Insert aliases
            if alias_values:
                execute_batch(
                    cursor,
                    "INSERT INTO aliases (entity_id, alias) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    alias_values,
                    page_size=batch_size,
                )

            self.conn.commit()
            return len(entities)

        except Exception as e:
            self.conn.rollback()
            print(f"[ERROR Postgres] Load bulk failed: {e}")
            raise
        finally:
            cursor.close()

    def clear(self):
        """Clear all data"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("TRUNCATE entities, aliases CASCADE")
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            print(f"[ERROR Postgres] Clear error: {e}")
        finally:
            cursor.close()

    def count(self) -> int:
        """Count entities"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM entities")
            return cursor.fetchone()[0]
        except:
            return 0
        finally:
            cursor.close()

    def get_all_entities(self) -> List[DatabaseRecord]:
        """Get all entities from PostgreSQL"""
        entities = []

        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            sql = """
                SELECT
                    e.entity_id,
                    e.label,
                    e.description,
                    e.entity_type,
                    e.popularity,
                    e.embedding,
                    e.embedding_model_id,
                    COALESCE(array_agg(a.alias) FILTER (WHERE a.alias IS NOT NULL), ARRAY[]::text[]) as aliases
                FROM entities e
                LEFT JOIN aliases a ON e.entity_id = a.entity_id
                GROUP BY e.entity_id, e.label, e.description, e.entity_type, e.popularity, e.embedding, e.embedding_model_id
            """
            cursor.execute(sql)

            for row in cursor.fetchall():
                row_dict = dict(row)
                row_dict["source"] = "postgres"

                # Deserialize embedding from bytes if needed
                if row_dict.get("embedding"):
                    import pickle

                    if isinstance(row_dict["embedding"], (bytes, memoryview)):
                        row_dict["embedding"] = pickle.loads(bytes(row_dict["embedding"]))

                record = self.map_to_record(row_dict)
                entities.append(record)

            cursor.close()

        except Exception as e:
            print(f"[ERROR Postgres] get_all_entities error: {e}")

        return entities

    def update_embeddings(
        self, entity_ids: List[str], embeddings: List[List[float]], model_id: str
    ) -> int:
        """Update embeddings in PostgreSQL"""
        cursor = self.conn.cursor()

        try:
            import pickle

            # Prepare batch data
            batch_data = []
            for eid, emb in zip(entity_ids, embeddings):
                emb_bytes = pickle.dumps(emb)
                batch_data.append((emb_bytes, model_id, eid))

            # Batch update
            execute_batch(
                cursor,
                "UPDATE entities SET embedding = %s, embedding_model_id = %s WHERE entity_id = %s",
                batch_data,
                page_size=500,
            )

            self.conn.commit()
            return len(batch_data)

        except Exception as e:
            self.conn.rollback()
            print(f"[ERROR Postgres] update_embeddings error: {e}")
            return 0
        finally:
            cursor.close()

    def is_available(self) -> bool:
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except:
            return False


class DatabaseChainComponent(BaseComponent[L2Config]):
    """Multi-layer database chain component"""

    def _setup(self):
        self.layers: List[DatabaseLayer] = []

        for layer_config in self.config.layers:
            if isinstance(layer_config, dict):
                layer_config = LayerConfig(**layer_config)

            if layer_config.type == "dict":
                layer = DictLayer(layer_config)
            elif layer_config.type == "redis":
                layer = RedisLayer(layer_config)
            elif layer_config.type == "elasticsearch":
                layer = ElasticsearchLayer(layer_config)
            elif layer_config.type == "postgres":
                layer = PostgresLayer(layer_config)
            else:
                raise ValueError(f"Unknown layer type: {layer_config.type}")

            self.layers.append(layer)

        self.layers.sort(key=lambda x: x.priority, reverse=True)  # Higher priority checked first

    def get_available_methods(self) -> List[str]:
        return [
            "search",
            "filter_by_popularity",
            "deduplicate_candidates",
            "limit_candidates",
            "sort_by_popularity",
        ]

    def search(self, mention: str) -> List[DatabaseRecord]:
        """Search through layers with fallback"""
        found_in_layer = None
        results = []

        for layer in self.layers:
            if not layer.is_available():
                continue

            layer_results = []

            for mode in layer.config.search_mode:
                if mode == "exact":
                    layer_results.extend(layer.search(mention))
                    if layer_results:
                        break
                elif mode == "fuzzy":
                    if layer.supports_fuzzy():
                        layer_results.extend(layer.search_fuzzy(mention))
                        if layer_results:
                            break

            if layer_results:
                layer_results = self.deduplicate_candidates(layer_results)
                results = layer_results
                found_in_layer = layer
                break

        if results and found_in_layer:
            self._cache_write(mention, results, found_in_layer)

        return results

    def search_many(self, mentions: List[str]) -> List[List[DatabaseRecord]]:
        """Search mentions in layer-priority order using native bulk APIs."""
        return [retrieval.candidates for retrieval in self.search_many_detailed(mentions)]

    def search_many_detailed(self, mentions: List[str]) -> List[MentionRetrieval]:
        """Search mentions in layer-priority order, keeping failures explicit.

        Layers are consulted in priority order exactly like search_many, but
        a layer that raises (or reports a backend failure) no longer loses
        the mention: the failure is attached to the mention and the next
        layer still gets a chance. A cache-layer outage degrades to a
        Postgres lookup instead of an exception, and a Postgres timeout
        surfaces as backend_error instead of a silent miss.
        """
        if not mentions:
            return []

        results: List[MentionRetrieval] = [MentionRetrieval() for _ in mentions]
        pending = list(range(len(mentions)))

        for layer in self.layers:
            if not pending or not layer.is_available():
                continue

            try:
                layer_out = layer.search_many_detailed([mentions[index] for index in pending])
            except Exception as e:
                print(f"[ERROR L2] {type(layer).__name__} search_many failed: {e}")
                for index in pending:
                    results[index].failures.append(
                        RetrievalFailure(
                            layer=type(layer).__name__,
                            phase="availability",
                            kind=_classify_retrieval_error(e),
                            message=str(e),
                        )
                    )
                continue

            next_pending = []
            cache_entries = []
            for position, index in enumerate(pending):
                retrieval = layer_out[position] if position < len(layer_out) else MentionRetrieval()
                candidates = self.deduplicate_candidates(retrieval.candidates)
                if candidates:
                    # A hit from this layer resolves the mention, superseding
                    # any failure reported by an earlier (cache) layer.
                    results[index] = MentionRetrieval(candidates=candidates)
                    cache_entries.append((mentions[index], candidates))
                else:
                    results[index].failures.extend(retrieval.failures)
                    next_pending.append(index)
            self._cache_write_many(cache_entries, layer)
            pending = next_pending

        return results

    def _cache_write_many(
        self,
        entries: List[tuple[str, List[DatabaseRecord]]],
        source_layer: DatabaseLayer,
    ):
        """Write a batch of results to eligible higher-priority cache layers."""
        if not entries:
            return
        for layer in self.layers:
            if layer.priority <= source_layer.priority or not layer.write:
                continue
            if layer.cache_policy == "always":
                layer.write_cache_many(entries, layer.ttl)
            else:
                for query, results in entries:
                    self._cache_write_to_layer(query, results, layer)

    def batch_search(self, mentions: List[str]) -> List[List[DatabaseRecord]]:
        """Compatibility alias with per-mention fallback and cache writeback."""
        return self.search_many(mentions)

    def _cache_write(self, query: str, results: List[DatabaseRecord], source_layer: DatabaseLayer):
        """Write results to upper layers (higher priority = checked earlier)"""
        for layer in self.layers:
            # Skip source layer and all layers with lower priority
            if layer.priority <= source_layer.priority:
                continue
            if not layer.write:
                continue
            self._cache_write_to_layer(query, results, layer)

    @staticmethod
    def _cache_write_to_layer(query: str, results: List[DatabaseRecord], layer: DatabaseLayer):
        if layer.cache_policy == "always":
            layer.write_cache(query, results, layer.ttl)
        elif layer.cache_policy == "miss":
            existing = layer.search(query)
            if not existing:
                layer.write_cache(query, results, layer.ttl)
        elif layer.cache_policy == "hit":
            existing = layer.search(query)
            if existing:
                layer.write_cache(query, results, layer.ttl)

    def filter_by_popularity(
        self, records: List[DatabaseRecord], min_popularity: int = None
    ) -> List[DatabaseRecord]:
        threshold = min_popularity if min_popularity is not None else self.config.min_popularity
        return [r for r in records if r.popularity >= threshold]

    def deduplicate_candidates(self, records: List[DatabaseRecord]) -> List[DatabaseRecord]:
        seen = set()
        unique = []
        for record in records:
            if record.entity_id not in seen:
                unique.append(record)
                seen.add(record.entity_id)
        return unique

    def limit_candidates(
        self, records: List[DatabaseRecord], limit: int = None
    ) -> List[DatabaseRecord]:
        max_cands = limit if limit is not None else self.config.max_candidates
        return records[:max_cands]

    def sort_by_popularity(self, records: List[DatabaseRecord]) -> List[DatabaseRecord]:
        return sorted(records, key=lambda x: x.popularity, reverse=True)

    def load_entities(
        self,
        source: Union[str, Path, List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
        target_layers: List[str] = None,
        batch_size: int = 1000,
        overwrite: bool = False,
    ) -> Dict[str, int]:
        """
        Load entities from JSONL file, list of dicts, or dict.

        Accepts:
          - ``str`` / ``Path``: path to a JSONL file (one DatabaseRecord per line)
          - ``list[dict]``: each dict has at least ``entity_id`` and ``label``
          - ``dict[str, dict]``: keys are entity_ids, values are entity data

        Args:
            source: entity data (file path, list, or dict)
            target_layers: ['dict', 'redis', 'elasticsearch', 'postgres'] or None (all writable)
            batch_size: batch size for bulk operations
            overwrite: overwrite existing entities

        Returns:
            {'redis': 1500, 'elasticsearch': 1500}
        """
        if isinstance(source, (str, Path)):
            entities = self._parse_jsonl(source)
        elif isinstance(source, dict):
            entities = [
                DatabaseRecord(entity_id=eid, **data)
                if "entity_id" not in data
                else DatabaseRecord(**data)
                for eid, data in source.items()
            ]
        elif isinstance(source, list):
            entities = [DatabaseRecord(**e) if isinstance(e, dict) else e for e in source]
        else:
            raise TypeError(f"Expected file path, list, or dict; got {type(source)}")

        return self.load_records(
            entities,
            target_layers=target_layers,
            batch_size=batch_size,
            overwrite=overwrite,
        )

    def load_records(
        self,
        entities: List[DatabaseRecord],
        target_layers: List[str] = None,
        batch_size: int = 1000,
        overwrite: bool = False,
    ) -> Dict[str, int]:
        """
        Load pre-built DatabaseRecord objects into layers.

        Args:
            entities: list of DatabaseRecord instances
            target_layers: layer types to target (None = all writable)
            batch_size: batch size for bulk operations
            overwrite: overwrite existing entities

        Returns:
            Dict of layer_type -> count loaded
        """
        # Determine target layers
        if target_layers is None:
            target_layers = [l.config.type for l in self.layers if l.write]

        # Load to each layer
        results = {}
        for layer in self.layers:
            if layer.config.type not in target_layers:
                continue

            if not layer.is_available():
                continue

            count = layer.load_bulk(entities, overwrite=overwrite, batch_size=batch_size)
            results[layer.config.type] = count

        return results

    @staticmethod
    def _parse_jsonl(filepath: Union[str, Path]) -> List[DatabaseRecord]:
        """Parse JSONL file into DatabaseRecord list."""
        entities = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entities.append(DatabaseRecord(**data))
                except Exception as e:
                    print(f"[WARN] Line {line_num} parse error: {e}")
                    continue
        return entities

    def clear_layers(self, layer_names: List[str] = None):
        """Clear all entities in specified layers"""
        for layer in self.layers:
            if layer_names and layer.config.type not in layer_names:
                continue

            print(f"Clearing {layer.config.type}...")
            layer.clear()
            print(f"✓ Cleared")

    def get_all_entities(self) -> List[DatabaseRecord]:
        """Get all entities from all available layers (deduplicated)"""
        all_entities = []
        for layer in self.layers:
            if layer.is_available():
                all_entities.extend(layer.get_all_entities())
        return self.deduplicate_candidates(all_entities)

    def count_entities(self) -> Dict[str, int]:
        """Count entities in each layer"""
        counts = {}
        for layer in self.layers:
            counts[layer.config.type] = layer.count()
        return counts

    def precompute_embeddings(
        self,
        encoder_fn,
        template: str,
        model_id: str,
        target_layers: List[str] = None,
        batch_size: int = 32,
    ) -> Dict[str, int]:
        """
        Precompute embeddings for all entities in specified layers.

        Args:
            encoder_fn: Callable that takes List[str] and returns embeddings tensor
            template: Template string for formatting labels (e.g., "{label}: {description}")
            model_id: Model identifier to store with embeddings
            target_layers: Layer types to update (None = all)
            batch_size: Batch size for encoding

        Returns:
            Dict with count of updated entities per layer
        """
        from tqdm import tqdm

        results = {}

        for layer in self.layers:
            if target_layers and layer.config.type not in target_layers:
                continue

            if not layer.is_available():
                print(f"[WARN] {layer.config.type} unavailable, skipping")
                continue

            print(f"\nPrecomputing embeddings for {layer.config.type}...")

            # Get all entities
            entities = layer.get_all_entities()
            if not entities:
                print(f"  No entities found in {layer.config.type}")
                continue

            print(f"  Found {len(entities)} entities")

            # Format labels using template
            labels = []
            entity_ids = []
            for entity in entities:
                try:
                    formatted = template.format(**entity.dict())
                    labels.append(formatted)
                    entity_ids.append(entity.entity_id)
                except KeyError as e:
                    print(f"  [WARN] Template error for {entity.entity_id}: {e}")
                    continue

            # Encode in batches
            all_embeddings = []
            for i in tqdm(range(0, len(labels), batch_size), desc="Encoding"):
                batch_labels = labels[i : i + batch_size]
                batch_embeddings = encoder_fn(batch_labels)

                # Convert to list if tensor
                if hasattr(batch_embeddings, "tolist"):
                    batch_embeddings = batch_embeddings.tolist()
                elif hasattr(batch_embeddings, "cpu"):
                    batch_embeddings = batch_embeddings.cpu().numpy().tolist()

                all_embeddings.extend(batch_embeddings)

            # Update layer
            updated = layer.update_embeddings(entity_ids, all_embeddings, model_id)
            results[layer.config.type] = updated
            print(f"  Updated {updated} entities with embeddings")

        return results

    def get_layer(self, layer_type: str) -> DatabaseLayer:
        """Get layer by type"""
        for layer in self.layers:
            if layer.config.type == layer_type:
                return layer
        return None
