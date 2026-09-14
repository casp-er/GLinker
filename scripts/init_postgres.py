"""Explicitly provision a dedicated GLiNKER schema; never reads .env.

Usage: python scripts/init_postgres.py --dsn 'host=... dbname=...' --schema glinker
Requires extension privileges on first use. Existing KB tables are not migrated.
"""

import argparse
import re

import psycopg2
from psycopg2 import sql


def initialize(conn, schema):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", schema) or schema in {
        "public",
        "pg_catalog",
        "information_schema",
    }:
        raise ValueError("Select a dedicated lowercase GLiNKER schema")
    with conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public")
            cur.execute("CREATE EXTENSION IF NOT EXISTS unaccent WITH SCHEMA public")
            cur.execute("CREATE EXTENSION IF NOT EXISTS fuzzystrmatch WITH SCHEMA public")
            cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
            cur.execute(
                sql.SQL("SET LOCAL search_path TO {}, public").format(sql.Identifier(schema))
            )
            cur.execute("""
                CREATE OR REPLACE FUNCTION glinker_fold(value text) RETURNS text
                LANGUAGE sql STABLE STRICT PARALLEL SAFE AS $$
                    SELECT btrim(lower(public.unaccent('public.unaccent',
                        normalize(value, NFKC))))
                $$;
                CREATE TABLE IF NOT EXISTS entities (
                    entity_id text PRIMARY KEY,
                    label text NOT NULL,
                    description text NOT NULL DEFAULT '',
                    entity_type text NOT NULL DEFAULT '',
                    popularity bigint NOT NULL DEFAULT 0,
                    label_folded text NOT NULL,
                    description_folded text NOT NULL,
                    embedding bytea,
                    embedding_model_id text
                );
                CREATE TABLE IF NOT EXISTS aliases (
                    entity_id text REFERENCES entities(entity_id) ON DELETE CASCADE,
                    alias text NOT NULL,
                    alias_folded text NOT NULL,
                    PRIMARY KEY (entity_id, alias)
                );
                CREATE OR REPLACE FUNCTION glinker_fold_entity() RETURNS trigger
                LANGUAGE plpgsql AS $$ BEGIN
                    NEW.label_folded := glinker_fold(NEW.label);
                    NEW.description_folded := glinker_fold(NEW.description);
                    RETURN NEW;
                END $$;
                CREATE OR REPLACE FUNCTION glinker_fold_alias() RETURNS trigger
                LANGUAGE plpgsql AS $$ BEGIN
                    NEW.alias_folded := glinker_fold(NEW.alias);
                    RETURN NEW;
                END $$;
                CREATE OR REPLACE TRIGGER entities_fold
                    BEFORE INSERT OR UPDATE OF label, description ON entities
                    FOR EACH ROW EXECUTE FUNCTION glinker_fold_entity();
                CREATE OR REPLACE TRIGGER aliases_fold
                    BEFORE INSERT OR UPDATE OF alias ON aliases
                    FOR EACH ROW EXECUTE FUNCTION glinker_fold_alias();
                CREATE INDEX IF NOT EXISTS entities_label_trgm ON entities USING gin (label_folded gin_trgm_ops);
                CREATE INDEX IF NOT EXISTS entities_description_trgm ON entities USING gin (description_folded gin_trgm_ops);
                CREATE INDEX IF NOT EXISTS aliases_alias_trgm ON aliases USING gin (alias_folded gin_trgm_ops);
            """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn", required=True, help="Explicit target; prefer a libpq service for credentials"
    )
    parser.add_argument("--schema", required=True)
    args = parser.parse_args()
    with psycopg2.connect(args.dsn) as conn:
        initialize(conn, args.schema)
