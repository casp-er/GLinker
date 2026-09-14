"""Stream a GLiNKER JSONL KB into an explicitly selected, provisioned schema.

Never reads .env. This writes the target database; obtain operator approval
before running against a real application database. Run init_postgres.py first.
"""

import argparse
import json

from glinker.l2.component import PostgresLayer
from glinker.l2.models import DatabaseRecord, LayerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    cfg = {"dsn": args.dsn, "schema": args.schema}
    layer = PostgresLayer(LayerConfig(type="postgres", priority=0, config=cfg))
    count = 0
    try:
        batch = []
        with open(args.input) as source:
            for line in source:
                if not line.strip():
                    continue
                batch.append(DatabaseRecord(**json.loads(line)))
                if len(batch) >= args.batch_size:
                    count += layer.load_bulk(
                        batch, overwrite=args.overwrite, batch_size=args.batch_size
                    )
                    batch = []
            if batch:
                count += layer.load_bulk(
                    batch, overwrite=args.overwrite, batch_size=args.batch_size
                )
        with layer.conn:
            with layer.conn.cursor() as cursor:
                cursor.execute("ANALYZE entities")
                cursor.execute("ANALYZE aliases")
        print(f"Processed {count} entities")
    finally:
        layer.conn.close()


if __name__ == "__main__":
    main()
