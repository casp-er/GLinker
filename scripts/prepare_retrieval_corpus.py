"""Select first N dump rows plus all expected IDs, without duplicating entities."""

import argparse
import hashlib
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--mentions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--distractors", type=int, default=50000)
    args = parser.parse_args()
    cases = json.load(open(args.mentions))
    wanted = {case["expected_id"] for case in cases}
    found = set()
    seen = set()
    digest = hashlib.sha256()
    with open(args.input) as source, open(args.output, "w") as target:
        for i, line in enumerate(source):
            record = json.loads(line)
            eid = record["entity_id"]
            if eid in wanted:
                found.add(eid)
            if (i < args.distractors or eid in wanted) and eid not in seen:
                seen.add(eid)
                target.write(line)
                digest.update(line.encode())
    if missing := wanted - found:
        raise ValueError(f"Expected IDs absent from source: {sorted(missing)}")
    print(json.dumps({"entities": len(seen), "sha256": digest.hexdigest()}))


if __name__ == "__main__":
    main()
