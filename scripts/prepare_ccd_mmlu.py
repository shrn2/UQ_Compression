"""Create deterministic non-test MMLU auxiliary prompts for CCD supervision."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


DATASET = "cais/mmlu"
CONFIG = "all"
SPLIT = "auxiliary_train"
API = "https://datasets-server.huggingface.co/rows"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def fetch(count: int, cache_path: Path) -> list[dict]:
    rows: list[dict] = []
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    while len(rows) < count:
        length = min(100, count - len(rows))
        query = urllib.parse.urlencode(
            {"dataset": DATASET, "config": CONFIG, "split": SPLIT, "offset": len(rows), "length": length}
        )
        request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": "tuac/0.4"})
        for attempt in range(8):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 7:
                    raise
                time.sleep(min(30, 2 ** (attempt + 1)))
        page = [item["row"] for item in payload["rows"]]
        if not page:
            break
        rows.extend(page)
        with cache_path.open("a", encoding="utf-8") as handle:
            for row in page:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        time.sleep(0.1)
    if len(rows) < count:
        raise ValueError(f"requested {count} rows but received {len(rows)}")
    return rows[:count]


def convert(index: int, row: dict) -> dict:
    choices = [str(value) for value in row["choices"]]
    options = "\n".join(f"{LETTERS[position]}. {value}" for position, value in enumerate(choices))
    return {
        "id": f"mmlu-aux-{index:05d}",
        "dataset": "MMLU-auxiliary-train",
        "subject": str(row.get("subject") or "mixed"),
        "prompt": f"Question: {row['question']}\nOptions:\n{options}\nAnswer:",
        "choices": [f" {LETTERS[position]}" for position in range(len(choices))],
        "answer": int(row["answer"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    destination = Path(args.output)
    rows = fetch(args.count, destination.parent / ".mmlu_auxiliary_rows_cache.jsonl")
    converted = [convert(index, row) for index, row in enumerate(rows)]
    if len({item["id"] for item in converted}) != len(converted):
        raise ValueError("generated duplicate IDs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for item in converted:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(converted)} {DATASET}/{SPLIT} examples to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
