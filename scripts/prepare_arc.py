"""Create deterministic TUAC JSONL files from Hugging Face's ARC viewer API."""

from __future__ import annotations

import argparse
import json
import random
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "allenai/ai2_arc"
CONFIG = "ARC-Challenge"
API = "https://datasets-server.huggingface.co/rows"


def fetch(split: str, pool_size: int) -> list[dict]:
    rows: list[dict] = []
    while len(rows) < pool_size:
        length = min(100, pool_size - len(rows))
        query = urllib.parse.urlencode(
            {"dataset": DATASET, "config": CONFIG, "split": split, "offset": len(rows), "length": length}
        )
        request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": "tuac/0.2"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
        page = [item["row"] for item in payload["rows"]]
        if not page:
            break
        rows.extend(page)
    if len(rows) < pool_size:
        raise ValueError(f"requested {pool_size} rows but {split} only returned {len(rows)}")
    return rows


def convert(row: dict) -> dict:
    labels = [str(value) for value in row["choices"]["label"]]
    texts = [str(value) for value in row["choices"]["text"]]
    answer = str(row["answerKey"])
    if answer not in labels:
        # Some ARC records use numeric answers while choices use A/B/C/D.
        numeric = int(answer) - 1 if answer.isdigit() else -1
        if not 0 <= numeric < len(labels):
            raise ValueError(f"cannot map answer {answer!r} for {row['id']}")
        answer_index = numeric
    else:
        answer_index = labels.index(answer)
    return {
        "id": str(row["id"]),
        "dataset": "ARC-Challenge",
        "prompt": f"Question: {row['question']}\nAnswer:",
        "choices": [f" {text}" for text in texts],
        "answer": answer_index,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= args.count <= args.pool_size:
        parser.error("require 1 <= count <= pool-size")
    rows = fetch(args.split, args.pool_size)
    selected = random.Random(args.seed).sample(rows, args.count)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(convert(row), ensure_ascii=False) + "\n")
    print(f"wrote {len(selected)} {CONFIG}/{args.split} examples to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
