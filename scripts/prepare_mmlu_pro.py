"""Create deterministic TUAC JSONL files from the official MMLU-Pro viewer API."""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

DATASET = "TIGER-Lab/MMLU-Pro"
API = "https://datasets-server.huggingface.co/rows"
LETTERS = "ABCDEFGHIJ"


def fetch(split: str, pool_size: int, cache_path: Path) -> list[dict]:
    rows: list[dict] = []
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        rows = rows[:pool_size]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    while len(rows) < pool_size:
        length = min(100, pool_size - len(rows))
        query = urllib.parse.urlencode(
            {"dataset": DATASET, "config": "default", "split": split, "offset": len(rows), "length": length}
        )
        request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": "tuac/0.3"})
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
        time.sleep(0.25)
    if len(rows) < pool_size:
        raise ValueError(f"requested {pool_size} rows but {split} returned {len(rows)}")
    return rows


def stratified_sample(rows: list[dict], count: int, seed: int) -> list[dict]:
    if count == len(rows):
        return rows
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["category"])].append(row)
    rng = random.Random(seed)
    selected: list[dict] = []
    quotas = {name: count * len(values) / len(rows) for name, values in groups.items()}
    base = {name: min(len(groups[name]), int(value)) for name, value in quotas.items()}
    remaining = count - sum(base.values())
    order = sorted(groups, key=lambda name: quotas[name] - base[name], reverse=True)
    for name in order[:remaining]:
        base[name] += 1
    for name, values in groups.items():
        selected.extend(rng.sample(values, base[name]))
    rng.shuffle(selected)
    return selected


def convert(row: dict) -> dict:
    options = [str(value) for value in row["options"]]
    option_text = "\n".join(f"{LETTERS[index]}. {value}" for index, value in enumerate(options))
    return {
        "id": f"mmlu-pro-{row['question_id']}",
        "dataset": "MMLU-Pro",
        "category": str(row["category"]),
        "prompt": f"Question: {row['question']}\nOptions:\n{option_text}\nAnswer:",
        "choices": [f" {LETTERS[index]}" for index in range(len(options))],
        "answer": int(row["answer_index"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--pool-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= args.count <= args.pool_size:
        parser.error("require 1 <= count <= pool-size")
    destination = Path(args.output)
    cache_path = destination.parent / f".{args.split}_rows_cache.jsonl"
    selected = stratified_sample(fetch(args.split, args.pool_size, cache_path), args.count, args.seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(convert(row), ensure_ascii=False) + "\n")
    print(f"wrote {len(selected)} MMLU-Pro/{args.split} examples to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
