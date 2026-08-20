"""Download deterministic multiple-choice transfer cohorts from the HF datasets viewer API."""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


API = "https://datasets-server.huggingface.co/rows"


DATASETS = {
    "openbookqa": {"dataset": "allenai/openbookqa", "config": "main", "split": "test"},
    "commonsenseqa": {"dataset": "tau/commonsense_qa", "config": "default", "split": "validation"},
    "hellaswag": {"dataset": "Rowan/hellaswag", "config": "default", "split": "validation"},
    "winogrande": {"dataset": "allenai/winogrande", "config": "winogrande_xl", "split": "validation"},
}


def fetch(spec: dict, count: int) -> list[dict]:
    rows: list[dict] = []
    while len(rows) < count:
        length = min(100, count - len(rows))
        query = urllib.parse.urlencode({**spec, "offset": len(rows), "length": length})
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
        page = [item["row"] for item in payload.get("rows", [])]
        if not page:
            break
        rows.extend(page)
        time.sleep(0.15)
    if len(rows) < count:
        raise ValueError(f"requested {count} rows but {spec['dataset']} returned {len(rows)}")
    return rows[:count]


def letters_choices(choices: dict) -> tuple[list[str], list[str]]:
    labels = [str(value) for value in choices["label"]]
    texts = [str(value) for value in choices["text"]]
    return labels, texts


def convert(name: str, row: dict) -> dict:
    if name == "openbookqa":
        labels, texts = letters_choices(row["choices"])
        answer = labels.index(str(row["answerKey"]))
        return {"id": f"openbookqa-{row['id']}", "dataset": "OpenBookQA", "prompt": f"Question: {row['question_stem']}\nAnswer:", "choices": [f" {text}" for text in texts], "answer": answer}
    if name == "commonsenseqa":
        labels, texts = letters_choices(row["choices"])
        answer = labels.index(str(row["answerKey"]))
        return {"id": f"commonsenseqa-{row['id']}", "dataset": "CommonsenseQA", "prompt": f"Question: {row['question']}\nAnswer:", "choices": [f" {text}" for text in texts], "answer": answer}
    if name == "hellaswag":
        return {"id": f"hellaswag-{row['ind']}", "dataset": "HellaSwag", "prompt": f"Question: {row['ctx']}\nContinuation:", "choices": [f" {text}" for text in row["endings"]], "answer": int(row["label"])}
    if name == "winogrande":
        return {"id": f"winogrande-{row['sentence']}", "dataset": "Winogrande", "prompt": f"Question: {row['sentence']}\nAnswer:", "choices": [f" {row['option1']}", f" {row['option2']}"], "answer": int(row["answer"]) - 1}
    raise ValueError(f"unknown dataset {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260850)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for offset, name in enumerate(DATASETS):
        rows = fetch(DATASETS[name], args.count)
        selected = random.Random(args.seed + offset).sample(rows, args.count)
        destination = output_dir / f"{name}.jsonl"
        with destination.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(convert(name, row), ensure_ascii=False) + "\n")
        print(f"wrote {len(selected)} {name} examples to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
