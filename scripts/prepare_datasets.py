#!/usr/bin/env python3
"""Download and normalize the four supported labeled datasets."""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as parquet

from dualkd.config import SUPPORTED_DATASETS, load_config


PARQUET_API = "https://huggingface.co/api/datasets/{dataset}/parquet/{config}/{split}/0.parquet"
LETTERS = "ABCDEFGHIJ"


def download_parquet(spec: dict, split: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = PARQUET_API.format(
        dataset=spec["hub_dataset"], config=spec["hub_config"], split=split
    )
    print(f"[download] {spec['hub_dataset']} {split}", flush=True)
    urllib.request.urlretrieve(url, destination)


def normalize(dataset_key: str, split: str, rows: list[dict], label: str) -> list[dict]:
    output = []
    for index, row in enumerate(rows):
        if dataset_key == "arc_challenge":
            labels = [str(value) for value in row["choices"]["label"]]
            texts = [str(value) for value in row["choices"]["text"]]
            answer = str(row.get("answerKey", ""))
            if answer not in labels:
                raise ValueError(f"unrecognized ARC answer at {split}:{index}")
            output.append(
                {
                    "id": f"arc-challenge-{row['id']}",
                    "dataset": label,
                    "source_split": split,
                    "prompt": f"Question: {row['question']}\nAnswer:",
                    "choices": [f" {text}" for text in texts],
                    "answer": labels.index(answer),
                }
            )
        elif dataset_key == "mmlu_aux":
            choices = [str(value) for value in row["choices"]]
            answer = row.get("answer")
            if not isinstance(answer, int) or not 0 <= answer < len(choices):
                raise ValueError(f"invalid MMLU label at {split}:{index}")
            options = "\n".join(
                f"{LETTERS[position]}. {value}" for position, value in enumerate(choices)
            )
            output.append(
                {
                    "id": f"mmlu-aux-{split}-{index:06d}",
                    "dataset": label,
                    "source_split": split,
                    "subject": str(row.get("subject") or "mixed"),
                    "prompt": f"Question: {row['question']}\nOptions:\n{options}\nAnswer:",
                    "choices": [f" {LETTERS[position]}" for position in range(len(choices))],
                    "answer": int(answer),
                }
            )
        elif dataset_key == "mmlu_pro":
            choices = [str(value) for value in row["options"]]
            while choices and choices[-1] == "N/A":
                choices.pop()
            answer = row.get("answer_index")
            if not isinstance(answer, int) or not 0 <= answer < len(choices):
                raise ValueError(f"invalid MMLU-Pro label at {split}:{index}")
            options = "\n".join(
                f"{LETTERS[position]}. {value}" for position, value in enumerate(choices)
            )
            output.append(
                {
                    "id": f"mmlu-pro-{row['question_id']}",
                    "dataset": label,
                    "source_split": split,
                    "category": str(row.get("category") or "other"),
                    "prompt": f"Question: {row['question']}\nOptions:\n{options}\nAnswer:",
                    "choices": [f" {LETTERS[position]}" for position in range(len(choices))],
                    "answer": int(answer),
                }
            )
        elif dataset_key == "hellaswag":
            answer = str(row.get("label", ""))
            if answer not in {"0", "1", "2", "3"}:
                raise ValueError(f"invalid HellaSwag label at {split}:{index}")
            output.append(
                {
                    "id": f"hellaswag-{split}-{index:06d}",
                    "dataset": label,
                    "source_split": split,
                    "source_index": int(row["ind"]),
                    "split_type": str(row.get("split_type") or "unknown"),
                    "activity_label": str(row.get("activity_label") or "unknown"),
                    "prompt": f"Question: {row['ctx']}\nContinuation:",
                    "choices": [f" {value}" for value in row["endings"]],
                    "answer": int(answer),
                }
            )
        else:  # protected by strict configuration validation
            raise ValueError(dataset_key)
    return output


def stratified_partition(rows: list[dict], first_count: int, *, key: str, seed: int):
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "unknown")].append(row)
    quotas = {name: first_count * len(values) / len(rows) for name, values in groups.items()}
    counts = {name: int(value) for name, value in quotas.items()}
    remainder = first_count - sum(counts.values())
    order = sorted(groups, key=lambda name: (quotas[name] - counts[name], name), reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    rng = random.Random(seed)
    selected = {
        row["id"]
        for name in sorted(groups)
        for row in rng.sample(groups[name], counts[name])
    }
    return [row for row in rows if row["id"] in selected], [row for row in rows if row["id"] not in selected]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=SUPPORTED_DATASETS)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    config = load_config(args.config)
    source_root = args.source_root or Path(config["source_root"])
    output_root = source_root / "datasets"
    raw_root = source_root / "raw"
    requested = tuple(dict.fromkeys(args.datasets or SUPPORTED_DATASETS))
    manifest = {
        "study": config["study"],
        "source": "Hugging Face official parquet exports",
        "seed": args.seed,
        "datasets": {},
    }
    all_ids: set[str] = set()
    for dataset_key in requested:
        spec = config["datasets"][dataset_key]
        normalized = {}
        for name, split in spec["source_splits"].items():
            raw_path = raw_root / f"{dataset_key}_{split}.parquet"
            if not raw_path.exists():
                download_parquet(spec, split, raw_path)
            normalized[name] = normalize(
                dataset_key,
                split,
                parquet.read_table(raw_path).to_pylist(),
                spec["label"],
            )
        if dataset_key == "mmlu_pro":
            gate, evaluation = stratified_partition(
                normalized["development_pool"],
                int(spec["expected"]["gate_training"]),
                key="category",
                seed=args.seed,
            )
            partitions = {
                "gate_training": (gate, "test (internal train partition)"),
                "lambda_selection": (normalized["lambda_selection"], "validation"),
                "evaluation": (evaluation, "test (internal held-out partition)"),
            }
        elif dataset_key == "hellaswag":
            selection, evaluation = stratified_partition(
                normalized["development_pool"],
                int(spec["expected"]["lambda_selection"]),
                key="split_type",
                seed=args.seed + 1,
            )
            partitions = {
                "gate_training": (normalized["gate_training"], "train"),
                "lambda_selection": (selection, "validation (selection partition)"),
                "evaluation": (evaluation, "validation (held-out partition)"),
            }
        else:
            partitions = {
                partition: (normalized[partition], source_split)
                for partition, source_split in spec["source_splits"].items()
            }
        manifest["datasets"][dataset_key] = {
            "label": spec["label"],
            "source": spec["hub_dataset"],
            "config": spec["hub_config"],
            "protocol": spec["protocol"],
            "splits": {},
        }
        dataset_ids: set[str] = set()
        for partition, (rows, source_split) in partitions.items():
            expected = int(spec["expected"][partition])
            if len(rows) != expected:
                raise ValueError(f"{dataset_key}/{partition}: expected {expected}, got {len(rows)}")
            ids = {row["id"] for row in rows}
            if len(ids) != len(rows) or dataset_ids.intersection(ids) or all_ids.intersection(ids):
                raise ValueError(f"duplicate or overlapping IDs in {dataset_key}/{partition}")
            dataset_ids.update(ids)
            all_ids.update(ids)
            path = output_root / partition / f"{dataset_key}.jsonl"
            write_jsonl(path, rows)
            manifest["datasets"][dataset_key]["splits"][partition] = {
                "source_split": source_split,
                "count": len(rows),
                "path": str(path),
            }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_root": str(output_root), "datasets": list(requested)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
