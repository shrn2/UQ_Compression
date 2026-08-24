"""JSONL contract for multiple-choice gate training and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MultipleChoiceExample:
    id: str
    prompt: str
    choices: tuple[str, ...]
    answer: int | None = None
    dataset: str | None = None


def load_jsonl(path: str | Path, *, limit: int | None = None) -> list[MultipleChoiceExample]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    examples: list[MultipleChoiceExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                choices = tuple(str(value) for value in item["choices"])
                answer = item.get("answer")
                if len(choices) < 2 or not str(item["prompt"]).strip():
                    raise ValueError("prompt must be non-empty and at least two choices are required")
                if answer is not None and (
                    not isinstance(answer, int) or not 0 <= answer < len(choices)
                ):
                    raise ValueError("answer must be a valid zero-based choice index")
                examples.append(
                    MultipleChoiceExample(
                        id=str(item.get("id", line_number)),
                        prompt=str(item["prompt"]),
                        choices=choices,
                        answer=answer,
                        dataset=str(item["dataset"]) if item.get("dataset") is not None else None,
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid example at {path}:{line_number}: {exc}") from exc
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"no examples found in {path}")
    return examples


def pad_probabilities(rows: Iterable[Iterable[float]], width: int | None = None):
    import numpy as np

    materialized = [np.asarray(row, dtype=np.float64) for row in rows]
    if not materialized:
        raise ValueError("at least one probability row is required")
    target = width or max(row.size for row in materialized)
    if any(row.ndim != 1 or row.size > target for row in materialized):
        raise ValueError("probability rows must be vectors no wider than width")
    result = np.zeros((len(materialized), target), dtype=np.float64)
    for index, row in enumerate(materialized):
        result[index, : row.size] = row
    return result
