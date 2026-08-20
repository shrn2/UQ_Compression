"""Portable calibration artifacts; model inference and analysis stay separable."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CalibrationArtifact:
    teacher_probabilities: np.ndarray
    student_probabilities: np.ndarray
    quantized_probabilities: np.ndarray
    layer_names: tuple[str, ...]
    parameter_counts: np.ndarray
    labels: np.ndarray
    example_ids: tuple[str, ...]
    probe_bits: int
    metadata: dict

    def validate(self) -> None:
        examples, classes = self.student_probabilities.shape
        layers = len(self.layer_names)
        if self.teacher_probabilities.shape != (examples, classes):
            raise ValueError("teacher and student probability shapes differ")
        if self.quantized_probabilities.shape != (layers, examples, classes):
            raise ValueError("quantized probabilities do not match layers/examples/classes")
        if self.parameter_counts.shape != (layers,) or np.any(self.parameter_counts <= 0):
            raise ValueError("parameter counts must contain one positive value per layer")
        if self.labels.shape != (examples,) or len(self.example_ids) != examples:
            raise ValueError("labels/example IDs do not match example count")

    def save(self, path: str | Path) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            teacher_probabilities=np.asarray(self.teacher_probabilities, dtype=np.float32),
            student_probabilities=np.asarray(self.student_probabilities, dtype=np.float32),
            quantized_probabilities=np.asarray(self.quantized_probabilities, dtype=np.float32),
            layer_names=np.asarray(self.layer_names, dtype=np.str_),
            parameter_counts=np.asarray(self.parameter_counts, dtype=np.int64),
            labels=np.asarray(self.labels, dtype=np.int64),
            example_ids=np.asarray(self.example_ids, dtype=np.str_),
            probe_bits=np.asarray(self.probe_bits, dtype=np.int64),
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True), dtype=np.str_),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationArtifact":
        with np.load(path, allow_pickle=False) as values:
            artifact = cls(
                teacher_probabilities=values["teacher_probabilities"],
                student_probabilities=values["student_probabilities"],
                quantized_probabilities=values["quantized_probabilities"],
                layer_names=tuple(str(value) for value in values["layer_names"]),
                parameter_counts=values["parameter_counts"],
                labels=values["labels"],
                example_ids=tuple(str(value) for value in values["example_ids"]),
                probe_bits=int(values["probe_bits"]),
                metadata=json.loads(str(values["metadata"])),
            )
        artifact.validate()
        return artifact


@dataclass(frozen=True)
class MultiBitCalibrationArtifact:
    """Calibration cache with empirical probes at every candidate precision."""

    teacher_probabilities: np.ndarray
    student_probabilities: np.ndarray
    quantized_probabilities: np.ndarray
    probe_bits: tuple[int, ...]
    layer_names: tuple[str, ...]
    parameter_counts: np.ndarray
    labels: np.ndarray
    example_ids: tuple[str, ...]
    metadata: dict

    def validate(self) -> None:
        examples, classes = self.student_probabilities.shape
        bits = len(self.probe_bits)
        layers = len(self.layer_names)
        if bits == 0 or len(set(self.probe_bits)) != bits or any(bit < 2 for bit in self.probe_bits):
            raise ValueError("probe_bits must contain unique widths of at least 2 bits")
        if self.teacher_probabilities.shape != (examples, classes):
            raise ValueError("teacher and student probability shapes differ")
        if self.quantized_probabilities.shape != (bits, layers, examples, classes):
            raise ValueError("quantized probabilities do not match bits/layers/examples/classes")
        if self.parameter_counts.shape != (layers,) or np.any(self.parameter_counts <= 0):
            raise ValueError("parameter counts must contain one positive value per layer")
        if self.labels.shape != (examples,) or len(self.example_ids) != examples:
            raise ValueError("labels/example IDs do not match example count")

    def save(self, path: str | Path) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            teacher_probabilities=np.asarray(self.teacher_probabilities, dtype=np.float32),
            student_probabilities=np.asarray(self.student_probabilities, dtype=np.float32),
            quantized_probabilities=np.asarray(self.quantized_probabilities, dtype=np.float32),
            probe_bits=np.asarray(self.probe_bits, dtype=np.int64),
            layer_names=np.asarray(self.layer_names, dtype=np.str_),
            parameter_counts=np.asarray(self.parameter_counts, dtype=np.int64),
            labels=np.asarray(self.labels, dtype=np.int64),
            example_ids=np.asarray(self.example_ids, dtype=np.str_),
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True), dtype=np.str_),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MultiBitCalibrationArtifact":
        with np.load(path, allow_pickle=False) as values:
            artifact = cls(
                teacher_probabilities=values["teacher_probabilities"],
                student_probabilities=values["student_probabilities"],
                quantized_probabilities=values["quantized_probabilities"],
                probe_bits=tuple(int(value) for value in values["probe_bits"]),
                layer_names=tuple(str(value) for value in values["layer_names"]),
                parameter_counts=values["parameter_counts"],
                labels=values["labels"],
                example_ids=tuple(str(value) for value in values["example_ids"]),
                metadata=json.loads(str(values["metadata"])),
            )
        artifact.validate()
        return artifact
