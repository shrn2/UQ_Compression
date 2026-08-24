"""Hugging Face causal-LM scorer for multiple-choice probabilities."""

from __future__ import annotations

import inspect
from collections.abc import Sequence

import numpy as np

from .data import MultipleChoiceExample, pad_probabilities


class HFCausalLMScorer:
    def __init__(
        self,
        model_name: str,
        *,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        length_normalize: bool = True,
        batch_size: int = 32,
        max_batch_tokens: int = 8192,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional integration
            raise RuntimeError(
                "Hugging Face execution requires: pip install -e '.[experiment]'"
            ) from exc
        if batch_size < 1 or max_batch_tokens < 1:
            raise ValueError("batch_size and max_batch_tokens must be positive")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        ).eval()
        self.length_normalize = bool(length_normalize)
        self.batch_size = int(batch_size)
        self.max_batch_tokens = int(max_batch_tokens)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        parameters = inspect.signature(self.model.forward).parameters
        self._logits_keep_parameter = (
            "logits_to_keep"
            if "logits_to_keep" in parameters
            else "num_logits_to_keep"
            if "num_logits_to_keep" in parameters
            else None
        )

    def _input_device(self):
        return self.model.get_input_embeddings().weight.device

    def score(self, examples: Sequence[MultipleChoiceExample]) -> np.ndarray:
        """Return full-choice, length-normalized continuation probabilities."""

        if not examples:
            raise ValueError("at least one example is required")
        tokenized_choices = [
            [
                self.tokenizer.encode(choice, add_special_tokens=False)
                for choice in example.choices
            ]
            for example in examples
        ]
        if any(not choice for choices in tokenized_choices for choice in choices):
            raise ValueError("a choice tokenized to an empty sequence")
        if all(len(choice) == 1 for choices in tokenized_choices for choice in choices):
            return self._score_single_token_choices(examples, tokenized_choices)
        return self._score_multi_token_choices(examples, tokenized_choices)

    def _batches(self, lengths: list[int]) -> list[list[int]]:
        order = sorted(range(len(lengths)), key=lengths.__getitem__)
        batches: list[list[int]] = []
        current: list[int] = []
        for index in order:
            projected = lengths[index] * (len(current) + 1)
            if current and (
                len(current) >= self.batch_size or projected > self.max_batch_tokens
            ):
                batches.append(current)
                current = []
            current.append(index)
        if current:
            batches.append(current)
        return batches

    def _score_multi_token_choices(self, examples, tokenized_choices) -> np.ndarray:
        torch = self.torch
        max_length = int(getattr(self.model.config, "max_position_embeddings", 2048))
        sequences: list[list[int]] = []
        choice_lengths: list[int] = []
        for example, choices in zip(examples, tokenized_choices):
            prompt_ids = self.tokenizer.encode(example.prompt, add_special_tokens=True)
            for choice_ids in choices:
                allowed_prompt = max(1, max_length - len(choice_ids))
                sequences.append(prompt_ids[-allowed_prompt:] + choice_ids)
                choice_lengths.append(len(choice_ids))
        scores = np.empty(len(sequences), dtype=np.float64)
        for batch_indices in self._batches([len(row) for row in sequences]):
            encoded = self.tokenizer.pad(
                {"input_ids": [sequences[index] for index in batch_indices]},
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._input_device()) for key, value in encoded.items()}
            model_inputs = dict(encoded)
            if self._logits_keep_parameter is not None:
                model_inputs[self._logits_keep_parameter] = (
                    max(choice_lengths[index] for index in batch_indices) + 1
                )
            with torch.inference_mode():
                log_probs = torch.log_softmax(
                    self.model(**model_inputs).logits.float(), dim=-1
                )
            for local, global_index in enumerate(batch_indices):
                length = choice_lengths[global_index]
                positions = log_probs[local, -length - 1 : -1]
                targets = encoded["input_ids"][local, -length:]
                token_scores = positions.gather(1, targets[:, None]).squeeze(1)
                value = token_scores.mean() if self.length_normalize else token_scores.sum()
                scores[global_index] = float(value.item())
        rows = []
        cursor = 0
        for example in examples:
            values = scores[cursor : cursor + len(example.choices)]
            cursor += len(example.choices)
            values = values - values.max()
            probabilities = np.exp(values)
            rows.append(probabilities / probabilities.sum())
        return pad_probabilities(rows)

    def _score_single_token_choices(self, examples, tokenized_choices) -> np.ndarray:
        torch = self.torch
        max_length = int(getattr(self.model.config, "max_position_embeddings", 2048))
        sequences = [
            self.tokenizer.encode(example.prompt, add_special_tokens=True)[-max_length:]
            for example in examples
        ]
        rows: list[np.ndarray | None] = [None] * len(examples)
        for batch_indices in self._batches([len(row) for row in sequences]):
            encoded = self.tokenizer.pad(
                {"input_ids": [sequences[index] for index in batch_indices]},
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._input_device()) for key, value in encoded.items()}
            model_inputs = dict(encoded)
            if self._logits_keep_parameter is not None:
                model_inputs[self._logits_keep_parameter] = 1
            with torch.inference_mode():
                log_probs = torch.log_softmax(
                    self.model(**model_inputs).logits[:, -1].float(), dim=-1
                )
            for local, global_index in enumerate(batch_indices):
                target_ids = torch.tensor(
                    [choice[0] for choice in tokenized_choices[global_index]],
                    device=log_probs.device,
                )
                values = log_probs[local].gather(0, target_ids).detach().cpu().numpy()
                values = values - values.max()
                probabilities = np.exp(values)
                rows[global_index] = probabilities / probabilities.sum()
        return pad_probabilities(row for row in rows if row is not None)
