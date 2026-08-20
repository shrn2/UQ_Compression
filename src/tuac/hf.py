"""Optional Hugging Face causal-LM adapter for multiple-choice probabilities."""

from __future__ import annotations

import inspect
from typing import Sequence

import numpy as np

from .data import MultipleChoiceExample, pad_probabilities


class HFCausalLMScorer:
    def __init__(
        self,
        model_name: str,
        *,
        device_map: str = "auto",
        dtype: str = "auto",
        quantization: str = "none",
        bnb_4bit_compute_dtype: str = "float16",
        trust_remote_code: bool = False,
        length_normalize: bool = True,
        batch_size: int = 32,
        max_batch_tokens: int = 4096,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional integration
            raise RuntimeError("Hugging Face execution requires: pip install 'tuac[hf]'") from exc
        self.torch = torch
        quantization = str(quantization).lower()
        if quantization not in {"none", "4bit", "8bit"}:
            raise ValueError("quantization must be one of: none, 4bit, 8bit")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        load_kwargs = {
            "device_map": device_map,
            "torch_dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        if quantization != "none":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:  # pragma: no cover - optional integration
                raise RuntimeError("quantized execution requires transformers BitsAndBytesConfig") from exc
            compute_dtype = getattr(torch, str(bnb_4bit_compute_dtype), None)
            if compute_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
                raise ValueError("bnb_4bit_compute_dtype must be float16, bfloat16, or float32")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantization == "4bit",
                load_in_8bit=quantization == "8bit",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            load_kwargs["torch_dtype"] = compute_dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs).eval()
        self.quantization = quantization
        self.bnb_4bit_compute_dtype = str(bnb_4bit_compute_dtype)
        self.length_normalize = length_normalize
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        if max_batch_tokens < 1:
            raise ValueError("max_batch_tokens must be positive")
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
        embedding = self.model.get_input_embeddings()
        return embedding.weight.device

    def choice_log_likelihood(self, prompt: str, choice: str) -> float:
        torch = self.torch
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        choice_ids = self.tokenizer.encode(choice, add_special_tokens=False)
        if not choice_ids:
            raise ValueError("a choice tokenized to an empty sequence")
        max_length = int(getattr(self.model.config, "max_position_embeddings", 2048))
        allowed_prompt = max(1, max_length - len(choice_ids))
        prompt_ids = prompt_ids[-allowed_prompt:]
        input_ids = torch.tensor([prompt_ids + choice_ids], device=self._input_device())
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids).logits[0]
        start = len(prompt_ids) - 1
        positions = logits[start : start + len(choice_ids)]
        targets = torch.tensor(choice_ids, device=positions.device)
        token_scores = torch.log_softmax(positions.float(), dim=-1).gather(1, targets[:, None]).squeeze(1)
        score = token_scores.mean() if self.length_normalize else token_scores.sum()
        return float(score.item())

    def score(self, examples: Sequence[MultipleChoiceExample]) -> np.ndarray:
        """Score all choices with right-padded batched continuation likelihoods."""

        torch = self.torch
        tokenized_choices = [
            [self.tokenizer.encode(choice, add_special_tokens=False) for choice in example.choices]
            for example in examples
        ]
        if all(len(choice) == 1 for choices in tokenized_choices for choice in choices):
            return self._score_single_token_choices(examples, tokenized_choices)
        sequences: list[list[int]] = []
        choice_lengths: list[int] = []
        max_length = int(getattr(self.model.config, "max_position_embeddings", 2048))
        for example in examples:
            prompt_ids = self.tokenizer.encode(example.prompt, add_special_tokens=True)
            for choice in example.choices:
                choice_ids = self.tokenizer.encode(choice, add_special_tokens=False)
                if not choice_ids:
                    raise ValueError("a choice tokenized to an empty sequence")
                allowed_prompt = max(1, max_length - len(choice_ids))
                current_prompt = prompt_ids[-allowed_prompt:]
                sequences.append(current_prompt + choice_ids)
                choice_lengths.append(len(choice_ids))

        choice_scores = np.empty(len(sequences), dtype=np.float64)
        # Batch size 1 matches the unbatched reference exactly. Larger fixed batches
        # are available for throughput-sensitive fallback scoring, with their batch
        # configuration recorded in the experiment artifact.
        if self.batch_size == 1:
            batches = [[index] for index in range(len(sequences))]
        else:
            length_order = sorted(range(len(sequences)), key=lambda index: len(sequences[index]))
            batches = []
            current: list[int] = []
            for index in length_order:
                projected = len(sequences[index]) * (len(current) + 1)
                if current and (
                    len(current) >= self.batch_size or projected > self.max_batch_tokens
                ):
                    batches.append(current)
                    current = []
                current.append(index)
            if current:
                batches.append(current)
        for batch_indices in batches:
            encoded = self.tokenizer.pad(
                {"input_ids": [sequences[index] for index in batch_indices]},
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._input_device()) for key, value in encoded.items()}
            model_inputs = dict(encoded)
            if self._logits_keep_parameter is not None:
                model_inputs[self._logits_keep_parameter] = max(choice_lengths[index] for index in batch_indices) + 1
            with torch.inference_mode():
                logits = self.model(**model_inputs).logits
                log_probs = torch.log_softmax(logits.float(), dim=-1)
            for local, global_index in enumerate(batch_indices):
                choice_length = choice_lengths[global_index]
                positions = log_probs[local, -choice_length - 1 : -1]
                targets = encoded["input_ids"][local, -choice_length:]
                token_scores = positions.gather(1, targets[:, None]).squeeze(1)
                value = token_scores.mean() if self.length_normalize else token_scores.sum()
                choice_scores[global_index] = float(value.item())
            del logits, log_probs, encoded

        rows = []
        cursor = 0
        for example in examples:
            scores = choice_scores[cursor : cursor + len(example.choices)]
            cursor += len(example.choices)
            scores = scores - scores.max()
            probabilities = np.exp(scores)
            rows.append(probabilities / probabilities.sum())
        return pad_probabilities(rows)

    def _score_single_token_choices(
        self,
        examples: Sequence[MultipleChoiceExample],
        tokenized_choices: Sequence[Sequence[Sequence[int]]],
    ) -> np.ndarray:
        """Score shared-prompt, one-token choices with one forward row per example."""

        torch = self.torch
        max_length = int(getattr(self.model.config, "max_position_embeddings", 2048))
        sequences = [
            self.tokenizer.encode(example.prompt, add_special_tokens=True)[-max_length:]
            for example in examples
        ]
        rows: list[np.ndarray | None] = [None] * len(examples)
        length_order = sorted(range(len(sequences)), key=lambda index: len(sequences[index]))
        batches: list[list[int]] = []
        current: list[int] = []
        for index in length_order:
            projected = len(sequences[index]) * (len(current) + 1)
            if current and (len(current) >= self.batch_size or projected > self.max_batch_tokens):
                batches.append(current)
                current = []
            current.append(index)
        if current:
            batches.append(current)
        for batch_indices in batches:
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
                logits = self.model(**model_inputs).logits[:, -1]
                log_probs = torch.log_softmax(logits.float(), dim=-1)
            for local, global_index in enumerate(batch_indices):
                target_ids = torch.tensor(
                    [choice[0] for choice in tokenized_choices[global_index]],
                    device=log_probs.device,
                )
                values = log_probs[local].gather(0, target_ids).detach().cpu().numpy()
                values = values - values.max()
                probabilities = np.exp(values)
                rows[global_index] = probabilities / probabilities.sum()
            del logits, log_probs, encoded
        return pad_probabilities(row for row in rows if row is not None)
