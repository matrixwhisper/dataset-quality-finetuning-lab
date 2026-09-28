"""Base-versus-QLoRA evaluation for five human-rated quality dimensions."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

from quality_lab.dataset import SCORE_DIMENSIONS, TrainingExample


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 96
    temperature: float = 0.0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if self.max_new_tokens < 16:
            raise ValueError("max_new_tokens must be at least 16")
        if not 0 <= self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("invalid temperature or top_p")


def _fingerprint(examples: Iterable[TrainingExample]) -> str:
    digest = hashlib.sha256()
    for example in sorted(examples, key=lambda row: row.example_id):
        digest.update(json.dumps(example.to_record(), sort_keys=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _prompt(example: TrainingExample) -> str:
    return (
        "Predict the five human ratings for this response. Return JSON only with "
        "integer values from 0 to 4 for exactly these keys: helpfulness, correctness, "
        "coherence, complexity, verbosity.\n\n"
        f"Prompt:\n{example.instruction}\n\nResponse:\n{example.response}"
    )


def parse_ratings(text: str) -> dict[str, int] | None:
    match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        values = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(values, dict) or set(values) != set(SCORE_DIMENSIONS):
        return None

    parsed: dict[str, int] = {}
    for dimension in SCORE_DIMENSIONS:
        value = values[dimension]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not 0 <= value <= 4:
            return None
        parsed[dimension] = max(0, min(4, int(round(value))))
    return parsed


class QualityJudge:
    """Shared 4-bit inference wrapper for base model and saved adapter."""

    def __init__(
        self,
        model_id: str,
        revision: str = "main",
        adapter_path: str | None = None,
    ) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("install CUDA-compatible Transformers, PEFT, and bitsandbytes") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("quality judge inference requires CUDA")

        self.torch = torch
        self.model_id = model_id
        self.adapter_path = adapter_path
        tokenizer_path = str(adapter_path) if adapter_path else model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, revision=revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            ),
            device_map={"": torch.cuda.current_device()},
        )
        if adapter_path:
            self.model = PeftModel.from_pretrained(
                self.model,
                str(adapter_path),
                is_trainable=False,
            )
        self.model.eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision

    def predict(
        self,
        example: TrainingExample,
        settings: GenerationSettings,
    ) -> dict[str, object]:
        torch = self.torch
        encoded = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": _prompt(example)}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=768 - settings.max_new_tokens,
        )
        device = self.model.get_input_embeddings().weight.device
        encoded = encoded.to(device)
        input_count = int(encoded.shape[-1])

        kwargs: dict[str, object] = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": settings.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if settings.temperature > 0:
            kwargs["temperature"] = settings.temperature
            kwargs["top_p"] = settings.top_p

        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(input_ids=encoded, **kwargs)
        torch.cuda.synchronize()

        text = self.tokenizer.decode(
            generated[0, input_count:],
            skip_special_tokens=True,
        )
        prediction = parse_ratings(text)
        expected = example.quality_scores
        if expected is None:
            raise ValueError(f"{example.example_id}: missing human quality scores")
        return {
            "example_id": example.example_id,
            "expected": {key: int(expected[key]) for key in SCORE_DIMENSIONS},
            "predicted": prediction,
            "parse_error": prediction is None,
            "latency_seconds": time.perf_counter() - started,
            "raw_output": text[:1000],
        }

    def metadata(self) -> dict[str, object]:
        torch = self.torch
        return {
            "model_id": self.model_id,
            "resolved_revision": self.revision,
            "adapter_path": self.adapter_path,
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }


def evaluate_judge(
    examples: Iterable[TrainingExample],
    judge: QualityJudge,
    settings: GenerationSettings,
    split_name: str,
) -> dict[str, object]:
    if split_name not in {"tuning", "test"}:
        raise ValueError("split_name must be tuning or test")
    records = tuple(examples)
    if not records:
        raise ValueError(f"{split_name} split is empty")
    if any(example.quality_scores is None for example in records):
        raise ValueError(f"{split_name} contains rows without HelpSteer2 ratings")

    rows = [judge.predict(example, settings) for example in records]
    valid = [row for row in rows if row["predicted"] is not None]
    mae_by_dimension = {}

    for dimension in SCORE_DIMENSIONS:
        errors = [
            abs(row["expected"][dimension] - row["predicted"][dimension])
            for row in valid
        ]
        mae_by_dimension[dimension] = sum(errors) / len(errors) if errors else None

    available = [value for value in mae_by_dimension.values() if value is not None]
    exact = sum(
        row["expected"] == row["predicted"]
        for row in valid
    )
    within_one = sum(
        all(
            abs(row["expected"][dimension] - row["predicted"][dimension]) <= 1
            for dimension in SCORE_DIMENSIONS
        )
        for row in valid
    )
    return {
        "model_id": judge.model_id,
        "adapter_path": judge.adapter_path,
        "split": split_name,
        "example_count": len(records),
        "valid_json_count": len(valid),
        "parse_error_count": len(records) - len(valid),
        "mae_by_dimension": mae_by_dimension,
        "macro_mae": sum(available) / len(available) if available else None,
        "exact_vector_accuracy": exact / len(valid) if valid else None,
        "within_one_all_dimensions": within_one / len(valid) if valid else None,
        "mean_latency_seconds": sum(row["latency_seconds"] for row in rows) / len(rows),
        "settings": asdict(settings),
        "model_metadata": judge.metadata(),
        "per_example": rows,
    }


def search_tuning(
    examples: tuple[TrainingExample, ...],
    judge_factory: Callable[[], QualityJudge],
    output_path: str | Path,
    token_budgets: tuple[int, ...] = (64, 96, 128),
) -> dict[str, object]:
    """Choose generation length on tuning data only, minimizing macro MAE."""
    if not examples:
        raise ValueError("tuning split is empty")
    if any(example.quality_scores is None for example in examples):
        raise ValueError("tuning rows need human scores")

    judge = judge_factory()
    history = []
    for token_budget in token_budgets:
        metrics = evaluate_judge(
            examples,
            judge,
            GenerationSettings(max_new_tokens=token_budget),
            "tuning",
        )
        history.append({"max_new_tokens": token_budget, "metrics": metrics})

    candidates = [
        item for item in history
        if item["metrics"]["macro_mae"] is not None
    ]
    if not candidates:
        raise RuntimeError("all tuning outputs failed score parsing")
    selected = min(candidates, key=lambda item: item["metrics"]["macro_mae"])

    artifact = {
        "search_split": "tuning",
        "test_accessed": False,
        "tuning_example_ids": [example.example_id for example in examples],
        "tuning_data_sha256": _fingerprint(examples),
        "selected_settings": {"max_new_tokens": selected["max_new_tokens"]},
        "selected_metrics": selected["metrics"],
        "history": history,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact