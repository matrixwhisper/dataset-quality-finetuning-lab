"""Base-versus-QLoRA evaluation and validation-only generation search."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

from quality_lab.dataset import ReviewDecision, TrainingExample


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if self.max_new_tokens < 16:
            raise ValueError("max_new_tokens must be at least 16")
        if not 0 <= self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("invalid temperature or top_p")


def _prompt(example: TrainingExample) -> str:
    return (
        "Decide whether this training example contains a quality defect. "
        "Allowed labels: ambiguous, incomplete, contradictory, duplicate, "
        "irrelevant, incorrect, other. Return exactly one JSON object with "
        "keys is_defective (boolean) and labels (array of allowed label strings). "
        "Use an empty labels array for a clean example.\n\n"
        f"Instruction:\n{example.instruction}\n\nResponse:\n{example.response}"
    )


def _parse_prediction(text: str) -> tuple[bool, list[str], bool]:
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match is None:
        return False, [], True
    try:
        value = json.loads(match.group(0))
        defective = value["is_defective"]
        labels = value["labels"]
        if not isinstance(defective, bool) or not isinstance(labels, list):
            return False, [], True
        return defective, [str(label) for label in labels], False
    except (json.JSONDecodeError, KeyError, TypeError):
        return False, [], True


class QualityJudge:
    """Shared Transformers generation wrapper for base and adapter evaluation."""

    def __init__(
        self,
        model_id: str,
        revision: str = "main",
        adapter_path: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("install Transformers, PyTorch, and bitsandbytes") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("LLM inference requires a CUDA GPU in this experiment")

        self.torch = torch
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            quantization_config=quantization,
            device_map="auto",
        )
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
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
        latency = time.perf_counter() - started
        text = self.tokenizer.decode(
            generated[0, input_count:],
            skip_special_tokens=True,
        )
        defective, labels, parse_error = _parse_prediction(text)
        return {
            "example_id": example.example_id,
            "expected_defective": bool(example.defect_labels),
            "predicted_defective": defective,
            "predicted_labels": labels,
            "parse_error": parse_error,
            "latency_seconds": latency,
            "raw_output": text[:2000],
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
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    if split_name not in {"validation", "test"}:
        raise ValueError("LLM benchmark evaluation must name validation or test")
    records = list(examples)
    if not records:
        raise ValueError(f"{split_name} split is empty")
    if any(item.review_decision is ReviewDecision.UNREVIEWED for item in records):
        raise ValueError(f"{split_name} includes records without human labels")

    rows = [judge.predict(item, settings) for item in records]
    expected = [int(row["expected_defective"]) for row in rows]
    predicted = [int(row["predicted_defective"]) for row in rows]
    return {
        "model_id": judge.model_id,
        "adapter_path": judge.adapter_path,
        "split": split_name,
        "task_count": len(records),
        "accuracy": float(accuracy_score(expected, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
        "precision": float(precision_score(expected, predicted, zero_division=0)),
        "recall": float(recall_score(expected, predicted, zero_division=0)),
        "f1": float(f1_score(expected, predicted, zero_division=0)),
        "confusion_matrix_labels_clean_defective": confusion_matrix(
            expected, predicted, labels=[0, 1]
        ).tolist(),
        "parse_error_count": sum(bool(row["parse_error"]) for row in rows),
        "mean_latency_seconds": sum(float(row["latency_seconds"]) for row in rows) / len(rows),
        "settings": asdict(settings),
        "model_metadata": judge.metadata(),
        "per_example": rows,
    }


def _neighbors(settings: GenerationSettings) -> list[GenerationSettings]:
    values = [
        replace(settings, max_new_tokens=max(32, settings.max_new_tokens - 32)),
        replace(settings, max_new_tokens=min(256, settings.max_new_tokens + 32)),
        replace(settings, temperature=0.0 if settings.temperature else 0.2),
    ]
    return list(dict.fromkeys(value for value in values if value != settings))


def hill_climb_validation(
    validation_examples: tuple[TrainingExample, ...],
    judge_factory: Callable[[], QualityJudge],
    initial_settings: GenerationSettings,
    evaluation_budget: int,
    output_path: str | Path,
) -> tuple[GenerationSettings, list[dict[str, object]]]:
    """Select decoding settings using validation labels only."""
    if not validation_examples:
        raise ValueError("validation split is empty")
    if evaluation_budget < 1:
        raise ValueError("evaluation budget must be positive")
    if any(row.review_decision is ReviewDecision.UNREVIEWED for row in validation_examples):
        raise ValueError("validation search requires human-reviewed labels")

    judge = judge_factory()
    current = initial_settings
    history: list[dict[str, object]] = []
    best_f1 = -1.0
    evaluated: set[GenerationSettings] = set()

    for step in range(evaluation_budget):
        proposals = [current] if step == 0 else _neighbors(current)
        candidate = next((item for item in proposals if item not in evaluated), None)
        if candidate is None:
            break
        evaluated.add(candidate)
        result = evaluate_judge(validation_examples, judge, candidate, "validation")
        score = float(result["f1"])
        history.append({
            "step": step,
            "split": "validation",
            "settings": asdict(candidate),
            "metrics": result,
        })
        if score > best_f1:
            best_f1 = score
            current = candidate

    artifact = {
        "search_split": "validation",
        "test_accessed": False,
        "validation_example_ids": [row.example_id for row in validation_examples],
        "selected_settings": asdict(current),
        "history": history,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return current, history