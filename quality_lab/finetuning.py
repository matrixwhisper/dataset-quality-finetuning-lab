"""Real QLoRA training of an LLM quality judge from human-reviewed examples."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from quality_lab.config import LabConfig
from quality_lab.dataset import ReviewDecision, TrainingExample


def _training_messages(example: TrainingExample) -> tuple[dict[str, str], dict[str, str]]:
    if example.review_decision is ReviewDecision.UNREVIEWED:
        raise ValueError(f"{example.example_id}: cannot train from an unreviewed row")

    labels = sorted(label.value for label in example.defect_labels)
    if example.review_decision is ReviewDecision.ACCEPT and labels:
        raise ValueError(f"{example.example_id}: accepted example has defect labels")
    if example.review_decision is not ReviewDecision.ACCEPT and not labels:
        raise ValueError(f"{example.example_id}: defect decision has no defect labels")

    prompt = (
        "Review this instruction/response training example for quality defects. "
        "Use only these labels: ambiguous, incomplete, contradictory, duplicate, "
        "irrelevant, incorrect, other. Return JSON only with boolean is_defective "
        "and a labels array. Do not infer defects that are not evident.\n\n"
        f"Instruction:\n{example.instruction}\n\nResponse:\n{example.response}"
    )
    target = json.dumps(
        {"is_defective": bool(labels), "labels": labels},
        sort_keys=True,
    )
    return (
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    )


def _encode_row(tokenizer, example: TrainingExample, max_length: int) -> dict[str, list[int]]:
    user_message, assistant_message = _training_messages(example)
    prefix = tokenizer.apply_chat_template(
        [user_message],
        tokenize=True,
        add_generation_prompt=True,
    )
    full = tokenizer.apply_chat_template(
        [user_message, assistant_message],
        tokenize=True,
        add_generation_prompt=False,
    )
    if full[:len(prefix)] != prefix:
        raise ValueError(f"{example.example_id}: tokenizer chat prefix mismatch")
    if len(prefix) >= max_length:
        raise ValueError(f"{example.example_id}: prompt consumes the token budget")

    ids = full[:max_length]
    labels = [-100] * min(len(prefix), len(ids))
    labels.extend(ids[len(labels):])
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": labels,
    }


def _collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    import torch

    width = max(len(row["input_ids"]) for row in batch)
    input_ids, attention_masks, labels = [], [], []
    for row in batch:
        padding = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_token_id] * padding)
        attention_masks.append(row["attention_mask"] + [0] * padding)
        labels.append(row["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class _TokenizedRows:
    def __init__(self, rows: list[dict[str, list[int]]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        return {key: torch.tensor(value, dtype=torch.long) for key, value in self.rows[index].items()}


def _subset_fingerprint(examples: tuple[TrainingExample, ...]) -> str:
    digest = hashlib.sha256()
    for example in sorted(examples, key=lambda row: row.example_id):
        digest.update(json.dumps(example.to_record(), sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def train_qlora(
    config: LabConfig,
    train_examples: tuple[TrainingExample, ...],
    output_dir: str | Path,
) -> Path:
    """Fit and save an adapter using only the supplied training examples."""
    if not train_examples:
        raise ValueError("training split is empty")
    if len({item.example_id for item in train_examples}) != len(train_examples):
        raise ValueError("training split contains duplicate example IDs")
    for example in train_examples:
        _training_messages(example)

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("install the ML dependencies in a CUDA-enabled environment") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires a CUDA GPU; no CPU fallback is attempted")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        quantization_config=quantization,
        device_map={"": torch.cuda.current_device()},
    )
    resolved_revision = getattr(model.config, "_commit_hash", None) or config.model_revision
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    ))

    dataset = _TokenizedRows([
        _encode_row(tokenizer, example, config.max_sequence_length)
        for example in train_examples
    ])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output / "trainer"),
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
        fp16=not use_bf16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=lambda batch: _collate(batch, tokenizer.pad_token_id),
    )
    train_result = trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))

    manifest = {
        "status": "completed",
        "method": "QLoRA",
        "base_model_id": config.model_id,
        "base_model_revision": resolved_revision,
        "quantization": "4-bit NF4 with double quantization",
        "training_example_ids": [item.example_id for item in train_examples],
        "training_data_sha256": _subset_fingerprint(train_examples),
        "validation_used_for_gradient_updates": False,
        "test_accessed": False,
        "seed": config.seed,
        "lora": {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": "all-linear",
        },
        "training_metrics": train_result.metrics,
        "torch_version": torch.__version__,
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
    (output / "adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return output