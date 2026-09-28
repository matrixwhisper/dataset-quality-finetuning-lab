"""QLoRA supervised fine-tuning against HelpSteer2's five human ratings."""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from quality_lab.config import LabConfig
from quality_lab.dataset import SCORE_DIMENSIONS, TrainingExample


def _scores(example: TrainingExample) -> dict[str, int]:
    if example.quality_scores is None:
        raise ValueError(f"{example.example_id}: missing human quality scores")
    return {name: int(example.quality_scores[name]) for name in SCORE_DIMENSIONS}


def _user_text(example: TrainingExample) -> str:
    return (
        "Predict the five human ratings for this response. Each score is an "
        "integer from 0 to 4. Return JSON only, with exactly these keys: "
        "helpfulness, correctness, coherence, complexity, verbosity.\n\n"
        f"Prompt:\n{example.instruction}\n\nResponse:\n{example.response}"
    )


def _fit_user_ids(tokenizer, example: TrainingExample, token_budget: int) -> list[int]:
    prompt_text = example.instruction
    response_text = example.response

    for _ in range(40):
        text = (
            "Predict the five human ratings for this response. Each score is an "
            "integer from 0 to 4. Return JSON only, with exactly these keys: "
            "helpfulness, correctness, coherence, complexity, verbosity.\n\n"
            f"Prompt:\n{prompt_text}\n\nResponse:\n{response_text}"
        )
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if len(ids) <= token_budget:
            return ids

        excess = len(ids) - token_budget + 8
        response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
        if len(response_ids) > 16:
            response_text = tokenizer.decode(
                response_ids[:max(16, len(response_ids) - excess)],
                skip_special_tokens=True,
            )
            continue

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) > 16:
            prompt_text = tokenizer.decode(
                prompt_ids[:max(16, len(prompt_ids) - excess)],
                skip_special_tokens=True,
            )
            continue
        break

    raise ValueError(f"{example.example_id}: could not fit prompt into token budget")


def _encode(tokenizer, example: TrainingExample, max_length: int) -> dict[str, list[int]]:
    prefix = _fit_user_ids(tokenizer, example, max_length - 96)
    target = tokenizer(
        json.dumps(_scores(example), sort_keys=True),
        add_special_tokens=False,
    )["input_ids"]
    target.append(tokenizer.eos_token_id)

    room = max_length - len(prefix)
    if room < 1:
        raise ValueError(f"{example.example_id}: no room remains for target")
    target = target[:room]
    target[-1] = tokenizer.eos_token_id

    ids = prefix + target
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * len(prefix) + target,
    }


class _Rows:
    def __init__(self, rows: list[dict[str, list[int]]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


def _collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    import torch

    # Normalize values to Python lists before padding. This avoids the
    # Tensor-plus-list TypeError from the earlier Kaggle attempt.
    rows = [
        {
            key: value.tolist() if isinstance(value, torch.Tensor) else value
            for key, value in row.items()
        }
        for row in batch
    ]
    width = max(len(row["input_ids"]) for row in rows)
    ids, masks, labels = [], [], []
    for row in rows:
        padding = width - len(row["input_ids"])
        ids.append(row["input_ids"] + [pad_token_id] * padding)
        masks.append(row["attention_mask"] + [0] * padding)
        labels.append(row["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _fingerprint(examples: tuple[TrainingExample, ...]) -> str:
    digest = hashlib.sha256()
    for example in sorted(examples, key=lambda row: row.example_id):
        digest.update(json.dumps(example.to_record(), sort_keys=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def train_qlora(
    config: LabConfig,
    train_examples: tuple[TrainingExample, ...],
    output_dir: str | Path,
    max_examples: int | None = 512,
) -> Path:
    """Train a real 4-bit NF4 LoRA adapter on score-labeled training examples."""
    if not train_examples:
        raise ValueError("training split is empty")
    if len({row.example_id for row in train_examples}) != len(train_examples):
        raise ValueError("duplicate training example IDs")
    for row in train_examples:
        _scores(row)

    if max_examples is not None and len(train_examples) > max_examples:
        train_examples = tuple(
            random.Random(config.seed).sample(list(train_examples), max_examples)
        )

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from torch.utils.data import DataLoader
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("install CUDA-compatible PyTorch, Transformers, PEFT, bitsandbytes") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires a CUDA GPU")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
        # Single-device map avoids Trainer/DataParallel replication on Kaggle.
        device_map={"": torch.cuda.current_device()},
    )
    resolved_revision = getattr(model.config, "_commit_hash", None) or config.model_revision
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )

    max_length = min(config.max_sequence_length, 768)
    encoded = [_encode(tokenizer, row, max_length) for row in train_examples]
    dataset = _Rows(encoded)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, tokenizer.pad_token_id),
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    accumulation = config.gradient_accumulation_steps
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    optimizer_steps = 0
    started = time.perf_counter()

    for epoch in range(int(config.epochs + 0.999)):
        for step, batch in enumerate(loader, start=1):
            device = model.get_input_embeddings().weight.device
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=dtype):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch + 1}, step {step}")
            losses.append(float(loss.detach().cpu()))
            (loss / accumulation).backward()

            if step % accumulation == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if optimizer_steps % 10 == 0:
                    print(
                        f"epoch={epoch + 1} optimizer_step={optimizer_steps} "
                        f"loss={losses[-1]:.4f}"
                    )

    training_seconds = time.perf_counter() - started
    peak_gpu_bytes = torch.cuda.max_memory_allocated()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)

    manifest = {
        "status": "completed",
        "method": "QLoRA",
        "base_model_id": config.model_id,
        "base_model_revision": resolved_revision,
        "quantization": "4-bit NF4 with double quantization",
        "target": "five HelpSteer2 human ratings",
        "training_example_ids": [row.example_id for row in train_examples],
        "training_data_sha256": _fingerprint(train_examples),
        "validation_used_for_gradient_updates": False,
        "test_accessed": False,
        "seed": config.seed,
        "epochs": config.epochs,
        "optimizer_steps": optimizer_steps,
        "mean_training_loss": sum(losses) / len(losses),
        "training_seconds": training_seconds,
        "peak_gpu_memory_bytes": peak_gpu_bytes,
        "lora": {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": "all-linear",
        },
        "torch_version": torch.__version__,
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
    (destination / "adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination