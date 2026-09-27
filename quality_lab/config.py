"""Experiment settings shared by the CLI and model pipeline."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class LabConfig:
    seed: int = 2026
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: str = "main"
    data_path: Path = Path("data/reviewed.jsonl")
    run_dir: Path = Path("runs")
    max_sequence_length: int = 1536
    max_new_tokens: int = 128
    learning_rate: float = 2e-4
    epochs: float = 2.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_accumulation_steps: int = 8
    search_budget: int = 5

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if self.max_sequence_length < 128 or self.max_new_tokens < 16:
            raise ValueError("sequence and generation limits are too small")
        if self.learning_rate <= 0 or self.epochs <= 0:
            raise ValueError("learning_rate and epochs must be positive")
        if self.lora_rank < 1 or self.lora_alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.gradient_accumulation_steps < 1 or self.search_budget < 1:
            raise ValueError("gradient accumulation and search budget must be positive")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["data_path"] = str(self.data_path)
        result["run_dir"] = str(self.run_dir)
        return result

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "LabConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        values["data_path"] = Path(values["data_path"])
        values["run_dir"] = Path(values["run_dir"])
        return cls(**values)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass