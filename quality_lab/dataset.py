"""Provenance-aware training examples for human-rated response quality."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator


SCORE_DIMENSIONS = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)


class DatasetFormatError(ValueError):
    """Raised when a source record cannot safely enter the benchmark."""


class DefectLabel(str, Enum):
    AMBIGUOUS = "ambiguous"
    INCOMPLETE = "incomplete"
    CONTRADICTORY = "contradictory"
    DUPLICATE = "duplicate"
    IRRELEVANT = "irrelevant"
    INCORRECT = "incorrect"
    OTHER = "other"


class ReviewDecision(str, Enum):
    UNREVIEWED = "unreviewed"
    ACCEPT = "accept"
    REVISE = "revise"
    REJECT = "reject"


def normalized_group_id(prompt: str) -> str:
    """Hash a normalized prompt so alternate responses remain in one split."""
    normalized = unicodedata.normalize("NFKC", prompt).casefold()
    normalized = " ".join(normalized.split())
    if not normalized:
        raise DatasetFormatError("prompt must not be empty")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrainingExample:
    """One prompt/response example with source, scores, and review provenance."""

    example_id: str
    instruction: str
    response: str
    source_dataset: str
    source_record_id: str
    license_id: str
    group_id: str
    review_decision: ReviewDecision = ReviewDecision.UNREVIEWED
    defect_labels: tuple[DefectLabel, ...] = ()
    reviewer_id: str | None = None
    review_note: str = ""
    quality_scores: dict[str, int] | None = None

    def __post_init__(self) -> None:
        required = (
            "example_id",
            "instruction",
            "response",
            "source_dataset",
            "source_record_id",
            "license_id",
            "group_id",
        )
        for name in required:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DatasetFormatError(f"{name} must be a non-empty string")

        if self.review_decision is not ReviewDecision.UNREVIEWED and not self.reviewer_id:
            raise DatasetFormatError(
                "reviewer_id is required for a manual review decision"
            )

        if len(set(self.defect_labels)) != len(self.defect_labels):
            raise DatasetFormatError("defect_labels must not contain duplicates")

        if self.quality_scores is not None:
            if set(self.quality_scores) != set(SCORE_DIMENSIONS):
                missing = sorted(set(SCORE_DIMENSIONS) - set(self.quality_scores))
                extra = sorted(set(self.quality_scores) - set(SCORE_DIMENSIONS))
                raise DatasetFormatError(
                    f"quality_scores must contain exactly {SCORE_DIMENSIONS}; "
                    f"missing={missing}, extra={extra}"
                )
            for dimension, score in self.quality_scores.items():
                if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 4:
                    raise DatasetFormatError(
                        f"{dimension} score must be an integer from 0 to 4"
                    )

    @classmethod
    def from_helpsteer2(
        cls,
        record: dict[str, Any],
        source_split: str,
        row_index: int,
    ) -> "TrainingExample":
        """Convert one HelpSteer2 row without inventing defect labels.

        HelpSteer2 provides aggregated human ratings. Its scores are retained
        exactly as published; they are not converted into accept/reject labels.
        """
        required = ("prompt", "response", *SCORE_DIMENSIONS)
        missing = [field for field in required if field not in record]
        if missing:
            raise DatasetFormatError(
                "HelpSteer2 record is missing field(s): " + ", ".join(missing)
            )
        if source_split not in {"train", "validation"}:
            raise DatasetFormatError(
                "source_split must be 'train' or 'validation'"
            )

        prompt = record["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise DatasetFormatError("HelpSteer2 prompt must be non-empty text")

        source_record_id = f"helpsteer2:{source_split}:{row_index}"
        return cls(
            example_id=source_record_id,
            instruction=prompt,
            response=record["response"],
            source_dataset="nvidia/HelpSteer2",
            source_record_id=source_record_id,
            license_id="CC-BY-4.0",
            group_id=normalized_group_id(prompt),
            quality_scores={
                dimension: record[dimension]
                for dimension in SCORE_DIMENSIONS
            },
        )

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "TrainingExample":
        required = {
            "example_id",
            "instruction",
            "response",
            "source_dataset",
            "source_record_id",
            "license_id",
            "group_id",
        }
        missing = sorted(required - record.keys())
        if missing:
            raise DatasetFormatError(
                "missing required field(s): " + ", ".join(missing)
            )

        try:
            decision = ReviewDecision(
                record.get("review_decision", ReviewDecision.UNREVIEWED.value)
            )
            labels = tuple(
                DefectLabel(label) for label in record.get("defect_labels", [])
            )
        except (TypeError, ValueError) as exc:
            raise DatasetFormatError(
                f"invalid review decision or defect label: {exc}"
            ) from exc

        scores = record.get("quality_scores")
        if scores is not None and not isinstance(scores, dict):
            raise DatasetFormatError("quality_scores must be an object")

        return cls(
            example_id=record["example_id"],
            instruction=record["instruction"],
            response=record["response"],
            source_dataset=record["source_dataset"],
            source_record_id=record["source_record_id"],
            license_id=record["license_id"],
            group_id=record["group_id"],
            review_decision=decision,
            defect_labels=labels,
            reviewer_id=record.get("reviewer_id"),
            review_note=record.get("review_note", ""),
            quality_scores=scores,
        )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["review_decision"] = self.review_decision.value
        record["defect_labels"] = [
            label.value for label in self.defect_labels
        ]
        return record


def load_jsonl(path: str | Path) -> list[TrainingExample]:
    """Load strict UTF-8 JSON Lines and identify malformed source lines."""
    source = Path(path)
    examples: list[TrainingExample] = []

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise DatasetFormatError("each JSONL line must be an object")
                examples.append(TrainingExample.from_record(record))
            except (json.JSONDecodeError, DatasetFormatError) as exc:
                raise DatasetFormatError(
                    f"{source}:{line_number}: {exc}"
                ) from exc

    if not examples:
        raise DatasetFormatError(f"{source} contains no data records")
    return examples


def write_jsonl(
    examples: Iterable[TrainingExample],
    path: str | Path,
) -> Path:
    """Atomically write normalized records to UTF-8 JSON Lines."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for example in examples:
                handle.write(
                    json.dumps(
                        example.to_record(),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    return destination


def iter_records(path: str | Path) -> Iterator[TrainingExample]:
    """Stream records when a JSONL dataset should not be loaded all at once."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise DatasetFormatError("each JSONL line must be an object")
                yield TrainingExample.from_record(record)
            except (json.JSONDecodeError, DatasetFormatError) as exc:
                raise DatasetFormatError(
                    f"{source}:{line_number}: {exc}"
                ) from exc