"""Versionable records for licensed, human-reviewable instruction data."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator


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


@dataclass(frozen=True)
class TrainingExample:
    """One instruction/response record plus its source and review provenance."""

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
                "reviewer_id is required for a human review decision"
            )

        if len(set(self.defect_labels)) != len(self.defect_labels):
            raise DatasetFormatError("defect_labels must not contain duplicates")

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
            raise DatasetFormatError(f"invalid review decision or defect label: {exc}") from exc

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
        )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["review_decision"] = self.review_decision.value
        record["defect_labels"] = [label.value for label in self.defect_labels]
        return record


def load_jsonl(path: str | Path) -> list[TrainingExample]:
    """Load strict UTF-8 JSON Lines and report the exact bad line on failure."""
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
    """Atomically write normalized records so interrupted runs do not leave partial data."""
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
    """Stream records when a dataset is too large to keep in memory."""
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