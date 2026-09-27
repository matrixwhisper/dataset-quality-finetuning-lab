"""Deterministic data-quality checks and human-review summaries.

Rule-based findings are triage signals, not ground-truth defect labels.
Semantic and model-based checks are added in later stages.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable

from quality_lab.dataset import DefectLabel, TrainingExample


VAGUE_PHRASES = (
    "as appropriate",
    "as needed",
    "and so on",
    "etc.",
    "some details",
)
PLACEHOLDER_PATTERN = re.compile(
    r"\b(?:TODO|TBD|INSERT[_ -]HERE|PLACEHOLDER)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QualityIssue:
    example_id: str
    code: str
    severity: str
    message: str


@dataclass(frozen=True)
class QualityReport:
    record_count: int
    reviewed_count: int
    accepted_count: int
    revised_count: int
    rejected_count: int
    unreviewed_count: int
    defect_label_counts: dict[str, int]
    issue_counts: dict[str, int]
    issues: tuple[QualityIssue, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "record_count": self.record_count,
            "reviewed_count": self.reviewed_count,
            "accepted_count": self.accepted_count,
            "revised_count": self.revised_count,
            "rejected_count": self.rejected_count,
            "unreviewed_count": self.unreviewed_count,
            "defect_label_counts": self.defect_label_counts,
            "issue_counts": self.issue_counts,
            "issues": [asdict(issue) for issue in self.issues],
        }


def normalize_text(value: str) -> str:
    """Normalize Unicode and whitespace for exact-match comparisons."""
    unicode_normalized = unicodedata.normalize("NFKC", value)
    return " ".join(unicode_normalized.casefold().split())


def audit_examples(examples: Iterable[TrainingExample]) -> QualityReport:
    """Audit structure and obvious quality risks without pretending to judge truth."""
    records = list(examples)
    issues: list[QualityIssue] = []
    seen_ids: set[str] = set()
    exact_pairs: dict[tuple[str, str], str] = {}
    answers_by_instruction: dict[str, dict[str, str]] = defaultdict(dict)
    decision_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()

    for example in records:
        decision_counts[example.review_decision.value] += 1
        for label in example.defect_labels:
            label_counts[label.value] += 1

        if example.example_id in seen_ids:
            issues.append(QualityIssue(
                example_id=example.example_id,
                code="duplicate_id",
                severity="error",
                message="example_id occurs more than once",
            ))
        seen_ids.add(example.example_id)

        if not example.license_id.strip() or example.license_id.casefold() in {
            "unknown",
            "unspecified",
            "n/a",
        }:
            issues.append(QualityIssue(
                example_id=example.example_id,
                code="missing_license",
                severity="error",
                message="record needs a verifiable license identifier",
            ))

        instruction = normalize_text(example.instruction)
        response = normalize_text(example.response)

        if len(instruction.split()) < 3:
            issues.append(QualityIssue(
                example_id=example.example_id,
                code="very_short_instruction",
                severity="warning",
                message="instruction has fewer than three normalized words",
            ))

        if len(response.split()) < 3:
            issues.append(QualityIssue(
                example_id=example.example_id,
                code="very_short_response",
                severity="warning",
                message="response has fewer than three normalized words",
            ))

        if any(phrase in instruction for phrase in VAGUE_PHRASES):
            issues.append(QualityIssue(
                example_id=example.example_id,
                code="possible_ambiguity",
                severity="review",
                message="instruction contains a phrase from the ambiguity triage list",
            ))

        if PLACEHOLDER_PATTERN.search(example.instruction) or PLACEHOLDER_PATTERN.search(example.response):
            issues.append(QualityIssue(
                example_id=example.example_id,
                code="placeholder_text",
                severity="review",
                message="instruction or response contains a placeholder token",
            ))

        pair_key = (instruction, response)
        previous_id = exact_pairs.get(pair_key)
        if previous_id is not None:
            issues.append(QualityIssue(
                example_id=example.example_id,
                code="exact_duplicate",
                severity="review",
                message=f"normalized prompt/response pair duplicates {previous_id}",
            ))
        else:
            exact_pairs[pair_key] = example.example_id

        if instruction and response:
            answers_by_instruction[instruction][response] = example.example_id

    for instruction, answers in answers_by_instruction.items():
        if len(answers) > 1:
            for example_id in answers.values():
                issues.append(QualityIssue(
                    example_id=example_id,
                    code="conflicting_responses",
                    severity="review",
                    message="identical normalized instruction has multiple responses; reviewer adjudication required",
                ))

    issue_counts = Counter(issue.code for issue in issues)
    reviewed = (
        decision_counts["accept"]
        + decision_counts["revise"]
        + decision_counts["reject"]
    )

    return QualityReport(
        record_count=len(records),
        reviewed_count=reviewed,
        accepted_count=decision_counts["accept"],
        revised_count=decision_counts["revise"],
        rejected_count=decision_counts["reject"],
        unreviewed_count=decision_counts["unreviewed"],
        defect_label_counts=dict(sorted(label_counts.items())),
        issue_counts=dict(sorted(issue_counts.items())),
        issues=tuple(issues),
    )


def human_label_coverage(examples: Iterable[TrainingExample]) -> dict[str, object]:
    """Summarize adjudicated examples separately from automatic audit signals."""
    records = list(examples)
    reviewed = [
        example for example in records
        if example.review_decision.value != "unreviewed"
    ]
    counts = Counter(
        label.value
        for example in reviewed
        for label in example.defect_labels
    )
    return {
        "total_examples": len(records),
        "human_reviewed_examples": len(reviewed),
        "review_fraction": len(reviewed) / len(records) if records else 0.0,
        "human_defect_label_counts": dict(sorted(counts.items())),
        "labels_are_human_adjudicated": True,
    }