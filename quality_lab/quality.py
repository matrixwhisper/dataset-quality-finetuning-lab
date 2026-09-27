"""Quality auditing for human-rated response datasets."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Iterable

from quality_lab.dataset import ReviewDecision, SCORE_DIMENSIONS, TrainingExample


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
    scored_count: int
    unscored_count: int
    rejected_count: int
    score_summary: dict[str, dict[str, object]]
    issue_counts: dict[str, int]
    issues: tuple[QualityIssue, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "record_count": self.record_count,
            "scored_count": self.scored_count,
            "unscored_count": self.unscored_count,
            "rejected_count": self.rejected_count,
            "score_summary": self.score_summary,
            "issue_counts": self.issue_counts,
            "issues": [asdict(issue) for issue in self.issues],
        }


def normalize_text(value: str) -> str:
    """Normalize Unicode and whitespace for exact comparisons."""
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.casefold().split())


def audit_examples(examples: Iterable[TrainingExample]) -> QualityReport:
    """Flag structural risks and summarize human ratings without relabeling them."""
    records = list(examples)
    issues: list[QualityIssue] = []
    seen_ids: set[str] = set()
    exact_pairs: dict[tuple[str, str], str] = {}
    responses_by_prompt: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    scores_by_dimension: dict[str, list[int]] = {
        dimension: [] for dimension in SCORE_DIMENSIONS
    }
    scored_count = 0

    for example in records:
        if example.example_id in seen_ids:
            issues.append(QualityIssue(
                example_id=example.example_id,
                code="duplicate_id",
                severity="error",
                message="example_id occurs more than once",
            ))
        seen_ids.add(example.example_id)

        if not example.license_id.strip() or example.license_id.casefold() in {
            "unknown", "unspecified", "n/a",
        }:
            issues.append(QualityIssue(
                example_id=example.example_id,
                code="missing_license",
                severity="error",
                message="record needs a verifiable license identifier",
            ))

        prompt = normalize_text(example.instruction)
        response = normalize_text(example.response)

        if len(prompt.split()) < 3:
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
        if any(phrase in prompt for phrase in VAGUE_PHRASES):
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

        pair = (prompt, response)
        previous_id = exact_pairs.get(pair)
        if previous_id is not None:
            issues.append(QualityIssue(
                example_id=example.example_id,
                code="exact_duplicate",
                severity="review",
                message=f"normalized prompt/response pair duplicates {previous_id}",
            ))
        else:
            exact_pairs[pair] = example.example_id

        if prompt and response:
            responses_by_prompt[prompt][response].append(example.example_id)

        if example.quality_scores is not None:
            scored_count += 1
            for dimension in SCORE_DIMENSIONS:
                scores_by_dimension[dimension].append(
                    example.quality_scores[dimension]
                )

    for responses in responses_by_prompt.values():
        if len(responses) > 1:
            for example_ids in responses.values():
                for example_id in example_ids:
                    issues.append(QualityIssue(
                        example_id=example_id,
                        code="conflicting_responses",
                        severity="info",
                        message="prompt has multiple responses; expected for preference/rating data",
                    ))

    score_summary: dict[str, dict[str, object]] = {}
    for dimension, scores in scores_by_dimension.items():
        histogram = Counter(scores)
        score_summary[dimension] = {
            "count": len(scores),
            "mean": round(mean(scores), 4) if scores else None,
            "histogram_0_to_4": {
                str(score): histogram.get(score, 0) for score in range(5)
            },
        }

    issue_counts = Counter(issue.code for issue in issues)
    return QualityReport(
        record_count=len(records),
        scored_count=scored_count,
        unscored_count=len(records) - scored_count,
        rejected_count=sum(
            example.review_decision is ReviewDecision.REJECT
            for example in records
        ),
        score_summary=score_summary,
        issue_counts=dict(sorted(issue_counts.items())),
        issues=tuple(issues),
    )


def human_label_coverage(examples: Iterable[TrainingExample]) -> dict[str, object]:
    """Report score coverage separately from optional categorical review labels."""
    records = list(examples)
    scored = [example for example in records if example.quality_scores is not None]
    reviewed = [
        example for example in records
        if example.review_decision is not ReviewDecision.UNREVIEWED
    ]
    defect_label_counts = Counter(
        label.value
        for example in reviewed
        for label in example.defect_labels
    )
    return {
        "total_examples": len(records),
        "human_score_coverage": len(scored) / len(records) if records else 0.0,
        "scored_examples": len(scored),
        "human_reviewed_examples": len(reviewed),
        "review_fraction": len(reviewed) / len(records) if records else 0.0,
        "human_defect_label_counts": dict(sorted(defect_label_counts.items())),
        "score_dimensions": list(SCORE_DIMENSIONS),
        "score_source": "dataset-provided human ratings; not project-invented labels",
    }