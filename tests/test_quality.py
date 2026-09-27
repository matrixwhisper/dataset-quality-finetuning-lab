"""Tests for dataset records and deterministic quality-audit behavior."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from quality_lab.dataset import (
    DatasetFormatError,
    DefectLabel,
    ReviewDecision,
    TrainingExample,
    load_jsonl,
    write_jsonl,
)
from quality_lab.quality import audit_examples, human_label_coverage


def make_example(
    example_id: str,
    instruction: str = "Explain how sorted order works.",
    response: str = "It arranges values according to their ordering.",
    **overrides: object,
) -> TrainingExample:
    fields: dict[str, object] = {
        "example_id": example_id,
        "instruction": instruction,
        "response": response,
        "source_dataset": "local-reviewed-pilot",
        "source_record_id": example_id,
        "license_id": "CC-BY-4.0",
        "group_id": example_id,
    }
    fields.update(overrides)
    return TrainingExample(**fields)  # type: ignore[arg-type]


class DatasetRecordTests(unittest.TestCase):
    def test_human_decision_requires_reviewer_identity(self) -> None:
        with self.assertRaises(DatasetFormatError):
            make_example(
                "row-1",
                review_decision=ReviewDecision.ACCEPT,
            )

    def test_unknown_defect_label_is_rejected(self) -> None:
        with self.assertRaises(DatasetFormatError):
            TrainingExample.from_record({
                "example_id": "row-1",
                "instruction": "Explain how sorted order works.",
                "response": "It arranges values according to their ordering.",
                "source_dataset": "local-reviewed-pilot",
                "source_record_id": "source-1",
                "license_id": "CC-BY-4.0",
                "group_id": "group-1",
                "defect_labels": ["made_up_label"],
            })

    def test_jsonl_round_trip_preserves_labels_and_provenance(self) -> None:
        reviewed = make_example(
            "row-1",
            review_decision=ReviewDecision.REVISE,
            defect_labels=(DefectLabel.AMBIGUOUS,),
            reviewer_id="reviewer-01",
            review_note="The request did not define the expected output format.",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "examples.jsonl"
            write_jsonl([reviewed], path)
            loaded = load_jsonl(path)

        self.assertEqual(loaded, [reviewed])


class QualityAuditTests(unittest.TestCase):
    def test_exact_duplicate_is_reported(self) -> None:
        report = audit_examples([
            make_example("row-1"),
            make_example("row-2"),
        ])
        self.assertEqual(report.issue_counts["exact_duplicate"], 1)

    def test_same_prompt_with_different_responses_is_flagged(self) -> None:
        report = audit_examples([
            make_example("row-1"),
            make_example(
                "row-2",
                response="It arranges values using a different ordering rule.",
            ),
        ])
        self.assertEqual(report.issue_counts["conflicting_responses"], 2)

    def test_vague_prompt_and_placeholder_are_triaged_not_auto_rejected(self) -> None:
        report = audit_examples([
            make_example(
                "row-1",
                instruction="Explain this as appropriate, etc.",
                response="TODO: provide a clear answer.",
            ),
        ])
        self.assertEqual(report.issue_counts["possible_ambiguity"], 1)
        self.assertEqual(report.issue_counts["placeholder_text"], 1)
        self.assertEqual(report.rejected_count, 0)

    def test_human_label_coverage_excludes_unreviewed_records(self) -> None:
        report = human_label_coverage([
            make_example("row-1"),
            make_example(
                "row-2",
                review_decision=ReviewDecision.ACCEPT,
                defect_labels=(DefectLabel.INCOMPLETE,),
                reviewer_id="reviewer-01",
            ),
        ])
        self.assertEqual(report["human_reviewed_examples"], 1)
        self.assertEqual(report["review_fraction"], 0.5)
        self.assertEqual(
            report["human_defect_label_counts"],
            {"incomplete": 1},
        )

    def test_helpsteer_scores_are_summarized_without_relabeling(self) -> None:
        from quality_lab.dataset import SCORE_DIMENSIONS

        scores = {
            dimension: 3
            for dimension in SCORE_DIMENSIONS
        }
        example = make_example("rated-1", quality_scores=scores)
        report = audit_examples([example])

        self.assertEqual(report.scored_count, 1)
        self.assertEqual(report.unscored_count, 0)
        self.assertEqual(report.score_summary["helpfulness"]["mean"], 3)
        self.assertEqual(
            report.score_summary["helpfulness"]["histogram_0_to_4"]["3"],
            1,
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)