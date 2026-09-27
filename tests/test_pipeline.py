"""CPU tests for detector fitting, review persistence, and pipeline safeguards."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quality_lab.dataset import DefectLabel, ReviewDecision, TrainingExample
from quality_lab.detector import QualityRiskDetector, active_review_queue, evaluate_detector
from quality_lab.review import ReviewStore


def row(
    example_id: str,
    defective: bool,
    group_id: str | None = None,
) -> TrainingExample:
    return TrainingExample(
        example_id=example_id,
        instruction=f"Please classify this training request example number {example_id}.",
        response=(
            "This response provides a complete explanation with clear steps."
            if not defective
            else "TODO: fix this incomplete response."
        ),
        source_dataset="reviewed-test-fixture",
        source_record_id=example_id,
        license_id="CC-BY-4.0",
        group_id=group_id or example_id,
        review_decision=ReviewDecision.REVISE if defective else ReviewDecision.ACCEPT,
        defect_labels=(DefectLabel.INCOMPLETE,) if defective else (),
        reviewer_id="reviewer-test",
    )


class DetectorTests(unittest.TestCase):
    def test_fit_and_evaluate_on_human_labels(self) -> None:
        training = [
            row(f"clean-{index}", False) for index in range(6)
        ] + [
            row(f"defect-{index}", True) for index in range(6)
        ]
        detector = QualityRiskDetector().fit(training)
        result = evaluate_detector(detector, training)
        self.assertEqual(result["task_count"], 12)
        self.assertIn("f1", result)
        self.assertIn("brier_score", result)

    def test_review_queue_returns_only_unreviewed_examples(self) -> None:
        labeled = [
            row(f"clean-{index}", False) for index in range(6)
        ] + [
            row(f"defect-{index}", True) for index in range(6)
        ]
        detector = QualityRiskDetector().fit(labeled)
        pool = [
            TrainingExample(
                example_id=f"pool-{index}",
                instruction=f"Review this unique instruction number {index}.",
                response=f"This response gives enough detail for example {index}.",
                source_dataset="reviewed-test-fixture",
                source_record_id=f"pool-{index}",
                license_id="CC-BY-4.0",
                group_id=f"pool-{index}",
            )
            for index in range(4)
        ]
        queue = active_review_queue(detector, pool, batch_size=2)
        self.assertEqual(len(queue), 2)
        self.assertTrue(all(item["example_id"].startswith("pool-") for item in queue))


class ReviewLedgerTests(unittest.TestCase):
    def test_sqlite_review_history_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "reviews.sqlite"
            store = ReviewStore(database)
            store.import_examples([row("one", True)])
            store.record_review(
                "one",
                "reviewer-a",
                ReviewDecision.REVISE,
                (DefectLabel.INCOMPLETE,),
                "The answer omits the requested explanation.",
            )
            output = store.export_current(Path(directory) / "reviewed.jsonl")
            self.assertTrue(output.exists())
            history = store.event_history("one")
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["reviewer_id"], "reviewer-a")


if __name__ == "__main__":
    unittest.main(verbosity=2)