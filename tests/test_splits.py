"""Regression tests for deterministic group-safe dataset splitting."""

from __future__ import annotations

import unittest

from quality_lab.dataset import DefectLabel, TrainingExample
from quality_lab.splits import DatasetLeakageError, assert_no_leakage, make_splits


def example(
    example_id: str,
    group_id: str | None = None,
    instruction: str | None = None,
    defect_labels: tuple[DefectLabel, ...] = (),
) -> TrainingExample:
    return TrainingExample(
        example_id=example_id,
        instruction=instruction or f"Explain the behavior of function {example_id}.",
        response=f"This is a sufficiently detailed response for {example_id}.",
        source_dataset="reviewed-pilot",
        source_record_id=example_id,
        license_id="CC-BY-4.0",
        group_id=group_id or example_id,
        defect_labels=defect_labels,
    )


class GroupSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.examples = [
            example("a"),
            example("b"),
            example("c"),
            example("d", defect_labels=(DefectLabel.AMBIGUOUS,)),
            example("e", defect_labels=(DefectLabel.INCOMPLETE,)),
            example("f"),
            example("g"),
            example("h"),
            example("i"),
        ]

    def test_split_is_deterministic_and_nonempty(self) -> None:
        first = make_splits(self.examples, seed=41)
        second = make_splits(self.examples, seed=41)

        self.assertEqual(first.ids(), second.ids())
        self.assertTrue(all(first.counts()[name] > 0 for name in first.counts()))

    def test_shared_group_and_identical_prompt_stay_together(self) -> None:
        records = [
            example("a", group_id="same-source"),
            example("b", group_id="same-source"),
            example("c"),
            example("d"),
            example("e", instruction="Explain a stable sorting algorithm."),
            example("f", instruction="Explain a stable sorting algorithm."),
        ]
        splits = make_splits(records, seed=9)
        owner = {
            row.example_id: split_name
            for split_name in ("train", "validation", "test")
            for row in splits.get(split_name)
        }

        self.assertEqual(owner["a"], owner["b"])
        self.assertEqual(owner["e"], owner["f"])
        assert_no_leakage(splits)

    def test_too_few_independent_components_is_rejected(self) -> None:
        records = [
            example("a", group_id="one-component"),
            example("b", group_id="one-component"),
            example("c", group_id="one-component"),
        ]
        with self.assertRaises(DatasetLeakageError):
            make_splits(records)

    def test_duplicate_example_ids_are_rejected(self) -> None:
        records = [example("same"), example("same"), example("different")]
        with self.assertRaises(DatasetLeakageError):
            make_splits(records)


if __name__ == "__main__":
    unittest.main(verbosity=2)