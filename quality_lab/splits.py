"""Deterministic group-aware splits with leakage assertions."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

from quality_lab.dataset import TrainingExample
from quality_lab.quality import normalize_text


SplitName = Literal["train", "validation", "test"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "validation", "test")


class DatasetLeakageError(ValueError):
    """Raised when related examples cross dataset partitions."""


@dataclass(frozen=True)
class DatasetSplits:
    train: tuple[TrainingExample, ...]
    validation: tuple[TrainingExample, ...]
    test: tuple[TrainingExample, ...]
    seed: int
    component_by_example_id: dict[str, str]

    def get(self, name: str) -> tuple[TrainingExample, ...]:
        if name not in SPLIT_NAMES:
            raise ValueError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
        return getattr(self, name)

    def counts(self) -> dict[str, int]:
        return {name: len(self.get(name)) for name in SPLIT_NAMES}

    def ids(self) -> dict[str, list[str]]:
        return {
            name: [example.example_id for example in self.get(name)]
            for name in SPLIT_NAMES
        }


class _DisjointSet:
    """Union-find structure for joining records that must stay together."""

    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return

        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _related_components(
    examples: tuple[TrainingExample, ...],
) -> dict[str, list[TrainingExample]]:
    """Join rows by declared group or identical normalized instruction."""
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise DatasetLeakageError("example_id values must be unique before splitting")

    union_find = _DisjointSet(ids)
    first_by_group: dict[str, str] = {}
    first_by_instruction: dict[str, str] = {}

    for example in examples:
        group_previous = first_by_group.setdefault(example.group_id, example.example_id)
        union_find.union(group_previous, example.example_id)

        instruction_key = normalize_text(example.instruction)
        if instruction_key:
            instruction_previous = first_by_instruction.setdefault(
                instruction_key,
                example.example_id,
            )
            union_find.union(instruction_previous, example.example_id)

    components: dict[str, list[TrainingExample]] = defaultdict(list)
    for example in examples:
        components[union_find.find(example.example_id)].append(example)
    return dict(components)


def _assignment_cost(
    split_index: int,
    group_size: int,
    group_labels: Counter[str],
    sizes: list[int],
    label_counts: list[Counter[str]],
    target_sizes: list[float],
    target_labels: dict[str, float],
) -> float:
    """Estimate size and multilabel imbalance after adding one component."""
    candidate_sizes = list(sizes)
    candidate_sizes[split_index] += group_size

    size_error = sum(
        ((actual - target) / max(target, 1.0)) ** 2
        for actual, target in zip(candidate_sizes, target_sizes)
    )

    label_error = 0.0
    for label, target in target_labels.items():
        for index in range(len(SPLIT_NAMES)):
            actual = label_counts[index][label]
            if index == split_index:
                actual += group_labels[label]
            label_error += ((actual - target) / max(target, 1.0)) ** 2

    return size_error + 0.2 * label_error


def make_splits(
    examples: tuple[TrainingExample, ...] | list[TrainingExample],
    seed: int = 2026,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> DatasetSplits:
    """Split records while keeping shared groups and identical prompts together.

    Split proportions are targets rather than exact guarantees because connected
    components are indivisible. Each partition receives at least one component.
    """
    records = tuple(examples)
    if not records:
        raise ValueError("cannot split an empty dataset")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("train and validation fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("fractions must leave a non-empty test partition")

    components = _related_components(records)
    if len(components) < len(SPLIT_NAMES):
        raise DatasetLeakageError(
            "at least three independent components are required for train, validation, and test"
        )

    fractions = (
        train_fraction,
        validation_fraction,
        1.0 - train_fraction - validation_fraction,
    )
    target_sizes = [len(records) * fraction for fraction in fractions]

    total_labels = Counter(
        label.value
        for example in records
        for label in example.defect_labels
    )
    target_labels = {
        label: count * fractions[index]
        for label, count in total_labels.items()
        for index in range(1)
    }

    rng = random.Random(seed)
    component_items = list(components.items())
    rng.shuffle(component_items)
    component_items.sort(key=lambda item: -len(item[1]))

    sizes = [0, 0, 0]
    label_counts = [Counter(), Counter(), Counter()]
    assigned: dict[str, int] = {}

    for index, (component_id, members) in enumerate(component_items):
        group_labels = Counter(
            label.value
            for example in members
            for label in example.defect_labels
        )

        if index < len(SPLIT_NAMES):
            selected_split = index
        else:
            costs = [
                _assignment_cost(
                    split_index,
                    len(members),
                    group_labels,
                    sizes,
                    label_counts,
                    target_sizes,
                    {
                        label: count * fractions[split_index]
                        for label, count in total_labels.items()
                    },
                )
                for split_index in range(len(SPLIT_NAMES))
            ]
            selected_split = min(range(len(costs)), key=costs.__getitem__)

        assigned[component_id] = selected_split
        sizes[selected_split] += len(members)
        label_counts[selected_split].update(group_labels)

    partitions: list[list[TrainingExample]] = [[], [], []]
    component_by_example_id: dict[str, str] = {}
    for component_id, members in components.items():
        split_index = assigned[component_id]
        partitions[split_index].extend(members)
        for example in members:
            component_by_example_id[example.example_id] = component_id

    ordered_partitions = [
        tuple(sorted(partition, key=lambda example: example.example_id))
        for partition in partitions
    ]
    result = DatasetSplits(
        train=ordered_partitions[0],
        validation=ordered_partitions[1],
        test=ordered_partitions[2],
        seed=seed,
        component_by_example_id=component_by_example_id,
    )
    assert_no_leakage(result)
    return result


def assert_no_leakage(splits: DatasetSplits) -> None:
    """Fail if IDs, declared groups, or normalized prompts cross partitions."""
    owner_by_example_id: dict[str, str] = {}
    owner_by_group: dict[str, str] = {}
    owner_by_instruction: dict[str, str] = {}

    for split_name in SPLIT_NAMES:
        for example in splits.get(split_name):
            previous_split = owner_by_example_id.setdefault(example.example_id, split_name)
            if previous_split != split_name:
                raise DatasetLeakageError(
                    f"example {example.example_id!r} appears in multiple splits"
                )

            previous_split = owner_by_group.setdefault(example.group_id, split_name)
            if previous_split != split_name:
                raise DatasetLeakageError(
                    f"group {example.group_id!r} crosses {previous_split} and {split_name}"
                )

            instruction_key = normalize_text(example.instruction)
            if instruction_key:
                previous_split = owner_by_instruction.setdefault(
                    instruction_key,
                    split_name,
                )
                if previous_split != split_name:
                    raise DatasetLeakageError(
                        f"normalized instruction crosses {previous_split} and {split_name}"
                    )


def split_manifest(splits: DatasetSplits) -> dict[str, object]:
    """Return JSON-serializable split provenance for experiment manifests."""
    assert_no_leakage(splits)
    return {
        "seed": splits.seed,
        "counts": splits.counts(),
        "example_ids": splits.ids(),
        "component_by_example_id": dict(sorted(splits.component_by_example_id.items())),
        "leakage_checks": {
            "unique_example_ids": True,
            "group_ids_disjoint": True,
            "normalized_instructions_disjoint": True,
        },
    }