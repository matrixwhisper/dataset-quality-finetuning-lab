"""Command-line workflow for curation, detector training, QLoRA, and evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from quality_lab.benchmark import (
    GenerationSettings,
    QualityJudge,
    evaluate_judge,
    hill_climb_validation,
)
from quality_lab.config import LabConfig, seed_everything
from quality_lab.dataset import (
    DefectLabel,
    ReviewDecision,
    TrainingExample,
    load_jsonl,
    write_jsonl,
)
from quality_lab.detector import (
    QualityRiskDetector,
    active_review_queue,
    evaluate_detector,
    rules_baseline,
)
from quality_lab.finetuning import train_qlora
from quality_lab.quality import audit_examples, human_label_coverage
from quality_lab.review import ReviewStore
from quality_lab.splits import make_splits, split_manifest


def _sha256(examples: list[TrainingExample]) -> str:
    digest = hashlib.sha256()
    for example in sorted(examples, key=lambda row: row.example_id):
        digest.update(json.dumps(example.to_record(), sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _split_records(
    examples: list[TrainingExample],
    manifest_path: Path,
    seed: int,
) -> dict[str, list[TrainingExample]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_sha256") != _sha256(examples):
        raise ValueError("dataset changed after splitting; rebuild the split manifest")

    by_id = {item.example_id: item for item in examples}
    result: dict[str, list[TrainingExample]] = {}
    for split_name in ("train", "validation", "test"):
        ids = manifest["example_ids"][split_name]
        if len(ids) != len(set(ids)) or any(item not in by_id for item in ids):
            raise ValueError(f"{split_name} IDs are invalid or duplicated")
        result[split_name] = [by_id[item] for item in ids]

    rebuilt = make_splits(
        examples,
        seed=seed,
        train_fraction=manifest["train_fraction"],
        validation_fraction=manifest["validation_fraction"],
    )
    if split_manifest(rebuilt)["example_ids"] != manifest["example_ids"]:
        raise ValueError("split manifest does not match deterministic group-aware split")
    return result


def _reviewed(records: list[TrainingExample], split_name: str) -> tuple[TrainingExample, ...]:
    unreviewed = [item.example_id for item in records if item.review_decision is ReviewDecision.UNREVIEWED]
    if unreviewed:
        raise ValueError(
            f"{split_name} contains {len(unreviewed)} unreviewed examples; "
            "finish human review before supervised evaluation"
        )
    return tuple(records)


def _make_judge(config: LabConfig, adapter_path: str | None = None) -> QualityJudge:
    return QualityJudge(
        config.model_id,
        revision=config.model_revision,
        adapter_path=adapter_path,
    )


def _handle_audit(args: argparse.Namespace, config: LabConfig) -> None:
    examples = load_jsonl(args.input)
    report = audit_examples(examples)
    result = report.to_dict()
    result["human_label_coverage"] = human_label_coverage(examples)
    result["dataset_sha256"] = _sha256(examples)
    _write_json(args.output, result)
    print(f"Audited {len(examples)} records; report: {args.output}")


def _handle_split(args: argparse.Namespace, config: LabConfig) -> None:
    examples = load_jsonl(args.input)
    splits = make_splits(
        examples,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    manifest = split_manifest(splits)
    manifest.update({
        "dataset_sha256": _sha256(examples),
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
    })
    _write_json(args.output, manifest)
    print(json.dumps(manifest["counts"], sort_keys=True))


def _handle_detector_train(args: argparse.Namespace, config: LabConfig) -> None:
    examples = load_jsonl(args.input)
    splits = _split_records(examples, args.splits, args.seed)
    training = _reviewed(splits["train"], "train")
    detector = QualityRiskDetector().fit(training)
    model_path = Path(args.output)
    detector.save(model_path)
    validation = _reviewed(splits["validation"], "validation")
    metrics = evaluate_detector(detector, validation)
    metrics["split"] = "validation"
    metrics["dataset_sha256"] = _sha256(examples)
    _write_json(model_path.with_suffix(".validation.json"), metrics)
    print(f"Saved detector to {model_path}; validation F1={metrics['f1']:.3f}")


def _handle_detector_evaluate(args: argparse.Namespace, config: LabConfig) -> None:
    if args.split != "validation":
        raise ValueError("detector evaluation on test is reserved for the final command")
    examples = load_jsonl(args.input)
    splits = _split_records(examples, args.splits, args.seed)
    validation = _reviewed(splits["validation"], "validation")
    detector = QualityRiskDetector.load(args.model)
    metrics = evaluate_detector(detector, validation)
    metrics["rules_baseline"] = rules_baseline(validation)
    metrics["dataset_sha256"] = _sha256(examples)
    _write_json(args.output, metrics)
    print(f"Validation F1={metrics['f1']:.3f}; wrote {args.output}")


def _handle_review_queue(args: argparse.Namespace, config: LabConfig) -> None:
    examples = load_jsonl(args.input)
    detector = QualityRiskDetector.load(args.detector)
    queue = active_review_queue(detector, examples, batch_size=args.batch_size)
    _write_json(args.output, {
        "ranking_method": "uncertainty with group/prompt deduplication",
        "items": queue,
    })
    print(f"Queued {len(queue)} examples at {args.output}")


def _handle_review_db(args: argparse.Namespace, config: LabConfig) -> None:
    store = ReviewStore(args.database)
    if args.review_action == "import":
        count = store.import_examples(load_jsonl(args.input))
        print(f"Imported {count} new records")
    elif args.review_action == "record":
        labels = tuple(DefectLabel(value) for value in args.labels)
        store.record_review(
            args.example_id,
            args.reviewer_id,
            ReviewDecision(args.decision),
            labels,
            args.note,
        )
        print(f"Recorded review for {args.example_id}")
    elif args.review_action == "export":
        print(store.export_current(args.output))
    elif args.review_action == "agreement":
        print(json.dumps(store.reviewer_agreement(), indent=2))
    elif args.review_action == "history":
        print(json.dumps(store.event_history(args.example_id), indent=2))


def _handle_train_qlora(args: argparse.Namespace, config: LabConfig) -> None:
    examples = load_jsonl(args.input)
    splits = _split_records(examples, args.splits, args.seed)
    training = _reviewed(splits["train"], "train")
    config.model_revision = args.revision
    adapter = train_qlora(config, training, args.output)
    print(f"Saved QLoRA adapter to {adapter}")


def _handle_search(args: argparse.Namespace, config: LabConfig) -> None:
    examples = load_jsonl(args.input)
    splits = _split_records(examples, args.splits, args.seed)
    validation = _reviewed(splits["validation"], "validation")
    config.model_revision = args.revision
    factory = lambda: _make_judge(config, args.adapter)
    selected, _ = hill_climb_validation(
        validation,
        factory,
        GenerationSettings(),
        evaluation_budget=args.budget,
        output_path=args.output,
    )
    print(f"Selected validation settings: {selected}")


def _handle_final(args: argparse.Namespace, config: LabConfig) -> None:
    output_dir = Path(args.output_dir)
    lock_path = output_dir / "final_test_started.json"
    result_path = output_dir / "final_comparison.json"
    if lock_path.exists() or result_path.exists():
        raise FileExistsError("final test already started in this output directory")

    examples = load_jsonl(args.input)
    splits = _split_records(examples, args.splits, args.seed)
    search = json.loads(Path(args.search).read_text(encoding="utf-8"))
    split_manifest_data = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    if search.get("search_split") != "validation" or search.get("test_accessed") is not False:
        raise PermissionError("search artifact is not marked validation-only")
    if search.get("validation_example_ids") != split_manifest_data["example_ids"]["validation"]:
        raise ValueError("search validation IDs do not match current split")
    if search.get("dataset_sha256") not in (None, _sha256(examples)):
        raise ValueError("search dataset fingerprint is stale")

    adapter_manifest_path = Path(args.adapter) / "adapter_manifest.json"
    if not adapter_manifest_path.exists():
        raise FileNotFoundError(f"adapter manifest missing: {adapter_manifest_path}")
    adapter_manifest = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
    expected_train_ids = set(split_manifest_data["example_ids"]["train"])
    if set(adapter_manifest["training_example_ids"]) != expected_train_ids:
        raise ValueError("adapter was not trained on the current train split")
    if adapter_manifest.get("test_accessed") is not False:
        raise PermissionError("adapter manifest does not certify test isolation")

    test = _reviewed(splits["test"], "test")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(lock_path, {
        "status": "started",
        "dataset_sha256": _sha256(examples),
        "test_example_ids": [item.example_id for item in test],
    })

    config.model_revision = args.revision
    settings = GenerationSettings(**search["selected_settings"])
    seed_everything(args.seed)
    base = _make_judge(config)
    base_result = evaluate_judge(test, base, settings, "test")
    del base
    seed_everything(args.seed)
    tuned = _make_judge(config, args.adapter)
    tuned_result = evaluate_judge(test, tuned, settings, "test")

    detector_result = None
    if args.detector:
        detector = QualityRiskDetector.load(args.detector)
        detector_result = evaluate_detector(detector, test)

    comparison = {
        "status": "measured_final_test",
        "test_accessed": True,
        "dataset_sha256": _sha256(examples),
        "test_example_ids": [item.example_id for item in test],
        "settings_selected_on": "validation",
        "base": base_result,
        "qlora_adapter": tuned_result,
        "detector": detector_result,
        "f1_delta_adapter_minus_base": tuned_result["f1"] - base_result["f1"],
        "limitations": [
            "Results describe this reviewed dataset and split only.",
            "Human labels may contain disagreement or reviewer bias.",
            "A small pilot cannot establish broad generalization.",
        ],
    }
    _write_json(result_path, comparison)
    print(f"Final comparison written to {result_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quality-lab")
    parser.add_argument("--config", type=Path, default=Path("lab_config.json"))
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--output", type=Path, default=Path("runs/audit.json"))

    split = sub.add_parser("split")
    split.add_argument("--input", type=Path, required=True)
    split.add_argument("--output", type=Path, default=Path("runs/splits.json"))
    split.add_argument("--seed", type=int, default=2026)
    split.add_argument("--train-fraction", type=float, default=0.7)
    split.add_argument("--validation-fraction", type=float, default=0.15)

    detector_train = sub.add_parser("train-detector")
    detector_train.add_argument("--input", type=Path, required=True)
    detector_train.add_argument("--splits", type=Path, required=True)
    detector_train.add_argument("--output", type=Path, default=Path("runs/detector.joblib"))
    detector_train.add_argument("--seed", type=int, default=2026)

    detector_eval = sub.add_parser("evaluate-detector")
    detector_eval.add_argument("--input", type=Path, required=True)
    detector_eval.add_argument("--splits", type=Path, required=True)
    detector_eval.add_argument("--model", type=Path, required=True)
    detector_eval.add_argument("--output", type=Path, default=Path("runs/detector_validation.json"))
    detector_eval.add_argument("--split", choices=("validation", "test"), default="validation")
    detector_eval.add_argument("--seed", type=int, default=2026)

    queue = sub.add_parser("review-queue")
    queue.add_argument("--input", type=Path, required=True)
    queue.add_argument("--detector", type=Path, required=True)
    queue.add_argument("--output", type=Path, default=Path("runs/review_queue.json"))
    queue.add_argument("--batch-size", type=int, default=20)

    review = sub.add_parser("review")
    review.add_argument("review_action", choices=("import", "record", "export", "agreement", "history"))
    review.add_argument("--database", type=Path, default=Path("data/reviews.sqlite"))
    review.add_argument("--input", type=Path)
    review.add_argument("--output", type=Path, default=Path("data/reviewed.jsonl"))
    review.add_argument("--example-id")
    review.add_argument("--reviewer-id")
    review.add_argument("--decision", choices=("accept", "revise", "reject"))
    review.add_argument("--labels", nargs="*", default=[])
    review.add_argument("--note", default="")

    qlora = sub.add_parser("train-qlora")
    qlora.add_argument("--input", type=Path, required=True)
    qlora.add_argument("--splits", type=Path, required=True)
    qlora.add_argument("--output", type=Path, default=Path("runs/adapter"))
    qlora.add_argument("--revision", default="main")
    qlora.add_argument("--seed", type=int, default=2026)

    search = sub.add_parser("search")
    search.add_argument("--input", type=Path, required=True)
    search.add_argument("--splits", type=Path, required=True)
    search.add_argument("--adapter", type=Path, required=True)
    search.add_argument("--output", type=Path, default=Path("runs/search.json"))
    search.add_argument("--budget", type=int, default=5)
    search.add_argument("--revision", default="main")
    search.add_argument("--seed", type=int, default=2026)

    final = sub.add_parser("final")
    final.add_argument("--input", type=Path, required=True)
    final.add_argument("--splits", type=Path, required=True)
    final.add_argument("--search", type=Path, required=True)
    final.add_argument("--adapter", type=Path, required=True)
    final.add_argument("--detector", type=Path)
    final.add_argument("--output-dir", type=Path, default=Path("runs"))
    final.add_argument("--revision", default="main")
    final.add_argument("--seed", type=int, default=2026)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = LabConfig.load(args.config) if args.config.exists() else LabConfig()
    seed_everything(getattr(args, "seed", config.seed))
    handlers = {
        "audit": _handle_audit,
        "split": _handle_split,
        "train-detector": _handle_detector_train,
        "evaluate-detector": _handle_detector_evaluate,
        "review-queue": _handle_review_queue,
        "review": _handle_review_db,
        "train-qlora": _handle_train_qlora,
        "search": _handle_search,
        "final": _handle_final,
    }
    try:
        handlers[args.command](args, config)
    except (ValueError, RuntimeError, KeyError, PermissionError, FileNotFoundError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())