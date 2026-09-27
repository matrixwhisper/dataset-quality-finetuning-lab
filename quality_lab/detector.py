"""Rules baseline, TF-IDF defect detector, and active-review ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from quality_lab.dataset import ReviewDecision, TrainingExample
from quality_lab.quality import audit_examples, normalize_text


@dataclass(frozen=True)
class DetectorPrediction:
    example_id: str
    probability_defective: float
    predicted_defective: bool


def _text(example: TrainingExample) -> str:
    return f"Instruction: {example.instruction}\nResponse: {example.response}"


def _target(example: TrainingExample) -> int:
    if example.review_decision is ReviewDecision.UNREVIEWED:
        raise ValueError(f"{example.example_id}: needs a human review label")
    return int(bool(example.defect_labels))


class QualityRiskDetector:
    """A transparent conventional-ML baseline trained on human review labels."""

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        self.threshold = threshold
        self.pipeline = None

    def fit(self, examples: Iterable[TrainingExample]) -> "QualityRiskDetector":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        records = list(examples)
        if len(records) < 8:
            raise ValueError("need at least 8 human-reviewed examples to fit the detector")
        targets = [_target(example) for example in records]
        if targets.count(0) < 2 or targets.count(1) < 2:
            raise ValueError("training data needs at least 2 clean and 2 defective examples")

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_features=50_000,
                sublinear_tf=True,
                strip_accents="unicode",
            )),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=2026,
            )),
        ])
        self.pipeline.fit([_text(example) for example in records], targets)
        return self

    def predict(self, examples: Iterable[TrainingExample]) -> list[DetectorPrediction]:
        if self.pipeline is None:
            raise RuntimeError("fit the detector before prediction")
        records = list(examples)
        if not records:
            return []
        probabilities = self.pipeline.predict_proba(
            [_text(example) for example in records]
        )[:, 1]
        return [
            DetectorPrediction(
                example_id=example.example_id,
                probability_defective=float(probability),
                predicted_defective=bool(probability >= self.threshold),
            )
            for example, probability in zip(records, probabilities)
        ]

    def save(self, path: str | Path) -> Path:
        if self.pipeline is None:
            raise RuntimeError("cannot save an unfitted detector")
        import joblib

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "threshold": self.threshold}, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "QualityRiskDetector":
        import joblib

        saved = joblib.load(path)
        detector = cls(threshold=saved["threshold"])
        detector.pipeline = saved["pipeline"]
        return detector


def evaluate_detector(
    detector: QualityRiskDetector,
    examples: Iterable[TrainingExample],
) -> dict[str, object]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    records = list(examples)
    if not records:
        raise ValueError("cannot evaluate on an empty split")
    expected = [_target(example) for example in records]
    predictions = detector.predict(records)
    predicted = [int(item.predicted_defective) for item in predictions]
    probabilities = [item.probability_defective for item in predictions]

    result: dict[str, object] = {
        "task_count": len(records),
        "accuracy": float(accuracy_score(expected, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
        "precision": float(precision_score(expected, predicted, zero_division=0)),
        "recall": float(recall_score(expected, predicted, zero_division=0)),
        "f1": float(f1_score(expected, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(expected, probabilities)),
        "confusion_matrix_labels_clean_defective": confusion_matrix(
            expected, predicted, labels=[0, 1]
        ).tolist(),
        "predictions": [
            asdict(item) for item in predictions
        ],
    }
    result["roc_auc"] = (
        float(roc_auc_score(expected, probabilities))
        if len(set(expected)) == 2
        else None
    )
    return result


def rules_baseline(examples: Iterable[TrainingExample]) -> dict[str, object]:
    """Score obvious issues as a baseline; this is not a human quality label."""
    records = list(examples)
    report = audit_examples(records)
    flagged_codes = {
        "possible_ambiguity",
        "placeholder_text",
        "conflicting_responses",
        "exact_duplicate",
        "missing_license",
    }
    flagged_ids = {
        issue.example_id
        for issue in report.issues
        if issue.code in flagged_codes
    }
    return {
        "method": "deterministic_rules",
        "flagged_count": len(flagged_ids),
        "flagged_example_ids": sorted(flagged_ids),
        "warning": "Rule flags are triage signals, not ground-truth defect labels.",
    }


def active_review_queue(
    detector: QualityRiskDetector,
    candidates: Iterable[TrainingExample],
    batch_size: int = 20,
) -> list[dict[str, object]]:
    """Rank unreviewed rows by uncertainty with simple prompt/group diversity."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    records = [
        example for example in candidates
        if example.review_decision is ReviewDecision.UNREVIEWED
    ]
    if not records:
        return []

    predictions = detector.predict(records)
    ranked = sorted(
        zip(records, predictions),
        key=lambda pair: abs(pair[1].probability_defective - 0.5),
    )

    chosen: list[dict[str, object]] = []
    used_groups: set[str] = set()
    used_prompts: set[str] = set()
    for example, prediction in ranked:
        normalized_prompt = normalize_text(example.instruction)
        if example.group_id in used_groups or normalized_prompt in used_prompts:
            continue
        chosen.append({
            "example_id": example.example_id,
            "group_id": example.group_id,
            "probability_defective": prediction.probability_defective,
            "uncertainty": 1.0 - 2.0 * abs(prediction.probability_defective - 0.5),
            "instruction": example.instruction,
            "response": example.response,
        })
        used_groups.add(example.group_id)
        used_prompts.add(normalized_prompt)
        if len(chosen) == batch_size:
            break
    return chosen