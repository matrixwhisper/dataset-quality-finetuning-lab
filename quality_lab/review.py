"""SQLite review ledger with append-only decisions and agreement reporting."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from quality_lab.dataset import (
    DefectLabel,
    ReviewDecision,
    TrainingExample,
    write_jsonl,
)


class ReviewStore:
    """Stores source examples separately from immutable reviewer events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS examples (
                    example_id TEXT PRIMARY KEY,
                    original_json TEXT NOT NULL,
                    current_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    example_id TEXT NOT NULL REFERENCES examples(example_id),
                    reviewer_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    defect_labels_json TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS review_events_example_idx
                    ON review_events(example_id, event_id);
                CREATE INDEX IF NOT EXISTS review_events_reviewer_idx
                    ON review_events(reviewer_id, example_id);
            """)

    def import_examples(self, examples: list[TrainingExample]) -> int:
        inserted = 0
        with self._connect() as connection:
            for example in examples:
                record_json = json.dumps(
                    example.to_record(),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                cursor = connection.execute(
                    """INSERT OR IGNORE INTO examples
                       (example_id, original_json, current_json) VALUES (?, ?, ?)""",
                    (example.example_id, record_json, record_json),
                )
                inserted += cursor.rowcount
        return inserted

    def record_review(
        self,
        example_id: str,
        reviewer_id: str,
        decision: ReviewDecision,
        defect_labels: tuple[DefectLabel, ...],
        note: str,
    ) -> None:
        if not reviewer_id.strip():
            raise ValueError("reviewer_id must not be empty")
        if decision is ReviewDecision.UNREVIEWED:
            raise ValueError("a review event must be accept, revise, or reject")
        if decision is ReviewDecision.ACCEPT and defect_labels:
            raise ValueError("accepted examples must not carry defect labels")
        if decision in {ReviewDecision.REVISE, ReviewDecision.REJECT} and not defect_labels:
            raise ValueError("revise/reject decisions require at least one defect label")

        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT current_json FROM examples WHERE example_id = ?",
                (example_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown example_id {example_id!r}")

            current = TrainingExample.from_record(json.loads(row["current_json"]))
            updated = replace(
                current,
                review_decision=decision,
                defect_labels=defect_labels,
                reviewer_id=reviewer_id,
                review_note=note,
            )
            connection.execute(
                """INSERT INTO review_events
                   (example_id, reviewer_id, decision, defect_labels_json, note, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    example_id,
                    reviewer_id,
                    decision.value,
                    json.dumps(sorted(label.value for label in defect_labels)),
                    note,
                    now,
                ),
            )
            connection.execute(
                "UPDATE examples SET current_json = ? WHERE example_id = ?",
                (
                    json.dumps(updated.to_record(), ensure_ascii=False, sort_keys=True),
                    example_id,
                ),
            )

    def export_current(self, destination: str | Path) -> Path:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT current_json FROM examples ORDER BY example_id"
            ).fetchall()
        examples = [
            TrainingExample.from_record(json.loads(row["current_json"]))
            for row in rows
        ]
        if not examples:
            raise ValueError("review database contains no examples")
        return write_jsonl(examples, destination)

    def reviewer_agreement(self) -> dict[str, object]:
        """Compute pairwise Cohen's kappa on examples reviewed by both people."""
        with self._connect() as connection:
            rows = connection.execute("""
                SELECT event_id, example_id, reviewer_id, decision, defect_labels_json
                FROM review_events
                ORDER BY event_id
            """).fetchall()

        latest: dict[tuple[str, str], tuple[str, str]] = {}
        for row in rows:
            labels = tuple(sorted(json.loads(row["defect_labels_json"])))
            adjudication = row["decision"] + ":" + ",".join(labels)
            latest[(row["reviewer_id"], row["example_id"])] = (
                adjudication,
                str(row["event_id"]),
            )

        reviewers = sorted({reviewer for reviewer, _ in latest})
        comparisons: list[dict[str, object]] = []
        for left_index, left in enumerate(reviewers):
            for right in reviewers[left_index + 1:]:
                common_ids = sorted(
                    example_id
                    for reviewer, example_id in latest
                    if reviewer == left
                    and (right, example_id) in latest
                )
                if not common_ids:
                    continue
                left_labels = [latest[(left, item)][0] for item in common_ids]
                right_labels = [latest[(right, item)][0] for item in common_ids]
                comparisons.append({
                    "reviewer_a": left,
                    "reviewer_b": right,
                    "overlap_count": len(common_ids),
                    "cohen_kappa": float(cohen_kappa_score(left_labels, right_labels)),
                })
        return {"pairwise_agreement": comparisons}

    def event_history(self, example_id: str) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT reviewer_id, decision, defect_labels_json, note, created_at
                   FROM review_events WHERE example_id = ? ORDER BY event_id""",
                (example_id,),
            ).fetchall()
        return [
            {
                "reviewer_id": row["reviewer_id"],
                "decision": row["decision"],
                "defect_labels": json.loads(row["defect_labels_json"]),
                "note": row["note"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]