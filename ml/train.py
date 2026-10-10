"""Train an auditable Random Forest baseline from ASL Bridge JSONL samples.

Input rows must use the browser feature schema `wrist-relative-63-v1` and have
`label`, `features` (63 floats), and optionally `signer_id` fields.
This is a baseline classifier, not a complete ASL translation model.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
            label = row.get("label")
            features = row.get("features")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"Line {line_number} has no non-empty string label.")
            if not isinstance(features, list) or len(features) != 63:
                raise ValueError(f"Line {line_number} must contain exactly 63 feature values.")
            try:
                vector = np.asarray(features, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Line {line_number} has non-numeric features.") from exc
            if not np.isfinite(vector).all():
                raise ValueError(f"Line {line_number} contains non-finite features.")
            row["label"] = label.strip().upper()
            row["features"] = vector.tolist()
            row["signer_id"] = str(row.get("signer_id", "unspecified-signer"))
            row["feature_version"] = row.get("feature_version", "unknown")
            rows.append(row)
    if not rows:
        raise ValueError("Dataset has no samples.")
    return rows


def split_indices(rows: list[dict[str, Any]], labels: np.ndarray, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray, str]:
    groups = np.asarray([row["signer_id"] for row in rows])
    if len(set(groups.tolist())) >= 2:
        splitter = GroupShuffleSplit(n_splits=20, test_size=test_size, random_state=42)
        for train_index, test_index in splitter.split(np.zeros(len(labels)), labels, groups):
            if set(labels[train_index]) == set(labels) and len(set(labels[test_index])) >= 2:
                return train_index, test_index, "signer_holdout"
    counts = Counter(labels.tolist())
    if min(counts.values()) < 2:
        raise ValueError("Each label needs at least 2 samples; collect more before training.")
    train_index, test_index = train_test_split(
        np.arange(len(labels)), test_size=test_size, random_state=42, stratify=labels
    )
    return train_index, test_index, "stratified_random_fallback_not_signer_independent"


def train(input_path: Path, output_path: Path, metrics_path: Path) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    labels = np.asarray([row["label"] for row in rows])
    counts = Counter(labels.tolist())
    if len(counts) < 2:
        raise ValueError("At least 2 distinct labels are required to train a classifier.")
    sparse = {label: count for label, count in counts.items() if count < 2}
    if sparse:
        raise ValueError(f"Every label needs at least 2 samples. Too few: {sparse}")

    features = np.asarray([row["features"] for row in rows], dtype=np.float32)
    train_index, test_index, split_method = split_indices(rows, labels)
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features[train_index], labels[train_index])
    test_prediction = model.predict(features[test_index])
    report = classification_report(labels[test_index], test_prediction, output_dict=True, zero_division=0)
    matrix_labels = sorted(set(labels.tolist()))
    metrics: dict[str, Any] = {
        "model_version": "random-forest-landmarks-v1",
        "feature_schema": "wrist-relative-63-v1",
        "samples_total": len(rows),
        "samples_train": len(train_index),
        "samples_test": len(test_index),
        "class_count": len(counts),
        "class_counts": dict(sorted(counts.items())),
        "split_method": split_method,
        "accuracy": float(accuracy_score(labels[test_index], test_prediction)),
        "macro_f1": float(f1_score(labels[test_index], test_prediction, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix_labels": matrix_labels,
        "confusion_matrix": confusion_matrix(labels[test_index], test_prediction, labels=matrix_labels).tolist(),
        "limitations": [
            "Prototype baseline for one-hand static landmark vectors.",
            "Not a continuous or general-purpose ASL translator.",
            "J, Z, dynamic signs, facial grammar, and spatial/body cues are not modeled by this feature schema.",
            "Use signer_holdout metrics for generalization claims; random fallback metrics are optimistic.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "samples.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "sign_classifier.joblib")
    parser.add_argument("--metrics", type=Path, default=ROOT / "models" / "evaluation.json")
    args = parser.parse_args()
    try:
        metrics = train(args.input, args.output, args.metrics)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Training stopped: {exc}") from exc
    print(f"Model saved: {args.output}")
    print(f"Metrics saved: {args.metrics}")
    print(f"Samples: {metrics['samples_total']} | Classes: {metrics['class_count']}")
    print(f"Split: {metrics['split_method']}")
    print(f"Accuracy: {metrics['accuracy']:.3f} | Macro-F1: {metrics['macro_f1']:.3f}")
    if metrics["split_method"] != "signer_holdout":
        print("WARNING: test split is not signer-independent; do not use it to claim signer generalization.")


if __name__ == "__main__":
    main()
