"""Export the trained scikit-learn Random Forest into a Vercel/Node-compatible JSON artifact."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "sign_classifier.joblib")
    parser.add_argument("--metrics", type=Path, default=ROOT / "models" / "evaluation.json")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "sign_classifier.json")
    args = parser.parse_args()
    if not args.model.exists():
        raise SystemExit(f"Missing trained model: {args.model}. Run python ml/train.py first.")
    model = joblib.load(args.model)
    if not hasattr(model, "estimators_") or not hasattr(model, "classes_"):
        raise SystemExit("Expected a fitted scikit-learn RandomForestClassifier.")
    metrics = json.loads(args.metrics.read_text(encoding="utf-8")) if args.metrics.exists() else {}
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        trees.append({
            "children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "values": tree.value[:, 0, :].tolist(),
        })
    artifact = {
        "model_version": metrics.get("model_version", "random-forest-landmarks-v1"),
        "feature_schema": "wrist-relative-63-v1",
        "classes": [str(value) for value in model.classes_.tolist()],
        "trees": trees,
        "metrics": {
            key: metrics[key] for key in ("accuracy", "macro_f1", "samples_total", "class_count", "split_method")
            if key in metrics
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    print(f"Vercel-compatible model saved: {args.output} ({args.output.stat().st_size:,} bytes)")
if __name__ == "__main__":
    main()
