import json
import tempfile
import unittest
from pathlib import Path

from ml.train import load_jsonl


class DatasetValidationTests(unittest.TestCase):
    def test_loads_valid_landmark_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            file_path = Path(temp) / "samples.jsonl"
            row = {
                "label": "hello",
                "features": [0.0] * 63,
                "signer_id": "signer-01",
                "feature_version": "wrist-relative-63-v1",
            }
            file_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            rows = load_jsonl(file_path)
            self.assertEqual(rows[0]["label"], "HELLO")
            self.assertEqual(len(rows[0]["features"]), 63)

    def test_rejects_wrong_feature_length(self):
        with tempfile.TemporaryDirectory() as temp:
            file_path = Path(temp) / "samples.jsonl"
            file_path.write_text(json.dumps({"label": "A", "features": [0.0] * 62}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly 63"):
                load_jsonl(file_path)

    def test_rejects_non_finite_features(self):
        with tempfile.TemporaryDirectory() as temp:
            file_path = Path(temp) / "samples.jsonl"
            features = [0.0] * 63
            features[10] = float("nan")
            file_path.write_text(json.dumps({"label": "A", "features": features}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_jsonl(file_path)


if __name__ == "__main__":
    unittest.main()
