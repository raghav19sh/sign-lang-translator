# Dataset directory

The browser data collector exports `asl-bridge-samples.jsonl`. Move or copy it here as `samples.jsonl` when you are ready to train:

```bash
cp ~/Downloads/asl-bridge-samples.jsonl data/samples.jsonl
python ml/train.py
```

Sample records contain a label, 63 normalized hand landmark coordinates, a signer/session ID, capture timestamp, and feature-schema version. They do not include image frames. Avoid publishing personal or identifiable collection data.
