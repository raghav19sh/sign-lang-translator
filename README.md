# ASL Bridge

**ASL Bridge** is an accessible, privacy-conscious web workbench for American Sign Language (ASL) research and learning. This repository contains a complete runnable prototype with webcam hand-landmark tracking, local landmark sample collection, a reproducible Random Forest baseline training pipeline, a FastAPI inference service, and an English-to-gloss practice interface.

> **Scope and honesty note:** this is not a fully trained, general-purpose ASL translator. MediaPipe detects hands and 21 landmarks per hand; those landmarks are not meaning by themselves. The model must be trained on licensed, properly labeled data. Text-to-sign currently generates a starter gloss-practice queue and does not include reviewed ASL video or an animated signer. Do not use it as a substitute for a qualified interpreter or emergency communication.

## Features

- Next.js App Router, TypeScript, React, responsive custom CSS.
- Browser camera access starts only after the user clicks **Start camera**.
- MediaPipe hand landmark overlay, using browser video frames.
- Explicit opt-in to the local inference API; only normalized 63-number landmark vectors are sent, never raw video frames.
- Local sample collector with label and signer/session ID; samples stay in browser local storage until exported.
- JSONL export compatible with the Python training script.
- Random Forest baseline training, evaluation metrics, class report, confusion matrix, and signer-group split where sample metadata supports it.
- FastAPI endpoints for health, model info, and predictions.
- Starter phrase/gloss practice library with explicit warnings about the lack of reviewed sign demonstrations.
- Browser text-to-speech and session transcript controls.

## Requirements

- Node.js **20.9 or newer** and npm.
- Python **3.11 or newer** for the training and API components.
- A modern browser with camera access. Camera APIs generally require localhost or HTTPS.

## Run the web application

```bash
npm install
npm run dev
```

Open <http://localhost:3000>. Choose **Start camera** and permit access. The browser downloads MediaPipe's WebAssembly files and hand-landmarker model from their published Google/jsDelivr URLs at runtime.

Production checks:

```bash
npm run typecheck
npm run lint
npm run build
```

## Run the inference API

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r services/api/requirements.txt
uvicorn services.api.main:app --reload --host 127.0.0.1 --port 8000
```

Check <http://127.0.0.1:8000/api/v1/health>. The API reports that the classifier is unavailable until a model has been trained.

By default, the frontend shows its API connection toggle as disabled. Turning it on sends only normalized landmark vectors to the same-origin Next.js endpoint (`/api/v1/health` and `/api/v1/recognize`), which is suitable for a deployment such as Vercel. To use the separate FastAPI service instead, set `NEXT_PUBLIC_ASL_API_URL` in `.env.local` to its base URL (for example, `http://127.0.0.1:8000` for local development). Only use a trusted HTTPS URL for a remote service, and update your privacy notice to disclose that processing.

## Collect and train a baseline model

1. Start the web app and go to **Translator → Sign to Text**.
2. Start the camera, type a label such as `HELLO` or `A`, and set a meaningful signer/session ID.
3. Capture multiple examples per class under different conditions. Use distinct session IDs for independent sessions or people. Do not include identifiable recordings; the collector only stores feature vectors.
4. Export the data as JSONL.
5. Copy the export to `data/samples.jsonl`.
6. Install ML dependencies and train:

```bash
python -m pip install -r ml/requirements.txt
python ml/train.py
python -m unittest ml.tests.test_training
```

The model artifact is written to `models/sign_classifier.joblib`; metrics are written to `models/evaluation.json`. Restart the API after training so it loads the new artifact, then turn on **Connect local model API** in the web app.

Every row must contain 63 values using `wrist-relative-63-v1`: 21 hand landmarks × normalized x/y/z coordinates. Samples with a different schema are not compatible.

### Dataset quality requirements

- Capture enough examples per class; the trainer requires at least two examples per label as a basic guard, but useful accuracy generally requires substantially more and more diverse data.
- A signer-independent holdout is attempted when there are enough signer/session groups. If it falls back to a random stratified split, the metrics are not evidence of generalization to new signers.
- Avoid duplicate or near-identical frames across training and testing.
- This model represents a single hand's static pose only. It does not model temporal movement, facial expression, body posture, or ASL spatial grammar. Dynamic letters/signs (including J and Z) need temporal features or video models.
- Confidence is a model score, not a calibrated probability unless calibration is separately measured.

## API

- `GET /api/v1/health` — API and model readiness.
- `GET /api/v1/model-info` — feature schema and saved metrics metadata.
- `POST /api/v1/recognize` — accepts `{"features": [63 finite numeric values]}`; returns `model_unavailable` if the model is missing.

The API validates the feature vector length and finite values. CORS is restricted to localhost origins by default. Configure `ASL_ALLOWED_ORIGINS` explicitly for a trusted deployment; do not use `*` with credentialed flows.

## Text-to-sign roadmap

The included library is a **gloss-practice starter**, not a full linguistic translation system. Before publishing an ASL learning product:

1. Have Deaf ASL users or qualified reviewers validate phrases, glosses, context notes, and grammar.
2. Add demonstrations that your team has permission to use; record asset provenance and license.
3. Use proper video segmentation/playback or a validated signing avatar rather than letter-by-letter English substitution.
4. For continuous signing, study licensed datasets such as ASL Citizen (isolated signs), WLASL (word-level videos), and How2Sign (continuous signing) and follow each dataset's terms.
5. Evaluate on held-out signers and report per-class metrics, word error rate for sequences, latency, and unsupported-sign rejection.

## Privacy and security

- Raw video is not saved or uploaded by the provided web UI.
- Captured landmark features are stored in local storage only after the user clicks **Capture sample**. Use **Export JSONL** to download them.
- Clear local samples from the Model lab when no longer needed. Browser storage is not encrypted storage; protect access to the local account/device.
- Enabling the inference API sends the currently detected landmark features to the configured API. Keep it local or configure a trusted HTTPS endpoint and disclose that processing.
- Do not put credentials, private datasets, or model artifacts in the public repository.
- Validate with real ASL signers and accessibility users. Never claim full translation or production-grade accuracy without evidence.

## Repository layout

```text
src/app/                  Next.js route and global styling
src/components/            Translator, camera, dataset collector, library and model lab
src/lib/sign-library.ts    Starter gloss-practice entries
services/api/main.py       FastAPI inference service
ml/train.py                Dataset validation and Random Forest training
ml/tests/                  Dataset schema tests
data/                      Local training exports (ignored by Git)
models/                    Local model artifacts (ignored by Git)
```

## License

See `LICENSE` in the originating repository for the repository license. Dataset assets and any later-added sign demonstration media may have separate terms; check each asset individually.
