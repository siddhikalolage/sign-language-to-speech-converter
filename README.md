# Sign Language to Speech Converter

A production-oriented sign-language recognition system that converts visual sign input into human-readable phrases and optional speech output.

The repository is engineered around an existing serialized production model. Current hardening focuses on reliability, explicit contracts, validation, testing, reproducibility, API safety, CI, and maintainability rather than unnecessary model retraining.

## Engineering status

| Area | Status |
|---|---|
| Landmark inference pipeline | Production artifact validated |
| Feature contract | 144 raw → 328 engineered features |
| Production classifier | ExtraTrees, 250 trees, 43 classes |
| YOLO fallback | Available |
| Artifact integrity validation | Automated |
| API validation | Automated tests |
| CI quality gate | GitHub Actions |
| Production-model retraining | Intentionally not part of hardening |

Latest local verification in this hardening phase: **27 tests passed**. The application test suite reports two FastAPI deprecation warnings related to `@app.on_event`; these are warnings, not test failures.

## 1. System architecture

```text
                    Camera / Uploaded Image
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        Landmark inference             YOLO inference
                │                           │
        MediaPipe landmarks          Image classifier
                │                           │
           144 raw values                 Class
                │                           │
                ▼                           │
      GestureFeatureExtractor              │
                │                           │
        328 engineered features            │
                │                           │
                ▼                           │
       ExtraTreesClassifier                │
                │                           │
             43 classes                    │
                │                           │
                └─────────────┬─────────────┘
                              ▼
                       Phrase / Sentence
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              Web interface       Speech output
```

The landmark path is the primary production path. The web application can use YOLO as a secondary path and automatic fallback when the landmark result does not meet configured reliability thresholds.

## 2. Production ML contract

The serialized landmark model is a scikit-learn `Pipeline` containing:

```text
GestureFeatureExtractor
        ↓
ExtraTreesClassifier
```

| Property | Value |
|---|---:|
| Raw input features | 144 |
| Engineered features | 328 |
| Classifier | ExtraTreesClassifier |
| Trees | 250 |
| Random state | 42 |
| `n_jobs` | 1 |
| `min_samples_leaf` | 1 |
| Output classes | 43 |
| Validated scikit-learn version | 1.6.1 |

The 144 raw values represent 48 landmarks × 3 coordinates:

- Face: 6 × 3 = 18
- Left hand: 21 × 3 = 63
- Right hand: 21 × 3 = 63
- **Total: 144**

The feature extractor transforms these observations into 328 geometric features before classification.

## 3. Feature engineering

`gesture_features.py` contains the core feature-engineering layer and preserves the raw-landmark to classifier-input contract.

The engineered representation includes information such as:

- hand presence
- hand centroids and wrist positions
- normalized landmark coordinates
- wrist-relative coordinates
- hand extent and scale
- fingertip-to-wrist distances
- joint-angle features
- two-hand relative distances
- two-hand wrist and centroid displacement

The classifier therefore operates on geometric structure rather than directly depending on image background pixels.

## 4. Production artifacts

| Artifact | Purpose |
|---|---|
| `text_phrase_image_model.pkl` | Primary landmark classification pipeline |
| `text_phrase_image_label_encoder.pkl` | Class-index → phrase mapping |
| `text_phrase_yolo_cls.pt` | YOLO image-classification fallback |
| `holistic_landmarker.task` | MediaPipe landmark model |
| `yolo_phrase_dataset/class_name_map.json` | YOLO class-name mapping |

Artifact metadata is maintained in `models/model-manifest.json`. The manifest records artifact identity, SHA-256 fingerprints, framework/version information, feature dimensions, classifier configuration, and reproducibility information.

**Production policy:** the serialized production artifact is authoritative. The training factory may expose a different default estimator count; that metadata must not be interpreted as a request to retrain the production model.

## 5. Artifact validation

Run:

```powershell
python scripts/validate_artifacts.py
```

The validator checks artifact availability/integrity, serialized model loading, the 144-feature input contract, `GestureFeatureExtractor`, the 328-feature transformation contract, classifier dimensions, 43-class label compatibility, probability shape, and successful prediction decoding.

A successful run ends with `Status: PASS`. This is a contract-validation operation, not a training operation.

## 6. Application and API

`web_app.py` provides the FastAPI application and browser-facing prediction API.

| Endpoint | Purpose |
|---|---|
| `GET /` | Serves the web application |
| `GET /api/health` | Reports model/runtime availability |
| `POST /api/predict` | Performs image prediction |
| `/web/*` | Serves web assets |

Prediction modes:

- `landmark` — primary landmark path
- `yolo` — YOLO classifier
- `auto` — prefer a strong landmark result, then use YOLO fallback when appropriate

The API returns prediction metadata including confidence, quality, margin, hand-point counts, backend, orientation, latency, and top candidates.

### Input hardening

The prediction API validates image payloads before inference. It rejects missing/empty payloads, invalid Base64, empty decoded data, oversized Base64 payloads, oversized decoded image payloads, undecodable image bytes, and unsupported prediction modes.

The limits are defined centrally in `web_app.py` rather than duplicated across the application.

## 7. Web/API testing

`tests/test_web_app.py` covers:

- health response contract
- invalid Base64 handling
- invalid image bytes
- empty payload handling
- oversized payload handling
- unsupported prediction modes

The tests exercise the application contract directly without requiring a running network server for unit-level behavior.

Run the complete suite:

```powershell
python -m pytest -q
```

## 8. Continuous integration

`.github/workflows/ci.yml` provides the GitHub Actions quality gate.

```text
Checkout
   ↓
Python 3.10
   ↓
Install requirements-dev.txt
   ↓
Compile core Python modules
   ↓
Validate production artifacts
   ↓
Run pytest
   ↓
Check Git whitespace
```

This puts syntax validation, artifact compatibility, tests, and repository hygiene into one automated verification path.

## 9. Reproducible environment

`requirements.txt` records the validated runtime environment, including:

- FastAPI 0.141.1
- Uvicorn 0.52.3
- OpenCV 5.0.0.93
- NumPy 1.26.4
- joblib 1.5.3
- scikit-learn 1.6.1
- MediaPipe 0.10.21
- Ultralytics
- PyTorch

`requirements-dev.txt` layers development/test tooling over the runtime requirements.

A baseline environment snapshot is stored at `docs/environment/baseline-pip-freeze.txt`.

Do not casually change ML package versions: serialized model compatibility should be revalidated after dependency changes.

## 10. Repository structure

```text
sign-language-to-speech-converter/
├── .github/workflows/ci.yml
├── docs/environment/baseline-pip-freeze.txt
├── models/model-manifest.json
├── scripts/
│   ├── validate_artifacts.py
│   └── diagnostics/
│       ├── check_again_wrists.py
│       ├── check_hungry_wrists.py
│       ├── debug_hungry_detection.py
│       └── diagnose_hungry.py
├── tests/
│   ├── test_artifact_contract.py
│   ├── test_feature_contract.py
│   └── test_web_app.py
├── gesture_features.py
├── gesture_pipeline.py
├── gesture_dataset.py
├── torch_gesture_model.py
├── predict_single_gesture.py
├── predict_phrase_yolo.py
├── web_app.py
├── speech_output.py
├── create_text_gesture_data_from_images.py
├── train_text_gesture_model.py
├── train_yolo_phrase_model.py
├── prepare_yolo_classification_dataset.py
├── custom_train.py
├── holistic_landmarker.task
├── text_phrase_image_model.pkl
├── text_phrase_image_label_encoder.pkl
├── text_phrase_yolo_cls.pt
├── text_phrase_yolo_cls.metrics.json
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .gitignore
└── README.md
```

Large datasets and experimental model backups are intentionally excluded from the normal source-control workflow.

## 11. Main source modules

### `gesture_features.py`

Core feature engineering and production pipeline construction. Preserves the 144 → 328 feature contract.

### `gesture_pipeline.py`

Landmark extraction, signal-quality processing, and supporting inference logic.

### `predict_single_gesture.py`

Live/command-line landmark inference, production model loading, prediction handling, phrase construction, and optional speech output.

```powershell
python predict_single_gesture.py --classifier-path text_phrase_image_model.pkl --label-encoder-path text_phrase_image_label_encoder.pkl
```

With speech:

```powershell
python predict_single_gesture.py --classifier-path text_phrase_image_model.pkl --label-encoder-path text_phrase_image_label_encoder.pkl --speak
```

### `predict_phrase_yolo.py`

Secondary image-classification inference using the YOLO artifact.

### `web_app.py`

FastAPI application, runtime registry, prediction routing, image decoding, reliability thresholds, and web serving.

### `speech_output.py`

Speech synthesis/output functionality.

## 12. Automatic inference strategy

```text
Input image
    │
    ▼
Landmark inference
    │
    ├── Strong result ───────► Return landmark prediction
    │
    └── Weak/no usable result
                 │
                 ▼
          YOLO confidence check
                 │
                 ├── Strong enough ─► Return YOLO fallback
                 │
                 └── Otherwise ─────► Reject as unreliable
```

The goal is to prefer a reliable production landmark result rather than blindly returning the highest-scoring backend prediction.

## 13. Reliability controls

The web application uses explicit acceptance thresholds for:

- landmark confidence
- landmark probability margin
- landmark visibility/quality
- minimum detected hand points
- YOLO confidence
- stricter YOLO confidence for automatic fallback

These are application-level acceptance controls and do not modify trained model weights.

## 14. Speech and sentence handling

Recognized signs can be accumulated into a sentence and passed to the speech layer. Recognition, sentence handling, and audio output remain separate concerns.

## 15. Supported production classes

The production label encoder contains 43 classes:

```text
again
agree
answer
attendance
book
break
careful
change
chat
congratulations
email
file
good_morning
happy_birthday
home
how_are_you
i_need_help
join
keepsmile
meet
mistake
open
opinion
pass
please
practice
pressure
problem
questions
remember
seat
shift
sick
stop
sun
team
thirsty
this
together
understand
wait
where
write
```

## 16. Training and production-model policy

Training scripts are retained for reproducibility and future experimentation:

- `train_text_gesture_model.py`
- `train_yolo_phrase_model.py`
- `custom_train.py`

Repository hardening does **not** require retraining the production landmark model. The existing serialized artifact remains authoritative.

## 17. Dataset generation

Landmark dataset generation is handled by `create_text_gesture_data_from_images.py`:

```text
Phrase images
    ↓
MediaPipe landmark extraction
    ↓
Landmark validation
    ↓
CSV representation
    ↓
Training/evaluation workflow
```

YOLO dataset preparation is handled by `prepare_yolo_classification_dataset.py`.

Dataset folders are not intended to be committed wholesale because of repository size and artifact-management constraints.

## 18. Diagnostics

Targeted diagnostic scripts live under `scripts/diagnostics/`.

They are investigation tools, not production inference paths. The hungry/again diagnostics were introduced to investigate landmark-detection quality without altering production model weights.

Keeping diagnostics separate prevents experimental investigation from becoming an accidental runtime dependency.

## 19. Security and repository hygiene

`.gitignore` excludes common unsafe or unnecessary development artifacts, including:

- virtual environments
- Python cache files
- local `.env` files
- logs and temporary files
- tool caches
- datasets
- experimental/backup model artifacts

Never commit API keys, passwords, access tokens, private credentials, or local `.env` files.

The API also applies bounded image-payload validation before expensive ML inference.

## 20. Reproducibility and artifact provenance

`models/model-manifest.json` records artifact size, SHA-256 fingerprint, framework/version information, pipeline structure, feature dimensions, classifier configuration, class count, random state, and production estimator count.

Production classifier configuration:

```text
random_state = 42
n_estimators = 250
n_jobs = 1
min_samples_leaf = 1
```

The training factory default estimator count is separately recorded as 350. This difference is metadata and should not trigger production retraining.

## 21. Development workflow

Use a small, evidence-driven change cycle:

```text
Inspect implementation
        ↓
Confirm contract
        ↓
Make smallest useful change
        ↓
Run syntax checks
        ↓
Validate production artifacts
        ↓
Run targeted tests
        ↓
Run full pytest suite
        ↓
Review diff / whitespace
        ↓
Commit focused change
        ↓
Push and verify CI
```

Avoid broad rewrites when a focused engineering improvement is sufficient.

## 22. Validation commands

Compile core modules:

```powershell
python -m py_compile gesture_features.py gesture_pipeline.py predict_single_gesture.py predict_phrase_yolo.py web_app.py speech_output.py
```

Validate production artifacts:

```powershell
python scripts/validate_artifacts.py
```

Run tests:

```powershell
python -m pytest -q
```

Check whitespace:

```powershell
git diff --check
```

Check repository state:

```powershell
git status -sb
```

## 23. Quick start

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install development/runtime dependencies:

```powershell
python -m pip install -r requirements-dev.txt
```

Validate artifacts:

```powershell
python scripts/validate_artifacts.py
```

Run tests:

```powershell
python -m pytest -q
```

Start the web application:

```powershell
python web_app.py
```

Open:

```text
http://127.0.0.1:8000
```

## 24. Engineering principles

### Explicit contracts

Critical interfaces are validated instead of assumed:

```text
144 raw features
      ↓
328 engineered features
      ↓
43 classifier outputs
```

### Separation of concerns

```text
Landmark extraction
        ≠
Feature engineering
        ≠
Classification
        ≠
API routing
        ≠
Sentence handling
        ≠
Speech output
```

### Fail-fast validation

Invalid artifacts, incompatible dimensions, malformed API input, and unreliable predictions should be detected explicitly rather than silently accepted.

### Minimal-change engineering

Every change should have a clear reliability, maintainability, reproducibility, security, or deployment benefit.

### Production artifact preservation

Do not replace or retrain the production model merely because repository engineering can be improved without changing model weights.

## 25. Current project direction

The repository is being hardened toward expert-level software/ML engineering quality while preserving the existing production model.

Priority areas are:

1. production artifact integrity
2. deterministic contracts
3. API/runtime reliability
4. automated tests
5. CI quality gates
6. reproducible environments
7. repository hygiene
8. operational documentation

The objective is not complexity for its own sake. Each component should provide measurable engineering value.

## License and model/data provenance

Before public distribution, add an appropriate software license and document:

- dataset sources and permissions
- third-party model licenses
- MediaPipe/Ultralytics usage requirements
- model provenance
- restrictions on commercial use

Do not claim dataset or model ownership without verifying the original source and license.
