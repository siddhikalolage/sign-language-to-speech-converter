Sign Language to Speech Converter

A real-time sign-language phrase recognition system that converts hand/face landmark observations into human-readable phrases and optional speech output.

The project contains two inference paths:

Landmark-based recognition — primary production path

YOLO image classification — secondary/backup path

Production-model policy: This repository is being improved without retraining or replacing the existing trained models. The serialized production artifacts are treated as authoritative.

1. Project Overview

The system captures a user's sign-language gesture through a camera or uploaded image, extracts visual/landmark information, predicts a phrase, and can convert the recognized phrase into speech.

Primary inference architecture

Camera / Image
      │
      ▼
MediaPipe Holistic Landmarks
      │
      ▼
144 raw landmark values
      │
      ▼
GestureFeatureExtractor
      │
      ▼
328 engineered geometric features
      │
      ▼
ExtraTreesClassifier
      │
      ▼
43 gesture classes
      │
      ▼
Label Encoder
      │
      ▼
Human-readable phrase
      │
      ▼
Sentence Builder / Speech Output

The landmark pipeline deliberately uses geometric landmark features rather than depending directly on image background pixels. This makes the primary model more suitable for variation in background and camera framing.

2. Key Capabilities

Real-time webcam gesture recognition

Phrase-level sign recognition

43 supported phrase classes in the production landmark model

MediaPipe-based landmark extraction

Hand-centered geometric feature engineering

ExtraTrees classification

YOLO-based backup image classification

Automatic model selection/fallback in the web application

Sentence construction from recognized signs

Speech output

Webcam and image-upload web interface

Artifact integrity validation

Explicit model/feature contracts

Reproducibility metadata and SHA-256 fingerprints

3. Production ML Pipeline

The current serialized landmark model is a scikit-learn Pipeline:

Pipeline
├── GestureFeatureExtractor
└── ExtraTreesClassifier

Input contract

Property

Value

Raw input features

144

Landmark representation

48 × 3 coordinates

Engineered features

328

Classifier

ExtraTreesClassifier

Trees

250

Random state

42

Classes

43

scikit-learn validated version

1.6.1

The 144-value input is interpreted by the feature extractor as:

Face landmarks      6 × 3 = 18
Left-hand landmarks 21 × 3 = 63
Right-hand landmarks 21 × 3 = 63
--------------------------------
Total                    = 144

The feature extractor converts these observations into 328 engineered features.

4. Feature Engineering

gesture_features.py contains the main feature-engineering layer.

The extractor creates several types of geometric information:

hand presence indicators

hand centroids

wrist positions

reference-normalized landmark coordinates

wrist-relative landmark coordinates

hand extent/scale

fingertip-to-wrist distances

joint-angle features

two-hand relative distances

two-hand wrist displacement

two-hand centroid displacement

This is important because the classifier does not simply consume raw coordinates.

The feature transformation provides information about:

hand shape

finger configuration

relative finger geometry

hand position

scale-normalized structure

relationships between both hands

Feature contract

Raw landmarks
     │
     ├── 144 values
     │
     ▼
GestureFeatureExtractor
     │
     └── 328 engineered features
             │
             ▼
      ExtraTreesClassifier

5. Production Model Artifacts

The project uses the following runtime artifacts.

Artifact

Purpose

text_phrase_image_model.pkl

Primary landmark ML pipeline

text_phrase_image_label_encoder.pkl

Converts class indices into phrase labels

text_phrase_yolo_cls.pt

YOLO image-classification backup model

holistic_landmarker.task

MediaPipe landmark model

The production landmark model is intentionally not stored in Git history because of its size and repository-artifact policy.

Its local integrity is documented in:

models/model-manifest.json

The manifest records:

artifact name

artifact size

SHA-256 checksum

framework/version

pipeline structure

feature dimensions

classifier configuration

class count

reproducibility information

6. Artifact Validation

Before considering the project ready for demonstration or deployment, validate the production artifacts:

python scripts/validate_artifacts.py

A successful validation should confirm:

Artifact integrity
        ↓
Production pipeline loading
        ↓
144-feature input contract
        ↓
GestureFeatureExtractor
        ↓
328 engineered features
        ↓
ExtraTreesClassifier
        ↓
43 output classes
        ↓
Label encoder compatibility
        ↓
Successful prediction decoding

The validation script is intentionally designed as a contract test, not as a training script.

It does not retrain the model.

7. Supported Landmark Classes

The current production label encoder contains 43 classes:

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

8. Repository Structure

sign-language-to-speech-converter/
│
├── models/
│   ├── model-manifest.json
│   └── README.md
│
├── scripts/
│   └── validate_artifacts.py
│
├── gesture_features.py
├── gesture_pipeline.py
├── gesture_dataset.py
├── torch_gesture_model.py
│
├── predict_single_gesture.py
├── predict_phrase_yolo.py
├── web_app.py
│
├── create_text_gesture_data_from_images.py
├── train_text_gesture_model.py
├── test_text_gesture_model.py
│
├── prepare_yolo_classification_dataset.py
├── train_yolo_phrase_model.py
├── custom_train.py
│
├── speech_output.py
├── test_text_gesture_model.py
│
├── holistic_landmarker.task
├── text_phrase_image_label_encoder.pkl
├── text_phrase_yolo_cls.pt
├── text_phrase_yolo_cls.metrics.json
│
├── images for phrases/
├── text_phrase_image_data/
├── yolo_phrase_dataset/
│
├── web/
├── README.md
└── .gitignore

Dataset directories and large model artifacts may intentionally remain local rather than being committed to Git.

9. Important Source Files

gesture_features.py

Core feature-engineering module.

Responsibilities:

validate the 144-feature input contract

separate face and hand landmarks

calculate a hand-centered reference frame

normalize landmarks

calculate hand-relative features

calculate fingertip distances

calculate joint-angle features

calculate two-hand relationships

construct the final 328-feature representation

construct the scikit-learn production pipeline

This is one of the most important ML files in the repository.

gesture_pipeline.py

Responsible for the landmark-extraction and quality-processing layer.

It defines the expected landmark representation and supporting processing logic used before classification.

gesture_dataset.py

Responsible for dataset loading, organization, and dataset splitting used by the training/evaluation workflow.

predict_single_gesture.py

Primary live landmark inference application.

Typical responsibilities:

camera initialization

landmark extraction

preprocessing

production model loading

prediction

confidence handling

phrase construction

user controls

optional speech output

Recommended command:

python predict_single_gesture.py --classifier-path text_phrase_image_model.pkl --label-encoder-path text_phrase_image_label_encoder.pkl

With speech:

python predict_single_gesture.py --classifier-path text_phrase_image_model.pkl --label-encoder-path text_phrase_image_label_encoder.pkl --speak

predict_phrase_yolo.py

Secondary image-based inference path using the YOLO classifier.

Example:

python predict_phrase_yolo.py --model-path text_phrase_yolo_cls.pt --class-map yolo_phrase_dataset/class_name_map.json

The YOLO path is useful as a backup for gestures where reliable landmarks cannot be obtained.

web_app.py

Browser-based application layer.

The web application provides:

webcam prediction

image upload prediction

model selection

landmark inference

YOLO inference

automatic fallback

sentence construction

speech output

result-focused UI

Run:

python web_app.py

Then open:

http://127.0.0.1:8000

10. Automatic Inference Strategy

The web application supports three model modes:

AUTO
LANDMARK
YOLO

Auto mode

The intended strategy is:

Input
  │
  ▼
Landmark extraction
  │
  ├── Reliable landmarks
  │       │
  │       ▼
  │   Landmark model
  │
  └── Unreliable landmarks
          │
          ▼
     YOLO fallback

This allows the system to retain the stronger landmark-based pipeline while still having an image-classification fallback.

11. Sentence Construction

The application does not stop at individual classification.

Accepted predictions can be accumulated into a sentence.

Typical controls include:

Key

Action

B

Undo last word

S

Speak sentence

C

Clear sentence

Q

Quit

The web interface provides equivalent controls.

The application also uses stability/acceptance logic so that noisy frame-to-frame predictions do not automatically become repeated words.

12. Speech Layer

speech_output.py contains the speech-output functionality.

The system can:

Recognized phrase
      ↓
Sentence text
      ↓
Speech engine
      ↓
Audio output

The web application can queue speech so recognition does not unnecessarily block on audio playback.

13. YOLO Backup Architecture

The project also contains an image-based classification path:

Image
  ↓
YOLO classifier
  ↓
Phrase class

Related files:

prepare_yolo_classification_dataset.py
train_yolo_phrase_model.py
predict_phrase_yolo.py
text_phrase_yolo_cls.pt
text_phrase_yolo_cls.metrics.json

This path should be treated as a secondary approach rather than replacing the current production landmark model.

14. Training Code Policy

The repository contains training scripts for reproducibility and future experimentation:

train_text_gesture_model.py
train_yolo_phrase_model.py
custom_train.py

However:

The current project improvement workflow does not require retraining the production model.

The existing serialized production model remains authoritative.

Training scripts should therefore not be executed merely to improve repository quality.

15. Dataset Generation

Landmark dataset generation is handled by:

create_text_gesture_data_from_images.py

The process is approximately:

Phrase image folders
        ↓
MediaPipe landmark extraction
        ↓
Landmark validation
        ↓
CSV dataset
        ↓
Training/evaluation workflow

The YOLO dataset preparation process is handled separately by:

prepare_yolo_classification_dataset.py

16. Testing and Validation

The repository contains multiple layers of verification.

Artifact validation

python scripts/validate_artifacts.py

Existing landmark-model test

python test_text_gesture_model.py

Basic application validation

python -m py_compile gesture_features.py gesture_pipeline.py predict_single_gesture.py predict_phrase_yolo.py web_app.py

The goal is to distinguish:

Syntax correctness
        +
Artifact correctness
        +
Pipeline contract correctness
        +
Runtime correctness
        +
Application behavior

rather than relying on one test alone.

17. Reproducibility

The production model uses:

random_state = 42
n_estimators = 250
n_jobs = 1

The current production artifact is authoritative.

The training factory currently exposes a different default estimator count. This difference is documented deliberately and should not be interpreted as a request to retrain the production model.

The artifact manifest is the source of truth for the serialized production model.

18. Security and Repository Hygiene

The repository intentionally excludes:

virtual environments

Python cache files

local environment files

datasets

experimental model backups

large serialized experimental artifacts

tool caches

Configured in:

.gitignore

Do not commit:

.venv/
.env
datasets
temporary files
large experimental model backups

Never commit API keys, passwords, tokens, or private credentials.

19. Current Production Artifact Fingerprints

The local production artifacts are identified by SHA-256 fingerprints in models/model-manifest.json.

This provides an integrity mechanism:

Artifact
   ↓
SHA-256
   ↓
Manifest
   ↓
Validation

If an artifact changes unexpectedly, validation should fail instead of silently treating the changed file as the same production model.

20. Quick Start

Step 1 — Activate environment

.\.venv\Scripts\Activate.ps1

Step 2 — Validate artifacts

python scripts/validate_artifacts.py

Step 3 — Run the primary predictor

python predict_single_gesture.py --classifier-path text_phrase_image_model.pkl --label-encoder-path text_phrase_image_label_encoder.pkl

Step 4 — Run the web application

python web_app.py

Open:

http://127.0.0.1:8000

21. Recommended Development Workflow

For changes to the repository, use this order:

1. Inspect existing implementation
        ↓
2. Define the contract
        ↓
3. Make the smallest useful change
        ↓
4. Run syntax checks
        ↓
5. Run artifact validation
        ↓
6. Run targeted functional tests
        ↓
7. Review Git diff
        ↓
8. Commit with a focused message

Avoid unnecessary rewrites.

Avoid changing the production ML model unless explicitly required.

22. Engineering Principles

This project follows several principles intended to make the repository easier to review and maintain:

Separation of concerns

Landmark extraction
        ≠
Feature engineering
        ≠
Classification
        ≠
Application UI
        ≠
Speech output

Explicit contracts

Important interfaces such as:

144 raw features

328 engineered features

43 classes

artifact checksums

are explicitly validated.

Reproducibility

Production artifact metadata is stored separately from experimental training code.

Fail-fast validation

Incorrect artifacts or incompatible model contracts should be detected before runtime inference.

No unnecessary retraining

Existing trained models are preserved when repository engineering improvements can be achieved without changing model weights.

23. Project Status

The project currently contains:

a production landmark-based ML pipeline

a 328-feature engineered representation

a 43-class ExtraTrees classifier

a YOLO backup classifier

MediaPipe landmark extraction

live inference

web inference

speech output

artifact integrity metadata

automated production-artifact validation

The current engineering focus is on improving:

Architecture
Documentation
Validation
Reproducibility
Maintainability
Testing
Deployment readiness
Repository quality

without retraining the production ML model.

24. Final Architecture Summary

                         SIGN LANGUAGE SYSTEM
                                  │
             ┌────────────────────┴────────────────────┐
             │                                         │
             ▼                                         ▼
       Landmark Path                              YOLO Path
             │                                         │
      MediaPipe Holistic                         Input Image
             │                                         │
       144 raw values                            YOLO Model
             │                                         │
   GestureFeatureExtractor                       Prediction
             │                                         │
      328 engineered                              Phrase
        features                                     │
             │                                         │
     ExtraTreesClassifier                            │
             │                                         │
        43 classes                                    │
             │                                         │
             └────────────────────┬────────────────────┘
                                  │
                                  ▼
                         Phrase / Sentence
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
              Web Interface               Speech Output

License / Usage

Add the appropriate license before distributing the repository publicly if the project is intended for external use.

For academic/project evaluation, document the dataset sources, model provenance, and any third-party model licenses used by the project.