# Final Project

This folder is the final review-ready project package for the image-based phrase recognition system.

It contains the final two approaches we agreed on:

- Landmark-based prediction
  Main final approach
- YOLO-based prediction
  Backup and custom-dataset-friendly approach

## Final Main Files

- `predict_single_gesture.py`
  Final landmark live predictor
- `predict_phrase_yolo.py`
  Final YOLO live predictor
- `create_text_gesture_data_from_images.py`
  Creates landmark CSV dataset from phrase image folders
- `train_text_gesture_model.py`
  Trains the landmark classifier
- `test_text_gesture_model.py`
  Tests the landmark classifier
- `prepare_yolo_classification_dataset.py`
  Creates YOLO train/val/test image splits
- `train_yolo_phrase_model.py`
  Trains the YOLO image classifier
- `custom_train.py`
  One-step custom dataset trainer for landmark, YOLO, or both
- `web_app.py`
  Website backend for browser-based prediction

## Core Helper Files

- `gesture_pipeline.py`
  Landmark extraction and quality scoring
- `gesture_features.py`
  Landmark feature engineering
- `gesture_dataset.py`
  Dataset loading and splitting
- `torch_gesture_model.py`
  Helper module imported by the training script

## Included Final Models

- `text_phrase_image_model.pkl`
- `text_phrase_image_label_encoder.pkl`
- `text_phrase_yolo_cls.pt`
- `text_phrase_yolo_cls.metrics.json`
- `holistic_landmarker.task`

## Included Final Data Folders

- `images for phrases`
  Original phrase image dataset
- `text_phrase_image_data`
  Extracted landmark CSV dataset
- `yolo_phrase_dataset`
  YOLO-ready split dataset

## Recommended Demo Run

Run the main landmark predictor:

```powershell
python predict_single_gesture.py --classifier-path text_phrase_image_model.pkl --label-encoder-path text_phrase_image_label_encoder.pkl
```

Add speech output:

```powershell
python predict_single_gesture.py --classifier-path text_phrase_image_model.pkl --label-encoder-path text_phrase_image_label_encoder.pkl --speak
```

Run the YOLO backup predictor:

```powershell
python predict_phrase_yolo.py --model-path text_phrase_yolo_cls.pt --class-map yolo_phrase_dataset/class_name_map.json
```

The live predictors build a sentence from accepted words. Press `B` to undo the last word, `S` to speak the full sentence, `C` to clear, and `Q` to quit.

The landmark dataset and live/web prediction now use the same non-mirrored input orientation. If you intentionally train from mirrored webcam images, pass `--mirror-input` while recreating the landmark CSV dataset and keep the same orientation during prediction.

For fast sentence signing, the web app accepts a new stable word immediately when the label changes. A blank release is only needed if you want to enter the same word twice in a row.

Both live predictors now resolve files from the project folder itself, so they can be launched even when your terminal is opened in a different directory.

## Website Run

Run the website backend:

```powershell
python web_app.py
```

Then open:

```text
http://127.0.0.1:8000
```

Website features:
- webcam prediction
- image upload prediction
- model switch between Auto, Landmark, and YOLO
- Landmark mode is the default and uses hand-centered landmark features instead of image background pixels
- Auto mode uses landmark prediction first, then high-confidence YOLO fallback when landmarks are not reliable
- sentence text output from recognized word sequences
- speech output for accepted words and full sentence replay
- low-latency camera polling with queued speech so recognition does not wait for audio playback
- automatic normal/mirrored landmark checking for live camera orientation differences
- simplified result-focused interface

Note: the `hungry` source images do not expose usable face/hand landmarks, so `hungry` is handled by YOLO mode and Auto fallback rather than Landmark-only mode.

## Recreate Landmark Dataset

```powershell
python create_text_gesture_data_from_images.py
```

## Train Landmark Model

```powershell
python train_text_gesture_model.py
```

## Test Landmark Model

```powershell
python test_text_gesture_model.py
```

## Recreate YOLO Dataset

```powershell
python prepare_yolo_classification_dataset.py --source-folder "images for phrases" --output-folder yolo_phrase_dataset --overwrite
```

## Train YOLO Model

```powershell
python train_yolo_phrase_model.py --dataset-folder yolo_phrase_dataset --model yolov8n-cls.yaml --epochs 25 --imgsz 224 --batch 16 --device cpu --save-model-to text_phrase_yolo_cls.pt
```

## Custom Dataset Training

You can train on your own image folders with one command:

```powershell
python custom_train.py
```

Or run it directly with arguments:

```powershell
python custom_train.py --images-folder "C:\path\to\my_images" --mode both --project-name my_project --overwrite
```

This script can:
- create landmark CSV data from your image folders
- train a landmark model
- prepare a YOLO dataset split
- train a YOLO model
- print the exact prediction commands for the new custom models
