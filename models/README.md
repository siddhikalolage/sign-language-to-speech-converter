# Model Artifacts

This directory documents the trained model artifacts used by the
Sign Language to Speech Converter.

The project uses the existing trained production model. **Retraining is
not required for normal application setup or validation.**

## Production Landmark Pipeline

The primary landmark-based inference pipeline is:

```text
144 raw landmark features
        ↓
GestureFeatureExtractor
        ↓
328 engineered geometric features
        ↓
ExtraTreesClassifier
        ↓
43 gesture classes
        ↓
Label Encoder
        ↓
Human-readable phrase
