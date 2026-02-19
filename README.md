# Deep Image Analysis Lab

This project is a Streamlit app for deep single-image intelligence and reasoning.

It combines:
- structural image analysis (focus, entropy, brightness, contrast)
- object detection
- OCR text extraction
- segmentation via superpixels
- scene graph construction (objects + text + spatial links)
- behavior/action hypotheses from contextual rules
- reference chaining for explainable reasoning trails
- novelty scoring from uncommon object relationships and visual complexity

## Why this is novel

Instead of only predicting objects, the pipeline builds a graph-centric reasoning layer:
- links text to nearby objects
- computes object-to-object spatial relations
- emits interpretable chains like `person -> interacts_with -> laptop`
- estimates scene novelty from uncommon co-occurrences

This gives a richer "intelligence report" rather than a plain detector output.

## Project structure

- `app.py` - Streamlit UI
- `src/deep_image_analysis/models.py` - core data models
- `src/deep_image_analysis/vision.py` - computer vision and OCR helpers
- `src/deep_image_analysis/reasoning.py` - scene graph + chaining + novelty
- `src/deep_image_analysis/pipeline.py` - full orchestration pipeline

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then upload an image and inspect the tabs:
- Overview
- Detection + Segments
- OCR
- Scene Graph
- Behavior + Novelty
- Raw JSON

## Notes

- `ultralytics` downloads model weights on first run.
- `easyocr` may download OCR models on first run.
- Outputs are probabilistic and are not a forensic certainty system.