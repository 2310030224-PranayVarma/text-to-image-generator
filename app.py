from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from deep_image_analysis import analyze_image
from deep_image_analysis.vision import bgr_to_rgb, label_distribution

st.set_page_config(page_title="Deep Image Analysis Lab", page_icon="🧠", layout="wide")

st.title("🧠 Deep Image Analysis Lab")
st.caption(
    "Multi-layer image intelligence: structure, object detection, OCR, scene graph, action inference, segmentation, and novelty estimation."
)

with st.sidebar:
    st.header("Analysis Controls")
    detection_confidence = st.slider("Detection confidence", min_value=0.05, max_value=0.9, value=0.25, step=0.05)
    superpixels = st.slider("Superpixel granularity", min_value=30, max_value=220, value=80, step=10)
    st.info(
        "This app produces probabilistic signals, not forensic certainty. Use as decision support, not as a standalone evidentiary system."
    )

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"])

if uploaded_file is None:
    st.markdown("### What you get")
    st.markdown(
        """
- Structural profile (focus, entropy, lighting, edge density)
- Object detection + density map cues
- OCR text extraction and local grounding
- Superpixel segmentation summary
- Scene graph with relation edges
- Action/behavior hypotheses
- Reference-chaining report
- Novelty and anomaly-style score
        """
    )
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

col1, col2 = st.columns([1.3, 1])
with col1:
    st.image(image, caption="Input", use_container_width=True)
with col2:
    st.write("**Image Details**")
    st.json({"filename": uploaded_file.name, "size": image.size, "mode": image.mode})

with st.spinner("Running deep analysis pipeline..."):
    report = analyze_image(
        image=image,
        detection_confidence=detection_confidence,
        n_superpixels=superpixels,
    )

summary_col1, summary_col2, summary_col3 = st.columns(3)
summary_col1.metric("Objects", len(report.detections))
summary_col2.metric("Text tokens", len(report.text_tokens))
summary_col3.metric("Reasoning Depth", report.context_reasoning["reasoning_depth"])


tab_overview, tab_detection, tab_text, tab_graph, tab_reasoning, tab_raw = st.tabs(
    ["Overview", "Detection + Segments", "OCR", "Scene Graph", "Behavior + Novelty", "Raw JSON"]
)

with tab_overview:
    st.subheader("Structural Analysis")
    st.json(report.structure)

    if report.detections:
        label_counts = label_distribution(report.detections)
        chart_df = pd.DataFrame({"label": list(label_counts.keys()), "count": list(label_counts.values())})
        st.subheader("Object Distribution")
        st.bar_chart(chart_df.set_index("label"))
    else:
        st.warning("No objects detected. Try lowering confidence threshold.")

with tab_detection:
    st.subheader("Detection Overlay")
    st.image(bgr_to_rgb(report.overlays["detections"]), use_container_width=True)

    st.subheader("Top Segments")
    seg_rows = [
        {
            "region_id": s.region_id,
            "area_px": s.area_px,
            "centroid_x": round(s.centroid[0], 1),
            "centroid_y": round(s.centroid[1], 1),
            "mean_color": tuple(round(v, 1) for v in s.mean_color_rgb),
        }
        for s in report.segments[:25]
    ]
    st.dataframe(pd.DataFrame(seg_rows), use_container_width=True)

with tab_text:
    st.subheader("OCR Overlay")
    st.image(bgr_to_rgb(report.overlays["ocr"]), use_container_width=True)
    if report.text_tokens:
        rows = [
            {
                "text": t.text,
                "confidence": round(t.confidence, 3),
                "bbox": [round(t.bbox.x1, 1), round(t.bbox.y1, 1), round(t.bbox.x2, 1), round(t.bbox.y2, 1)],
            }
            for t in report.text_tokens
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No OCR text extracted.")

with tab_graph:
    st.subheader("Scene Graph (JSON)")
    st.json(report.scene_graph)

with tab_reasoning:
    st.subheader("Context Reasoning")
    st.write(report.context_reasoning["summary"])

    st.subheader("Action Hypotheses")
    if report.action_hypotheses:
        st.dataframe(pd.DataFrame(report.action_hypotheses), use_container_width=True)
    else:
        st.info("No action hypothesis matched current object patterns.")

    st.subheader("Reference Chains")
    st.dataframe(pd.DataFrame(report.chains), use_container_width=True)

    st.subheader("Anomaly Signals")
    st.json(report.novelty)

with tab_raw:
    st.subheader("Complete Report")

    serializable = {
        "image_shape": report.image_shape,
        "metadata": report.metadata,
        "structure": report.structure,
        "detections": [
            {
                "label": d.label,
                "confidence": d.confidence,
                "bbox": d.bbox.__dict__,
                "source": d.source,
            }
            for d in report.detections
        ],
        "text_tokens": [
            {
                "text": t.text,
                "confidence": t.confidence,
                "bbox": t.bbox.__dict__,
            }
            for t in report.text_tokens
        ],
        "segments": [s.__dict__ for s in report.segments],
        "action_hypotheses": report.action_hypotheses,
        "chains": report.chains,
        "context_reasoning": report.context_reasoning,
        "novelty": report.novelty,
        "scene_graph": report.scene_graph,
    }

    st.code(json.dumps(serializable, indent=2, default=str), language="json")
