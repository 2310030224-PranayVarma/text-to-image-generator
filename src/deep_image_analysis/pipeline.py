from __future__ import annotations

from typing import Any

from PIL import Image

from .models import AnalysisReport
from .reasoning import (
    build_context_reasoning,
    build_reference_chains,
    build_scene_graph,
    infer_actions,
    novelty_signals,
)
from .vision import (
    draw_detection_overlay,
    draw_text_overlay,
    extract_metadata,
    image_structure_stats,
    pil_to_bgr_array,
    run_detection_ultralytics,
    run_ocr_easyocr,
    run_segmentation_superpixels,
)


def analyze_image(
    image: Image.Image,
    detection_confidence: float = 0.25,
    n_superpixels: int = 80,
) -> AnalysisReport:
    image_bgr = pil_to_bgr_array(image)
    h, w = image_bgr.shape[:2]

    metadata = extract_metadata(image)
    structure = image_structure_stats(image_bgr)

    detections = run_detection_ultralytics(image_bgr=image_bgr, conf=detection_confidence)
    text_tokens = run_ocr_easyocr(image_bgr=image_bgr)
    segments = run_segmentation_superpixels(image_bgr=image_bgr, n_segments=n_superpixels)

    scene_graph = build_scene_graph(
        detections=detections,
        tokens=text_tokens,
        image_w=w,
        image_h=h,
    )
    actions = infer_actions(detections)
    chains = build_reference_chains(detections=detections, tokens=text_tokens, scene_graph=scene_graph)
    context_reasoning = build_context_reasoning(
        detections=detections,
        tokens=text_tokens,
        chains=chains,
        actions=actions,
        scene_graph=scene_graph,
    )
    novelty = novelty_signals(detections=detections, structure=structure)

    overlays: dict[str, Any] = {
        "detections": draw_detection_overlay(image_bgr, detections),
        "ocr": draw_text_overlay(image_bgr, text_tokens),
    }

    return AnalysisReport(
        image_shape=image_bgr.shape,
        metadata=metadata,
        structure=structure,
        detections=detections,
        text_tokens=text_tokens,
        segments=segments,
        action_hypotheses=actions,
        chains=chains,
        context_reasoning=context_reasoning,
        novelty=novelty,
        scene_graph=scene_graph,
        overlays=overlays,
    )
