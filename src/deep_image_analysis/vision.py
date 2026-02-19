from __future__ import annotations

from collections import Counter
from fractions import Fraction
from typing import Any

import cv2
import numpy as np
from PIL import ExifTags, Image
from skimage.segmentation import slic

from .models import BoundingBox, Detection, OCRToken, SegmentRegion


def to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    try:
        return float(value)
    except Exception:
        return str(value)


def pil_to_bgr_array(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def extract_metadata(image: Image.Image) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "format": image.format,
        "mode": image.mode,
        "width": image.size[0],
        "height": image.size[1],
    }
    try:
        exif_data = image.getexif()
        parsed = {}
        for key, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(key, str(key))
            parsed[tag_name] = to_json_safe(value)
        metadata["exif"] = parsed
    except Exception:
        metadata["exif"] = {}
    return metadata


def image_structure_stats(image_bgr: np.ndarray) -> dict[str, Any]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    color_hist = cv2.calcHist([image_bgr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_norm = color_hist / (np.sum(color_hist) + 1e-8)
    color_entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-12)))

    thirds = {
        "top_left": (0, h // 3, 0, w // 3),
        "top_mid": (0, h // 3, w // 3, 2 * w // 3),
        "top_right": (0, h // 3, 2 * w // 3, w),
        "mid_left": (h // 3, 2 * h // 3, 0, w // 3),
        "center": (h // 3, 2 * h // 3, w // 3, 2 * w // 3),
        "mid_right": (h // 3, 2 * h // 3, 2 * w // 3, w),
        "bottom_left": (2 * h // 3, h, 0, w // 3),
        "bottom_mid": (2 * h // 3, h, w // 3, 2 * w // 3),
        "bottom_right": (2 * h // 3, h, 2 * w // 3, w),
    }
    thirds_brightness = {
        k: float(np.mean(gray[r0:r1, c0:c1]))
        for k, (r0, r1, c0, c1) in thirds.items()
    }

    return {
        "edge_density": edge_density,
        "focus_laplacian_variance": lap_var,
        "brightness": brightness,
        "contrast": contrast,
        "color_entropy": color_entropy,
        "rule_of_thirds_brightness": thirds_brightness,
    }


def run_detection_ultralytics(image_bgr: np.ndarray, conf: float = 0.25) -> list[Detection]:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        return []

    model = YOLO("yolov8n.pt")
    results = model(image_bgr, conf=conf, verbose=False)

    detections: list[Detection] = []
    for result in results:
        names = result.names
        for box in result.boxes:
            cls_idx = int(box.cls.item())
            score = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    label=str(names.get(cls_idx, cls_idx)),
                    confidence=score,
                    bbox=BoundingBox(x1, y1, x2, y2),
                    source="ultralytics-yolov8n",
                )
            )
    return detections


def run_ocr_easyocr(image_bgr: np.ndarray) -> list[OCRToken]:
    try:
        import easyocr  # type: ignore
    except Exception:
        return []

    reader = easyocr.Reader(["en"], gpu=False)
    raw = reader.readtext(image_bgr)

    tokens: list[OCRToken] = []
    for item in raw:
        points, text, score = item
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        token = OCRToken(
            text=str(text).strip(),
            confidence=float(score),
            bbox=BoundingBox(float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))),
        )
        if token.text:
            tokens.append(token)
    return tokens


def run_segmentation_superpixels(image_bgr: np.ndarray, n_segments: int = 80) -> list[SegmentRegion]:
    image_rgb = bgr_to_rgb(image_bgr)
    segments = slic(image_rgb, n_segments=n_segments, compactness=10.0, sigma=1.0, start_label=1)

    result: list[SegmentRegion] = []
    for region_id in np.unique(segments):
        mask = segments == region_id
        area = int(mask.sum())
        if area < 40:
            continue
        ys, xs = np.where(mask)
        mean_color = image_rgb[mask].mean(axis=0)
        result.append(
            SegmentRegion(
                region_id=int(region_id),
                area_px=area,
                centroid=(float(np.mean(xs)), float(np.mean(ys))),
                mean_color_rgb=(float(mean_color[0]), float(mean_color[1]), float(mean_color[2])),
            )
        )
    result.sort(key=lambda s: s.area_px, reverse=True)
    return result


def label_distribution(detections: list[Detection]) -> dict[str, int]:
    return dict(Counter(d.label for d in detections))


def draw_detection_overlay(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2])
        cv2.rectangle(out, (x1, y1), (x2, y2), (36, 255, 12), 2)
        text = f"{det.label} {det.confidence:.2f}"
        cv2.putText(out, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
    return out


def draw_text_overlay(image_bgr: np.ndarray, tokens: list[OCRToken]) -> np.ndarray:
    out = image_bgr.copy()
    for token in tokens:
        x1, y1, x2, y2 = map(int, [token.bbox.x1, token.bbox.y1, token.bbox.x2, token.bbox.y2])
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(out, token.text[:30], (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2)
    return out
