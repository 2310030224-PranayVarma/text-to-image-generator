from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.width / 2.0, self.y1 + self.height / 2.0)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: BoundingBox
    source: str = "detector"


@dataclass
class OCRToken:
    text: str
    confidence: float
    bbox: BoundingBox


@dataclass
class SegmentRegion:
    region_id: int
    area_px: int
    centroid: tuple[float, float]
    mean_color_rgb: tuple[float, float, float]


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    label: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    src: str
    dst: str
    relation: str
    score: float


@dataclass
class AnalysisReport:
    image_shape: tuple[int, int, int]
    metadata: dict[str, Any]
    structure: dict[str, Any]
    detections: list[Detection]
    text_tokens: list[OCRToken]
    segments: list[SegmentRegion]
    action_hypotheses: list[dict[str, Any]]
    chains: list[dict[str, Any]]
    context_reasoning: dict[str, Any]
    novelty: dict[str, Any]
    scene_graph: dict[str, Any]
    overlays: dict[str, Any]
