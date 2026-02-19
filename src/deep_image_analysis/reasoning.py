from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np

from .models import Detection, GraphEdge, GraphNode, OCRToken


COMMON_OBJECT_PAIRS = {
    tuple(sorted(pair))
    for pair in [
        ("person", "cell phone"),
        ("person", "laptop"),
        ("person", "chair"),
        ("car", "person"),
        ("person", "dog"),
        ("person", "backpack"),
        ("dining table", "chair"),
        ("person", "bicycle"),
        ("person", "motorcycle"),
        ("tv", "couch"),
    ]
}


ACTION_RULES = [
    {
        "if": {"person", "bicycle"},
        "action": "riding",
        "confidence": 0.72,
        "evidence": "person-bicycle co-occurrence",
    },
    {
        "if": {"person", "laptop"},
        "action": "working_on_computer",
        "confidence": 0.68,
        "evidence": "person near laptop",
    },
    {
        "if": {"person", "cell phone"},
        "action": "using_phone",
        "confidence": 0.65,
        "evidence": "person and phone jointly visible",
    },
    {
        "if": {"person", "dog"},
        "action": "walking_or_caring_for_pet",
        "confidence": 0.64,
        "evidence": "person-dog contextual pattern",
    },
    {
        "if": {"person", "book"},
        "action": "reading",
        "confidence": 0.60,
        "evidence": "person-book pair",
    },
]


def _bbox_iou(a: Detection, b: Detection) -> float:
    x1 = max(a.bbox.x1, b.bbox.x1)
    y1 = max(a.bbox.y1, b.bbox.y1)
    x2 = min(a.bbox.x2, b.bbox.x2)
    y2 = min(a.bbox.y2, b.bbox.y2)

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    union = a.bbox.area + b.bbox.area - inter + 1e-8
    return float(inter / union)


def _normalized_center_distance(a: Detection, b: Detection, image_w: int, image_h: int) -> float:
    ax, ay = a.bbox.center
    bx, by = b.bbox.center
    dx = (ax - bx) / max(1.0, image_w)
    dy = (ay - by) / max(1.0, image_h)
    return float(np.sqrt(dx * dx + dy * dy))


def _spatial_relation(a: Detection, b: Detection) -> str:
    ax, ay = a.bbox.center
    bx, by = b.bbox.center
    if abs(ax - bx) > abs(ay - by):
        return "left_of" if ax < bx else "right_of"
    return "above" if ay < by else "below"


def build_scene_graph(
    detections: list[Detection],
    tokens: list[OCRToken],
    image_w: int,
    image_h: int,
) -> dict[str, Any]:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    for idx, det in enumerate(detections):
        nodes.append(
            GraphNode(
                node_id=f"obj_{idx}",
                node_type="object",
                label=det.label,
                attrs={"confidence": det.confidence, "bbox": det.bbox.__dict__},
            )
        )

    for idx, token in enumerate(tokens):
        nodes.append(
            GraphNode(
                node_id=f"txt_{idx}",
                node_type="text",
                label=token.text,
                attrs={"confidence": token.confidence, "bbox": token.bbox.__dict__},
            )
        )

    for (i, a), (j, b) in combinations(enumerate(detections), 2):
        relation = _spatial_relation(a, b)
        dist = _normalized_center_distance(a, b, image_w, image_h)
        iou = _bbox_iou(a, b)
        relation_score = float(max(0.05, (1.0 - dist) * (0.6 + iou)))
        edges.append(GraphEdge(src=f"obj_{i}", dst=f"obj_{j}", relation=relation, score=relation_score))
        if dist < 0.18:
            edges.append(GraphEdge(src=f"obj_{i}", dst=f"obj_{j}", relation="near", score=float(1.0 - dist)))
        if iou > 0.15:
            edges.append(GraphEdge(src=f"obj_{i}", dst=f"obj_{j}", relation="overlaps", score=iou))

    for t_idx, token in enumerate(tokens):
        if not detections:
            continue
        tx, ty = token.bbox.center
        nearest = min(
            enumerate(detections),
            key=lambda d: ((d[1].bbox.center[0] - tx) ** 2 + (d[1].bbox.center[1] - ty) ** 2),
        )
        obj_idx, obj_det = nearest
        dist = _normalized_center_distance(
            obj_det,
            Detection(label="text", confidence=token.confidence, bbox=token.bbox),
            image_w,
            image_h,
        )
        edges.append(
            GraphEdge(
                src=f"txt_{t_idx}",
                dst=f"obj_{obj_idx}",
                relation="text_near_object",
                score=float(max(0.1, 1.0 - dist)),
            )
        )

    return {
        "nodes": [n.__dict__ for n in nodes],
        "edges": [e.__dict__ for e in edges],
    }


def infer_actions(detections: list[Detection]) -> list[dict[str, Any]]:
    labels = {d.label for d in detections}
    hypotheses: list[dict[str, Any]] = []
    for rule in ACTION_RULES:
        if rule["if"].issubset(labels):
            hypotheses.append(
                {
                    "action": rule["action"],
                    "confidence": rule["confidence"],
                    "evidence": rule["evidence"],
                }
            )
    hypotheses.sort(key=lambda x: x["confidence"], reverse=True)
    return hypotheses


def build_reference_chains(
    detections: list[Detection],
    tokens: list[OCRToken],
    scene_graph: dict[str, Any],
) -> list[dict[str, Any]]:
    label_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, det in enumerate(detections):
        label_to_indices[det.label].append(idx)

    chains: list[dict[str, Any]] = []

    for person_idx in label_to_indices.get("person", []):
        for target_label in ["laptop", "cell phone", "book", "bicycle", "dog"]:
            for obj_idx in label_to_indices.get(target_label, []):
                chains.append(
                    {
                        "chain": [f"obj_{person_idx}", f"obj_{obj_idx}"],
                        "statement": f"person -> interacts_with -> {target_label}",
                        "type": "interaction",
                    }
                )

    if tokens:
        for idx, token in enumerate(tokens[:8]):
            chains.append(
                {
                    "chain": [f"txt_{idx}"],
                    "statement": f"text cue: '{token.text}'",
                    "type": "text",
                }
            )

    edge_buckets: dict[str, int] = defaultdict(int)
    for edge in scene_graph.get("edges", []):
        edge_buckets[edge["relation"]] += 1

    chains.append(
        {
            "chain": ["graph_summary"],
            "statement": "spatial relations: " + ", ".join(f"{k}={v}" for k, v in sorted(edge_buckets.items())),
            "type": "topology",
        }
    )

    return chains


def novelty_signals(detections: list[Detection], structure: dict[str, Any]) -> dict[str, Any]:
    labels = [d.label for d in detections]
    pairs = [tuple(sorted(p)) for p in combinations(labels, 2)]

    uncommon_pairs = [p for p in pairs if p not in COMMON_OBJECT_PAIRS]
    pair_uncommon_ratio = float(len(uncommon_pairs) / max(1, len(pairs)))

    object_diversity = float(len(set(labels)) / max(1, len(labels)))
    entropy_proxy = float(structure.get("color_entropy", 0.0) / 9.0)
    edge_density = float(structure.get("edge_density", 0.0))

    novelty_score = float(np.clip(0.5 * pair_uncommon_ratio + 0.3 * object_diversity + 0.2 * entropy_proxy, 0.0, 1.0))

    descriptor = "routine"
    if novelty_score > 0.75:
        descriptor = "highly_unusual"
    elif novelty_score > 0.5:
        descriptor = "uncommon"

    return {
        "novelty_score": novelty_score,
        "descriptor": descriptor,
        "uncommon_object_pairs": uncommon_pairs[:30],
        "object_diversity": object_diversity,
        "texture_complexity": edge_density,
    }


def build_context_reasoning(
    detections: list[Detection],
    tokens: list[OCRToken],
    chains: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    scene_graph: dict[str, Any],
) -> dict[str, Any]:
    label_counts: dict[str, int] = defaultdict(int)
    for det in detections:
        label_counts[det.label] += 1

    relation_counts: dict[str, int] = defaultdict(int)
    for edge in scene_graph.get("edges", []):
        relation_counts[str(edge.get("relation", "unknown"))] += 1

    top_objects = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)[:6]
    top_relations = sorted(relation_counts.items(), key=lambda item: item[1], reverse=True)[:6]

    action_text = (
        ", ".join(f"{a['action']} ({a['confidence']:.2f})" for a in actions[:4])
        if actions
        else "No strong action pattern matched the current object context."
    )

    text_evidence = [t.text for t in tokens if t.text.strip()][:8]
    text_reasoning = (
        "Text cues found: " + "; ".join(text_evidence)
        if text_evidence
        else "No readable text cues were extracted from the image."
    )

    chain_statements = [c.get("statement", "") for c in chains if c.get("statement")]
    key_chains = chain_statements[:8]

    primary_object_sentence = (
        "Primary objects: " + ", ".join(f"{label} x{count}" for label, count in top_objects)
        if top_objects
        else "No objects were detected with the current threshold."
    )
    relation_sentence = (
        "Dominant spatial relations: " + ", ".join(f"{rel}={count}" for rel, count in top_relations)
        if top_relations
        else "No object-object spatial graph edges were formed."
    )

    narrative_lines = [
        primary_object_sentence,
        relation_sentence,
        f"Action inference: {action_text}",
        text_reasoning,
    ]
    if key_chains:
        narrative_lines.append("Reference chain highlights: " + " | ".join(key_chains[:5]))

    reasoning_depth = int(len(top_objects) + len(top_relations) + len(actions) + len(text_evidence) + len(key_chains))

    return {
        "summary": " ".join(narrative_lines),
        "reasoning_depth": reasoning_depth,
        "primary_objects": [{"label": label, "count": count} for label, count in top_objects],
        "dominant_relations": [{"relation": rel, "count": count} for rel, count in top_relations],
        "action_reasoning": actions,
        "text_reasoning": text_reasoning,
        "key_chains": key_chains,
    }
