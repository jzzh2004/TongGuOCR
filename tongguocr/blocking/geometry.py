from __future__ import annotations

from statistics import mean, median, pstdev

from .models import BBox, TextLine


EPSILON = 1e-6


def bbox_width(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0])


def bbox_height(bbox: BBox) -> float:
    return max(0.0, bbox[3] - bbox[1])


def bbox_area(bbox: BBox) -> float:
    return bbox_width(bbox) * bbox_height(bbox)


def union_bbox(lhs: BBox, rhs: BBox) -> BBox:
    return (
        min(lhs[0], rhs[0]),
        min(lhs[1], rhs[1]),
        max(lhs[2], rhs[2]),
        max(lhs[3], rhs[3]),
    )


def union_all(bboxes: list[BBox]) -> BBox:
    if not bboxes:
        raise ValueError("union_all requires at least one bbox")
    current = bboxes[0]
    for bbox in bboxes[1:]:
        current = union_bbox(current, bbox)
    return current


def overlap_length(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def intersection_area(lhs: BBox, rhs: BBox) -> float:
    return overlap_length(lhs[0], lhs[2], rhs[0], rhs[2]) * overlap_length(lhs[1], lhs[3], rhs[1], rhs[3])


def interval_gap(a1: float, a2: float, b1: float, b2: float) -> float:
    if overlap_length(a1, a2, b1, b2) > 0:
        return 0.0
    if a2 < b1:
        return b1 - a2
    return a1 - b2


def aspect_ratio(bbox: BBox) -> float:
    width = bbox_width(bbox)
    height = bbox_height(bbox)
    short = max(min(width, height), EPSILON)
    long = max(width, height)
    return long / short


def fill_ratio(sum_line_area: float, bbox: BBox) -> float:
    ratio = sum_line_area / max(bbox_area(bbox), EPSILON)
    return min(1.0, ratio)


def iou(lhs: BBox, rhs: BBox) -> float:
    inter = intersection_area(lhs, rhs)
    denom = bbox_area(lhs) + bbox_area(rhs) - inter
    return inter / max(denom, EPSILON)


def containment_ratio(lhs: BBox, rhs: BBox) -> float:
    inter = intersection_area(lhs, rhs)
    return inter / max(min(bbox_area(lhs), bbox_area(rhs)), EPSILON)


def bbox_y_center(bbox: BBox) -> float:
    return (bbox[1] + bbox[3]) * 0.5


def y_band_overlap_ratio(band_bbox: BBox, line_bbox: BBox) -> float:
    overlap = overlap_length(band_bbox[1], band_bbox[3], line_bbox[1], line_bbox[3])
    return overlap / max(min(bbox_height(band_bbox), bbox_height(line_bbox)), EPSILON)


def y_band_stability_score(bboxes: list[BBox], median_line_height: float) -> float:
    if len(bboxes) <= 1:
        return 1.0
    centers = [bbox_y_center(bbox) for bbox in bboxes]
    spans = [bbox_height(bbox) for bbox in bboxes]
    center_scale = max(mean(spans), median_line_height, EPSILON)
    span_scale = max(mean(spans), EPSILON)
    center_spread = pstdev(centers) / center_scale
    span_spread = pstdev(spans) / span_scale
    return 1.0 / (1.0 + center_spread + span_spread)


def median_line_size(lines: list[TextLine]) -> tuple[float, float]:
    widths = [bbox_width(line.bbox) for line in lines]
    heights = [bbox_height(line.bbox) for line in lines]
    return median(widths), median(heights)


def adjacency_features(
    prev_bbox: BBox,
    curr_bbox: BBox,
    median_line_width: float,
) -> tuple[float, float]:
    """Return normalized x gap and y overlap ratio for vertical layout."""

    x_gap = interval_gap(prev_bbox[0], prev_bbox[2], curr_bbox[0], curr_bbox[2])
    y_overlap = overlap_length(prev_bbox[1], prev_bbox[3], curr_bbox[1], curr_bbox[3])
    prev_height = bbox_height(prev_bbox)
    curr_height = bbox_height(curr_bbox)
    overlap_ratio = y_overlap / max(min(prev_height, curr_height), EPSILON)
    x_gap_norm = x_gap / max(median_line_width, EPSILON)
    return x_gap_norm, overlap_ratio
