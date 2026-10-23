"""Instance-mask helpers.

The detector already returns one mask outline per detection (from the same
YOLO-seg forward pass), so this module does not run another model. It turns
those outlines into binary masks, measures them, and draws them efficiently.

Note: these are *instance* masks (one per person / vehicle). They are not
semantic scene segmentation and not clothing segmentation.
"""

from __future__ import annotations

import cv2
import numpy as np


def polygon_mask_in_box(polygon: np.ndarray | None, box: np.ndarray, frame_shape: tuple[int, int]) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
    """Rasterise a mask outline inside its (clipped) bounding box.

    Returns (mask, (x1, y1, x2, y2)) where mask is a uint8 array of 0/1 the size of the
    clipped box, or (None, box) when the outline is missing.
    """
    h, w = frame_shape[:2]
    x1, y1 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x2, y2 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    if polygon is None or x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    pts = np.round(polygon - np.array([x1, y1], dtype=np.float32)).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask, (x1, y1, x2, y2)


def mask_fill_ratio(polygon: np.ndarray | None, box: np.ndarray, frame_shape: tuple[int, int]) -> float | None:
    """Mask area divided by box area. Low values hint at heavy occlusion or a poor mask."""
    mask, (x1, y1, x2, y2) = polygon_mask_in_box(polygon, box, frame_shape)
    if mask is None:
        return None
    area = (x2 - x1) * (y2 - y1)
    return float(mask.sum()) / area if area else None


def render_instance_masks(frame: np.ndarray, items: list[tuple[np.ndarray, tuple[int, int, int]]], alpha: float = 0.4) -> None:
    """Blend coloured instance masks into ``frame`` in place.

    Only the region covered by the masks is blended, which is much cheaper than a
    full-frame copy at 1080p. Each instance also gets an outline in its own colour
    so overlapping people stay distinguishable.
    """
    polys = [(np.round(p).astype(np.int32), c) for p, c in items if p is not None and len(p) >= 3]
    if not polys:
        return
    h, w = frame.shape[:2]
    all_pts = np.concatenate([p for p, _ in polys])
    x1, y1 = np.clip(all_pts.min(axis=0), 0, [w - 1, h - 1])
    x2, y2 = np.clip(all_pts.max(axis=0) + 1, 1, [w, h])
    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    offset = np.array([x1, y1], dtype=np.int32)
    for pts, color in polys:
        cv2.fillPoly(overlay, [pts - offset], color)
    covered = np.any(overlay != roi, axis=2)
    blended = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)
    roi[covered] = blended[covered]
    for pts, color in polys:
        cv2.polylines(roi, [pts - offset], True, color, 1, cv2.LINE_AA)
