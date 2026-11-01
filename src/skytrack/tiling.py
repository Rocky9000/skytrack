"""Sliced inference for small objects in large aerial frames.

A 4K drone frame resized to a 640 px network input shrinks a 20 px pedestrian
to about 3 px, below what most detectors can see. Running the detector on
overlapping full-resolution tiles and merging the results recovers those
objects at the cost of more forward passes (the idea behind SAHI).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .boxes import Detections, batched_nms

DetectFn = Callable[[np.ndarray], Detections]


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int


def _axis_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)  # last tile sits flush with the edge
    return starts


def make_tiles(width: int, height: int, tile: int = 640, overlap: float = 0.2) -> list[Tile]:
    """Cover the image with ``tile``-sized windows overlapping by ``overlap`` (0-1)."""
    if tile <= 0 or not 0 <= overlap < 1:
        raise ValueError("tile must be > 0 and overlap in [0, 1)")
    stride = max(1, int(round(tile * (1 - overlap))))
    return [
        Tile(x, y, min(x + tile, width), min(y + tile, height))
        for y in _axis_starts(height, tile, stride)
        for x in _axis_starts(width, tile, stride)
    ]


def sliced_predict(
    image: np.ndarray,
    detect: DetectFn,
    tile: int = 640,
    overlap: float = 0.2,
    iou_thr: float = 0.5,
    full_frame: bool = True,
) -> Detections:
    """Run ``detect`` on every tile (and optionally the whole frame), then merge.

    Tile detections are shifted back to frame coordinates. The full-frame
    pass keeps large objects that tiles would cut in half; class-aware NMS
    removes the duplicates both passes produce in overlap regions.
    """
    h, w = image.shape[:2]
    parts: list[Detections] = []
    for t in make_tiles(w, h, tile, overlap):
        d = detect(image[t.y0 : t.y1, t.x0 : t.x1])
        if len(d):
            shifted = d.boxes.astype(np.float32) + np.array([t.x0, t.y0, t.x0, t.y0], np.float32)
            parts.append(Detections(shifted, d.scores, d.classes))
    if full_frame and (w > tile or h > tile):
        parts.append(detect(image))
    return batched_nms(Detections.concat(parts), iou_thr)
