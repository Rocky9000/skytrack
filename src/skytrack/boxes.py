"""Box utilities. All boxes are float arrays in ``xyxy`` pixel coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detections:
    boxes: np.ndarray  # (N, 4) xyxy
    scores: np.ndarray  # (N,)
    classes: np.ndarray  # (N,) int

    @classmethod
    def empty(cls) -> Detections:
        return cls(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))

    def __len__(self) -> int:
        return len(self.boxes)

    def __getitem__(self, idx) -> Detections:
        return Detections(self.boxes[idx], self.scores[idx], self.classes[idx])

    @staticmethod
    def concat(parts: list[Detections]) -> Detections:
        parts = [p for p in parts if len(p)]
        if not parts:
            return Detections.empty()
        return Detections(
            np.concatenate([p.boxes for p in parts]).astype(np.float32),
            np.concatenate([p.scores for p in parts]).astype(np.float32),
            np.concatenate([p.classes for p in parts]).astype(np.int64),
        )


def area(b: np.ndarray) -> np.ndarray:
    return np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU, shape (len(a), len(b))."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """Greedy non-maximum suppression. Returns kept indices, highest score first."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(dets: Detections, iou_thr: float = 0.5) -> Detections:
    """Class-aware NMS via the coordinate-offset trick (boxes of different classes never overlap)."""
    if len(dets) == 0:
        return dets
    offset = dets.classes[:, None].astype(np.float64) * (dets.boxes.max() + 1.0)
    return dets[nms(dets.boxes + offset, dets.scores, iou_thr)]


def xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64)
    return np.concatenate([b[..., :2], b[..., :2] + b[..., 2:4]], axis=-1)


def cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64)
    half = b[..., 2:4] / 2
    return np.concatenate([b[..., :2] - half, b[..., :2] + half], axis=-1)
