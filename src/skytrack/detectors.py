"""Detector backends: ONNX Runtime (deployment) and Ultralytics (training/research).

Both return ``Detections`` in original-image pixel coordinates, so they plug
straight into ``sliced_predict`` and the trackers.
"""

from __future__ import annotations

import numpy as np

from .boxes import Detections, batched_nms, cxcywh_to_xyxy


def letterbox(img: np.ndarray, size: int = 640, color: int = 114) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio, pad to ``size`` x ``size``. Returns (image, scale, (pad_x, pad_y))."""
    import cv2

    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, img.shape[2]), color, dtype=img.dtype)
    out[py : py + nh, px : px + nw] = resized
    return out, r, (px, py)


def unletterbox(boxes: np.ndarray, scale: float, pad: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    """Map boxes from letterboxed network space back to the original image and clip."""
    b = boxes.astype(np.float64).copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - pad[0]) / scale
    b[:, [1, 3]] = (b[:, [1, 3]] - pad[1]) / scale
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, shape[1])
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, shape[0])
    return b


def decode_yolov8(output: np.ndarray, conf_thr: float = 0.25, iou_thr: float = 0.5, max_det: int = 1000) -> Detections:
    """Decode a raw YOLOv8/YOLO11 head of shape (1, 4 + num_classes, num_anchors).

    Rows 0-3 are (cx, cy, w, h) in network pixels; the rest are per-class
    scores (already sigmoid-activated in exported models; there is no
    separate objectness term).
    """
    pred = np.asarray(output)[0].T  # (anchors, 4 + nc)
    cls_scores = pred[:, 4:]
    cls = cls_scores.argmax(axis=1)
    score = cls_scores[np.arange(len(pred)), cls]
    keep = score >= conf_thr
    if not keep.any():
        return Detections.empty()
    dets = Detections(cxcywh_to_xyxy(pred[keep, :4]).astype(np.float32),
                      score[keep].astype(np.float32), cls[keep].astype(np.int64))
    dets = batched_nms(dets, iou_thr)
    return dets[: max_det]


class OnnxDetector:
    """YOLOv8/11 exported with ``yolo export format=onnx`` running on ONNX Runtime."""

    def __init__(self, path: str, imgsz: int = 640, conf: float = 0.25, iou: float = 0.5, providers=None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(path, providers=providers or ["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.imgsz, self.conf, self.iou = imgsz, conf, iou

    def __call__(self, img_bgr: np.ndarray) -> Detections:
        lb, scale, pad = letterbox(img_bgr, self.imgsz)
        x = np.ascontiguousarray(lb[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        out = self.session.run(None, {self.input: x})[0]
        dets = decode_yolov8(out, self.conf, self.iou)
        if len(dets):
            dets.boxes = unletterbox(dets.boxes, scale, pad, img_bgr.shape[:2]).astype(np.float32)
        return dets


class UltralyticsDetector:
    def __init__(self, weights: str, imgsz: int = 640, conf: float = 0.25, iou: float = 0.5, device=None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.kw = {"imgsz": imgsz, "conf": conf, "iou": iou, "verbose": False}
        if device is not None:
            self.kw["device"] = device

    def __call__(self, img_bgr: np.ndarray) -> Detections:
        r = self.model.predict(img_bgr, **self.kw)[0].boxes
        return Detections(r.xyxy.cpu().numpy().astype(np.float32), r.conf.cpu().numpy().astype(np.float32),
                          r.cls.cpu().numpy().astype(np.int64))
