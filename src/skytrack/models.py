"""ONNX Runtime wrappers for the per-detection models.

Both run only on crops around confirmed detections, never on every frame:

* ``OnnxSegmenter``: semantic segmentation (DeepLabv3+ in the reference design)
  that turns a YOLO box into a pixel-accurate pothole outline.
* ``OnnxDepth``: monocular relative depth (Depth Anything V2 small). Its output
  is relative inverse depth; ``measure.align_depth`` makes it metric.

Pass ``session=`` to inject any object with ONNX Runtime's ``run`` /
``get_inputs`` interface (used in the tests).
"""

from __future__ import annotations

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _session(path: str | None, session, providers):
    if session is not None:
        return session
    import onnxruntime as ort

    return ort.InferenceSession(path, providers=providers or ["CPUExecutionProvider"])


def _to_input(img_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    x = cv2.resize(img_bgr, size, interpolation=cv2.INTER_LINEAR)[..., ::-1]
    x = ((x.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x)


class OnnxSegmenter:
    def __init__(self, path: str | None = None, imgsz: int = 512, pothole_class: int = 1, session=None,
                 providers=None):
        self.session = _session(path, session, providers)
        self.input = self.session.get_inputs()[0].name
        self.imgsz, self.cls = imgsz, pothole_class

    def __call__(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Pothole probability map, same size as the crop."""
        h, w = crop_bgr.shape[:2]
        logits = np.asarray(self.session.run(None, {self.input: _to_input(crop_bgr, (self.imgsz, self.imgsz))})[0])[0]
        e = np.exp(logits - logits.max(axis=0, keepdims=True))
        prob = (e[self.cls] / e.sum(axis=0)).astype(np.float32)
        return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


class OnnxDepth:
    def __init__(self, path: str | None = None, imgsz: int = 518, session=None, providers=None):
        if imgsz % 14:
            raise ValueError("Depth Anything uses 14 px patches; imgsz must be a multiple of 14")
        self.session = _session(path, session, providers)
        self.input = self.session.get_inputs()[0].name
        self.imgsz = imgsz

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        """Relative inverse depth, same size as the input (larger = closer)."""
        h, w = img_bgr.shape[:2]
        out = np.asarray(self.session.run(None, {self.input: _to_input(img_bgr, (self.imgsz, self.imgsz))})[0])
        out = out.reshape(out.shape[-2:]).astype(np.float32)
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
