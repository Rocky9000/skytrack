"""Severity grading and repair priority.

Two graders share one interface:

* ``rule_grade``: deterministic 1-5 grade from measured depth and area. Depth
  bands follow the FHWA Distress Identification Manual (low < 25 mm,
  moderate 25-50 mm, high > 50 mm), split further so the scale has five steps.
  It needs no training data, so it is also how the first severity labels for
  the learned model can be bootstrapped and audited.
* ``OnnxSeverityClassifier``: an image classifier (EfficientNet-B4 in the
  reference design) run server-side on the full-resolution crop. It sees
  texture the depth map misses, such as exposed base layer or water.
"""

from __future__ import annotations

import numpy as np

from .measure import PotholeMeasurement

DEPTH_BANDS_M = (0.025, 0.05, 0.075, 0.10)
LARGE_AREA_M2 = 1.0

ROAD_WEIGHT = {"arterial": 1.5, "collector": 1.2, "local": 1.0}


def rule_grade(m: PotholeMeasurement) -> int:
    grade = 1 + int(np.searchsorted(DEPTH_BANDS_M, m.max_depth_m, side="right"))
    if m.area_m2 >= LARGE_AREA_M2:
        grade += 1
    return min(grade, 5)


def priority(grade: int, road_class: str = "local") -> float:
    """Higher is more urgent. Grade dominates; busier roads break ties upward."""
    return round(grade * ROAD_WEIGHT.get(road_class, 1.0), 2)


class OnnxSeverityClassifier:
    """Five-class severity classifier exported to ONNX (input NCHW RGB, ImageNet normalisation)."""

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, path: str | None = None, imgsz: int = 380, session=None):
        if session is None:
            import onnxruntime as ort

            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.session, self.imgsz = session, imgsz
        self.input = session.get_inputs()[0].name

    def __call__(self, crop_bgr: np.ndarray) -> tuple[int, np.ndarray]:
        import cv2

        x = cv2.resize(crop_bgr, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)[..., ::-1]
        x = ((x.astype(np.float32) / 255.0 - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        logits = np.asarray(self.session.run(None, {self.input: np.ascontiguousarray(x)})[0])[0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return int(p.argmax()) + 1, p
