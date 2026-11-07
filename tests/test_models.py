import numpy as np

from skytrack.models import OnnxDepth, OnnxSegmenter
from skytrack.severity import OnnxSeverityClassifier


class FakeSession:
    def __init__(self, fn):
        self.fn, self.seen = fn, None

    def get_inputs(self):
        return [type("I", (), {"name": "input"})()]

    def run(self, _, feed):
        self.seen = feed["input"]
        return [self.fn(self.seen)]


def test_segmenter_resizes_back_and_softmaxes():
    def logits(x):
        out = np.zeros((1, 2) + x.shape[2:], np.float32)
        out[0, 1, :, : x.shape[3] // 2] = 10.0  # left half is pothole
        out[0, 0, :, x.shape[3] // 2 :] = 10.0  # right half is road
        return out

    sess = FakeSession(logits)
    prob = OnnxSegmenter(session=sess, imgsz=64)(np.zeros((30, 50, 3), np.uint8))
    assert sess.seen.shape == (1, 3, 64, 64)
    assert prob.shape == (30, 50)
    assert prob[:, :20].min() > 0.99 and prob[:, 30:].max() < 0.01


def test_depth_input_is_patch_aligned():
    sess = FakeSession(lambda x: np.ones((1,) + x.shape[2:], np.float32))
    out = OnnxDepth(session=sess, imgsz=70)(np.zeros((40, 60, 3), np.uint8))
    assert sess.seen.shape == (1, 3, 70, 70) and out.shape == (40, 60)


def test_severity_classifier_returns_grade_1_to_5():
    sess = FakeSession(lambda x: np.array([[0.1, 0.2, 3.0, 0.1, 0.0]], np.float32))
    grade, p = OnnxSeverityClassifier(session=sess, imgsz=32)(np.zeros((50, 50, 3), np.uint8))
    assert grade == 3 and abs(p.sum() - 1) < 1e-6
