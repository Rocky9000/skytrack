import numpy as np
import pytest

from skytrack import Detections, detection_map, mot_metrics
from skytrack.detectors import decode_yolov8, letterbox, unletterbox
from skytrack.metrics import average_precision
from skytrack.visdrone import parse_annotation, to_yolo_lines


def D(boxes, scores, classes):
    return Detections(np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32),
                      np.asarray(classes, np.int64))


GT = [(np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float), np.array([0, 1]))]


def test_ap_perfect_and_empty():
    assert average_precision(np.array([0.5, 1.0]), np.array([1.0, 1.0])) == pytest.approx(1.0)
    assert average_precision(np.array([]), np.array([])) == 0.0


def test_map_perfect_predictions():
    m = detection_map([D([[0, 0, 10, 10], [20, 20, 30, 30]], [0.9, 0.8], [0, 1])], GT, num_classes=2)
    assert m["map50"] == pytest.approx(1.0) and m["map50_95"] == pytest.approx(1.0)


def test_map_penalises_high_scoring_false_positive():
    preds = [D([[50, 50, 60, 60], [0, 0, 10, 10], [20, 20, 30, 30]], [0.99, 0.9, 0.8], [0, 0, 1])]
    m = detection_map(preds, GT, num_classes=2)
    assert m["ap50_per_class"][1] == pytest.approx(1.0)
    assert m["ap50_per_class"][0] == pytest.approx(0.5, abs=0.01)


def test_map_loose_box_passes_at_50_not_at_95():
    m = detection_map([D([[0, 0, 12, 10], [20, 20, 30, 30]], [0.9, 0.9], [0, 1])], GT, num_classes=2)
    assert m["map50"] == pytest.approx(1.0) and m["map50_95"] < 1.0


def test_mot_perfect_tracking():
    gt = [{1: np.array(box), 2: np.array([50, 50, 60, 60])} for box in ([0, 0, 10, 10], [1, 0, 11, 10])]
    m = mot_metrics(gt, [{7: g[1], 8: g[2]} for g in gt])
    assert m["mota"] == 1.0 and m["idf1"] == 1.0 and m["id_switches"] == 0


def test_mot_counts_identity_switch():
    a, b = np.array([0, 0, 10, 10]), np.array([50, 50, 60, 60])
    gt = [{1: a, 2: b}] * 4
    hyp = [{1: a, 2: b}, {1: a, 2: b}, {1: b, 2: a}, {1: b, 2: a}]
    m = mot_metrics(gt, hyp)
    assert m["id_switches"] == 2
    assert m["idf1"] == pytest.approx(0.5)


def test_mot_fp_fn():
    gt = [{1: np.array([0, 0, 10, 10])}, {1: np.array([0, 0, 10, 10])}]
    hyp = [{}, {5: np.array([100, 100, 110, 110])}]
    m = mot_metrics(gt, hyp)
    assert (m["false_negatives"], m["false_positives"]) == (2, 1)
    assert m["mota"] == pytest.approx(1 - 3 / 2)


def test_visdrone_parsing_drops_ignored_and_others():
    text = "10,20,30,40,1,4,0,0\n0,0,50,50,0,0,0,0\n5,5,10,10,1,11,0,0\n1,1,0,5,1,1,0,0\n7,8,9,10,1,1,0,1,\n"
    boxes, classes = parse_annotation(text)
    assert classes.tolist() == [3, 0]
    assert boxes.tolist() == [[10, 20, 30, 40], [7, 8, 9, 10]]


def test_yolo_lines_are_normalised_and_clipped():
    lines = to_yolo_lines(np.array([[90.0, 40, 20, 20]]), np.array([2]), 100, 50)
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "2"
    assert float(cx) == pytest.approx(0.95) and float(w) == pytest.approx(0.1)
    assert float(cy) == pytest.approx(0.9) and float(h) == pytest.approx(0.2)


def test_letterbox_roundtrip():
    img = np.zeros((480, 1280, 3), np.uint8)
    out, r, pad = letterbox(img, 640)
    assert out.shape == (640, 640, 3) and r == pytest.approx(0.5) and pad == (0, 200)
    net_box = np.array([[100.0, 250, 200, 300]])
    np.testing.assert_allclose(unletterbox(net_box, r, pad, img.shape[:2]), [[200, 100, 400, 200]])


def test_decode_yolov8_head():
    nc, anchors = 3, 6
    raw = np.zeros((1, 4 + nc, anchors), np.float32)
    raw[0, :4, 0] = [100, 100, 20, 20]
    raw[0, 4 + 2, 0] = 0.9   # class 2
    raw[0, :4, 1] = [101, 101, 20, 20]
    raw[0, 4 + 2, 1] = 0.6   # duplicate, suppressed by NMS
    raw[0, :4, 2] = [300, 300, 40, 40]
    raw[0, 4 + 0, 2] = 0.1   # below threshold
    d = decode_yolov8(raw, conf_thr=0.25)
    assert len(d) == 1 and d.classes[0] == 2
    np.testing.assert_allclose(d.boxes[0], [90, 90, 110, 110])
