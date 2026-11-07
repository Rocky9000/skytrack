import numpy as np
import pytest

from skytrack import Detections, detection_map, mot_metrics
from skytrack.detectors import decode_yolov8, letterbox, unletterbox
from skytrack.metrics import average_precision
from skytrack.rdd2022 import parse_voc, to_yolo


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


RDD_XML = """<annotation><size><width>600</width><height>600</height><depth>3</depth></size>
<object><name>D40</name><bndbox><xmin>100</xmin><ymin>200</ymin><xmax>160</xmax><ymax>260</ymax></bndbox></object>
<object><name>D00</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>20</xmax><ymax>300</ymax></bndbox></object>
<object><name>Repair</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object>
<object><name>D40</name><bndbox><xmin>590</xmin><ymin>590</ymin><xmax>640</xmax><ymax>640</ymax></bndbox></object>
</annotation>"""


def test_rdd2022_parsing_keeps_known_classes():
    w, h, objs = parse_voc(RDD_XML)
    assert (w, h) == (600, 600)
    assert [o[0] for o in objs] == [3, 0, 3]  # Repair dropped; D40 is class 3 of four
    _, _, potholes = parse_voc(RDD_XML, classes=["D40"])
    assert [o[0] for o in potholes] == [0, 0]


def test_yolo_lines_are_normalised_and_clipped():
    w, h, objs = parse_voc(RDD_XML, classes=["D40"])
    first, edge = to_yolo(w, h, objs)
    cls, cx, cy, bw, bh = first.split()
    assert cls == "0"
    assert float(cx) == pytest.approx(130 / 600, abs=1e-6) and float(bw) == pytest.approx(60 / 600, abs=1e-6)
    _, cx, cy, bw, bh = (float(x) for x in edge.split())
    assert cx + bw / 2 <= 1.0 + 1e-6 and cy + bh / 2 <= 1.0 + 1e-6  # box running off the image is clipped


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
