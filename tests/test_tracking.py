import numpy as np

from skytrack import ByteTracker, Detections, IoUTracker, kalman, mot_metrics
from skytrack.synthetic import SceneConfig, make_scene, run_tracker


def dets(boxes, scores):
    return Detections(np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32),
                      np.zeros(len(scores), np.int64))


def box_at(x, y, s=20):
    return [x, y, x + s, y + s]


def test_kalman_learns_constant_velocity():
    mean, cov = kalman.initiate(kalman.xyxy_to_xyah(np.array(box_at(0, 0))))
    for f in range(1, 30):
        mean, cov = kalman.predict(mean, cov)
        mean, cov = kalman.update(mean, cov, kalman.xyxy_to_xyah(np.array(box_at(5 * f, 2 * f))))
    pred, _ = kalman.predict(mean, cov)
    assert abs(pred[4] - 5) < 0.3 and abs(pred[5] - 2) < 0.3
    np.testing.assert_allclose(kalman.xyah_to_xyxy(pred), box_at(150, 60), atol=1.5)


def test_crossing_objects_keep_identities():
    """Two objects pass through each other head-on; their boxes overlap for
    several frames, so only the motion model can keep the identities apart."""
    tracker = ByteTracker()
    ids_a, ids_b = set(), set()
    for f in range(60):
        a, b = box_at(10 + 8 * f, 200), box_at(482 - 8 * f, 206)
        out = tracker.update(dets([a, b], [0.9, 0.9]))
        if abs(a[0] - b[0]) < 40:
            continue  # boxes overlap: position alone cannot label them
        for t in out:
            near_a = abs(t.box[0] - a[0]) < abs(t.box[0] - b[0])
            (ids_a if near_a else ids_b).add(t.id)
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def test_low_score_pass_bridges_occlusion():
    def run(tracker):
        emitted = []
        for f in range(30):
            score = 0.3 if 10 <= f < 20 else 0.9  # partially occluded mid-clip
            out = tracker.update(dets([box_at(10 + 5 * f, 100)], [score]))
            emitted.append([t.id for t in out])
        return emitted

    byte = run(ByteTracker(high_thresh=0.5, low_thresh=0.1))
    sort = run(ByteTracker(high_thresh=0.5, low_thresh=0.5))
    assert all(len(e) == 1 for e in byte) and len({e[0] for e in byte}) == 1
    assert all(len(e) == 0 for e in sort[10:20])  # SORT drops low-score detections


def test_lost_track_recovers_same_id_after_gap():
    tracker = ByteTracker(max_lost=10)
    seen = []
    for f in range(25):
        visible = not (8 <= f < 13)
        out = tracker.update(dets([box_at(10 + 4 * f, 50)] if visible else [], [0.9] if visible else []))
        seen += [t.id for t in out]
    assert len(set(seen)) == 1


def test_single_frame_false_positive_is_never_confirmed():
    tracker = ByteTracker()
    tracker.update(dets([box_at(0, 0)], [0.9]))
    out = tracker.update(dets([box_at(5, 0), box_at(500, 500)], [0.9, 0.9]))
    assert len(out) == 1
    out = tracker.update(dets([box_at(10, 0)], [0.9]))
    assert len(out) == 1


def test_track_deleted_after_max_lost():
    tracker = ByteTracker(max_lost=3)
    tracker.update(dets([box_at(0, 0)], [0.9]))
    for _ in range(5):
        tracker.update(dets([], []))
    assert tracker.tracks == []


def test_bytetrack_beats_iou_baseline_on_synthetic_scene():
    gt, det = make_scene(SceneConfig(frames=120, objects=15), seed=1)
    byte = mot_metrics(gt, run_tracker(ByteTracker(), det))
    iou = mot_metrics(gt, run_tracker(IoUTracker(), det))
    assert byte["idf1"] > iou["idf1"] + 0.1
    assert byte["id_switches"] < iou["id_switches"]
