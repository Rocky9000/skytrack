import cv2
import numpy as np
import pytest

from skytrack import Detections, batched_nms, iou_matrix, make_tiles, nms, sliced_predict


def test_iou_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_iou_degenerate_box_is_zero():
    assert iou_matrix([[5, 5, 5, 5]], [[5, 5, 5, 5]])[0, 0] == 0.0


def test_nms_suppresses_overlaps_keeps_distinct():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert keep.tolist() == [0, 2]


def test_batched_nms_is_class_aware():
    d = Detections(np.array([[0, 0, 10, 10], [1, 1, 11, 11]], np.float32), np.array([0.9, 0.8], np.float32),
                   np.array([0, 1]))
    assert len(batched_nms(d, 0.5)) == 2


def test_tiles_cover_image_and_respect_size():
    w, h, t = 1920, 1080, 640
    tiles = make_tiles(w, h, t, 0.2)
    covered = np.zeros((h, w), bool)
    for tile in tiles:
        assert tile.x1 - tile.x0 <= t and tile.y1 - tile.y0 <= t
        covered[tile.y0 : tile.y1, tile.x0 : tile.x1] = True
    assert covered.all()
    assert max(tl.x1 for tl in tiles) == w and max(tl.y1 for tl in tiles) == h


def test_small_image_is_one_tile():
    assert len(make_tiles(300, 200, 640)) == 1


@pytest.mark.parametrize("tile,overlap", [(0, 0.2), (640, 1.0), (640, -0.1)])
def test_bad_tile_args(tile, overlap):
    with pytest.raises(ValueError):
        make_tiles(100, 100, tile, overlap)


def blob_detector(net_size: int = 640, min_px: int = 8):
    """Simulates a CNN: sees the image only after downscaling to ``net_size``,
    and cannot detect objects smaller than ``min_px`` at that resolution."""

    def detect(img: np.ndarray) -> Detections:
        h, w = img.shape[:2]
        r = min(1.0, net_size / max(h, w))
        small = cv2.resize(img, (max(1, int(w * r)), max(1, int(h * r))), interpolation=cv2.INTER_AREA)
        _, mask = cv2.threshold(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        boxes = [(x / r, y / r, (x + bw) / r, (y + bh) / r)
                 for x, y, bw, bh, _ in stats[1:n] if bw >= min_px and bh >= min_px]
        k = len(boxes)
        return Detections(np.asarray(boxes, np.float32).reshape(-1, 4), np.full(k, 0.9, np.float32),
                          np.zeros(k, np.int64))

    return detect


def test_sliced_inference_recovers_tiny_objects():
    img = np.zeros((2160, 3840, 3), np.uint8)
    rng = np.random.default_rng(0)
    truth = []
    for _ in range(30):
        x, y = rng.integers(0, 3840 - 24), rng.integers(0, 2160 - 24)
        img[y : y + 20, x : x + 20] = 255
        truth.append([x, y, x + 20, y + 20])
    truth = np.array(truth, float)
    det = blob_detector()

    full = det(img)
    sliced = sliced_predict(img, det, tile=640, overlap=0.2)

    def recall(d):
        return float((iou_matrix(truth, d.boxes).max(axis=1) >= 0.5).mean()) if len(d) else 0.0

    assert recall(full) == 0.0  # 20 px objects shrink to ~3 px at 640: invisible
    assert recall(sliced) >= 0.9
    # overlapping tiles must not produce duplicate boxes
    assert len(sliced) <= len(truth) + 2


def test_sliced_keeps_large_objects_via_full_frame_pass():
    img = np.zeros((1280, 1280, 3), np.uint8)
    img[300:1000, 300:1000] = 255  # spans four tiles
    out = sliced_predict(img, blob_detector(min_px=4), tile=640, overlap=0.2)
    assert (iou_matrix([[300, 300, 1000, 1000]], out.boxes) >= 0.9).any()
