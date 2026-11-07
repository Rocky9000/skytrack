import numpy as np
import pytest

from skytrack.geometry import Intrinsics, Pose, ground_hit
from skytrack.measure import DepthAlignmentError, align_depth, fit_plane, measure_pothole
from skytrack.road_sim import Pothole, render
from skytrack.severity import rule_grade

INTR = Intrinsics(1100, 1100, 300, 300, 600, 600)


def scene(pose: Pose, hole: Pothole):
    img, z = render(INTR, pose, [hole])
    v, u = np.mgrid[0:600, 0:600]
    pts, _ = ground_hit(INTR, pose, u + 0.5, v + 0.5)
    mask = hole.depth_at(pts[..., 0], pts[..., 1]) > 0
    return mask, z.astype(np.float64)


@pytest.mark.parametrize("pose", [Pose(0, 0, 10), Pose(0.4, -0.3, 8, yaw=0.6, pitch=0.12, roll=-0.08)])
def test_measures_known_pothole(pose):
    hole = Pothole(0.1, 0.05, 0.5, 0.3, 0.4, 0.08)
    mask, z = scene(pose, hole)
    m = measure_pothole(mask, z, INTR)
    assert m.area_m2 == pytest.approx(hole.area, rel=0.02)
    assert m.volume_m3 == pytest.approx(hole.volume, rel=0.03)
    assert m.max_depth_m == pytest.approx(hole.depth, abs=0.004)
    assert m.length_m == pytest.approx(1.0, rel=0.03) and m.width_m == pytest.approx(0.6, rel=0.04)
    assert m.plane_rmse_m < 1e-3


def test_metric_alignment_removes_bias_and_keeps_shape():
    pose = Pose(0, 0, 10)
    hole = Pothole(0, 0, 0.4, 0.4, 0, 0.06)
    mask, z = scene(pose, hole)
    _, expected = ground_hit(INTR, pose, *np.mgrid[0:600, 0:600][::-1] + 0.5)
    aligned = align_depth(z + 0.7, expected, ~mask, kind="metric")
    assert measure_pothole(mask, aligned, INTR).max_depth_m == pytest.approx(0.06, abs=0.003)


def test_relative_alignment_needs_depth_variation():
    hole = Pothole(0, 0, 0.4, 0.4, 0, 0.06)
    for pose, ok in ((Pose(0, 0, 10, pitch=0.35), True), (Pose(0, 0, 10), False)):
        mask, z = scene(pose, hole)
        _, expected = ground_hit(INTR, pose, *np.mgrid[0:600, 0:600][::-1] + 0.5)
        rel = 20.0 / z + 0.05  # inverse depth, arbitrary scale and shift
        if ok:
            aligned = align_depth(rel, expected, ~mask, kind="relative")
            np.testing.assert_allclose(aligned, z, rtol=1e-6)
        else:
            with pytest.raises(DepthAlignmentError):
                align_depth(rel, expected, ~mask, kind="relative")


def test_plane_fit_ignores_outliers():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1, 1, (400, 2))
    pts = np.c_[xy, 5 + 0.1 * xy[:, 0]]
    pts[:40, 2] += rng.uniform(0.2, 1.0, 40)  # a curb or a parked bike
    n, d = fit_plane(pts)
    ref = np.array([-0.1, 0, 1]) / np.linalg.norm([-0.1, 0, 1])
    assert abs(n @ ref) == pytest.approx(1.0, abs=1e-6)


def test_severity_grades_follow_depth_bands():
    from skytrack.measure import PotholeMeasurement

    def m(depth, area=0.3):
        return PotholeMeasurement(area, 0.6, 0.5, depth, depth / 2, area * depth / 2, 0.0)

    assert [rule_grade(m(d)) for d in (0.01, 0.03, 0.06, 0.09, 0.15)] == [1, 2, 3, 4, 5]
    assert rule_grade(m(0.03, area=1.5)) == 3  # large footprint bumps one grade
    assert rule_grade(m(0.2, area=2.0)) == 5
