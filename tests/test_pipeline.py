from functools import lru_cache

import numpy as np
import pytest

from skytrack.geometry import Intrinsics, Pose
from skytrack.pipeline import InspectionPipeline
from skytrack.road_sim import SimulatedDepth, SimulatedDetector, random_potholes, render, threshold_segmenter

INTR = Intrinsics(1100, 1100, 320, 180, 640, 360)


@lru_cache(maxsize=2)
def fly(compensate=True):
    holes = random_potholes(4, 40, road_half_width=1.8, seed=3)
    det, depth = SimulatedDetector(INTR, holes, seed=3), SimulatedDepth(seed=3)
    pipe = InspectionPipeline(INTR, det, threshold_segmenter, depth, origin=(17.4, 78.4))
    if not compensate:
        pipe.tracker.compensate = lambda warp: None
    for k, north in enumerate(np.arange(-3, 43, 0.5)):
        pose = Pose(0.0, north, 10.0, pitch=np.radians(2 * np.sin(k / 5)))
        img, z = render(INTR, pose, holes)
        det.pose, depth.depth = pose, z
        pipe.process(img, pose)
    return holes, pipe.results()


def test_each_pothole_reported_once_and_measured():
    holes, found = fly()
    assert len(found) == len(holes)
    for h in holes:
        r = min(found, key=lambda r: np.hypot(r.east - h.east, r.north - h.north))
        assert np.hypot(r.east - h.east, r.north - h.north) < 0.1
        assert r.measurement.area_m2 == pytest.approx(h.area, rel=0.1)
        assert r.measurement.max_depth_m == pytest.approx(h.depth, abs=0.01)
    assert [r.priority for r in found] == sorted((r.priority for r in found), reverse=True)


def test_without_motion_compensation_tracks_break():
    holes, found = fly(compensate=False)
    assert len(found) < len(holes)  # ground moves ~55 px/frame; IoU matching alone cannot follow it


def test_geojson_shape():
    _, found = fly()
    f = found[0].to_feature()
    assert f["geometry"]["type"] == "Point" and len(f["geometry"]["coordinates"]) == 2
    assert {"grade", "area_m2", "volume_m3", "max_depth_m"} <= set(f["properties"])
