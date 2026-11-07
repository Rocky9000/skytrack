import numpy as np
import pytest

from skytrack.planner import MPPIConfig, MPPIPlanner, point_segment_distance, simulate, velocity_setpoint_ned


def test_point_segment_distance():
    a, b = np.array([[0.0, 0, 0]]), np.array([[10.0, 0, 0]])
    p = np.array([[5.0, 3, 4], [-3.0, 0, 4], [12.0, 0, 0]])
    np.testing.assert_allclose(point_segment_distance(p, a, b)[:, 0], [5.0, 5.0, 2.0])


def test_reaches_goal_without_obstacles():
    r = simulate(MPPIPlanner(MPPIConfig(seed=1)), [0, 0, 10], [5, 25, 10])
    assert r["reached"] and r["min_alt_m"] > 9.0


@pytest.mark.parametrize("wire_z", [9.5, 10.0, 10.5])
def test_avoids_cable_across_path(wire_z):
    cfg = MPPIConfig(seed=0)
    wire = np.array([[[-25.0, 15, wire_z], [25.0, 15, wire_z]]])
    straight = point_segment_distance(np.array([[0.0, 15, 10]]), wire[:, 0], wire[:, 1]).min()
    assert straight < cfg.safety_radius  # flying straight would hit it
    r = simulate(MPPIPlanner(cfg), [0, 0, 10], [0, 30, 10], wire, steps=600)
    assert r["reached"]
    assert r["min_clearance_m"] >= cfg.safety_radius - 0.05


def test_respects_minimum_altitude():
    cfg = MPPIConfig(seed=0, min_alt=2.0)
    wire = np.array([[[-25.0, 10, 4.0], [25.0, 10, 4.0]]])  # low cable: going under would hit the floor
    r = simulate(MPPIPlanner(cfg), [0, 0, 3.5], [0, 20, 3.5], wire, steps=600)
    assert r["min_alt_m"] >= cfg.min_alt - 0.1
    assert r["min_clearance_m"] >= cfg.safety_radius - 0.05


def test_ned_setpoint():
    sp = velocity_setpoint_ned(np.array([1.0, 2.0, 0.5]))  # east, north, up
    assert (sp["vx"], sp["vy"], sp["vz"]) == (2.0, 1.0, -0.5)
    assert sp["type_mask"] == 3527
