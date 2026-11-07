"""Closed-loop MPPI wire-avoidance scenarios.

    python scripts/benchmark_planner.py

Each scenario flies 40 m north at 10 m altitude with perfect state feedback.
Wires are placed across the path the way they appear on Indian roads: single
cables at flight height, a pair stacked vertically, and a sagging cable
approximated by segments. Reports success, closest approach, and planning
rate on this machine.
"""

from __future__ import annotations

import time

import numpy as np

from skytrack.planner import MPPIConfig, MPPIPlanner, simulate


def sag(y: float, z_mid: float, span: float = 30.0, drop: float = 1.5, pieces: int = 6) -> np.ndarray:
    x = np.linspace(-span / 2, span / 2, pieces + 1)
    z = z_mid + drop * (2 * x / span) ** 2
    pts = np.stack([x, np.full_like(x, y), z], axis=1)
    return np.stack([pts[:-1], pts[1:]], axis=1)


def line(y: float, z: float, half: float = 25.0) -> np.ndarray:
    return np.array([[[-half, y, z], [half, y, z]]])


SCENARIOS = {
    "no wires": None,
    "one cable at flight height": line(20, 10),
    "cable 0.5 m below flight height": line(20, 9.5),
    "two stacked cables 1 m apart (9.5 m, 10.5 m)": np.concatenate([line(20, 9.5), line(20, 10.5)]),
    "two stacked cables 2 m apart (9 m, 11 m)": np.concatenate([line(20, 9), line(20, 11)]),
    "sagging cable, lowest point at 10 m": sag(20, 10),
    "two cables 6 m apart along the path": np.concatenate([line(15, 10), line(21, 10.5)]),
}


def main() -> None:
    cfg = MPPIConfig()
    print(f"MPPI: {cfg.samples} samples x {cfg.horizon} steps ({cfg.horizon * cfg.dt:.1f} s), "
          f"safety radius {cfg.safety_radius} m\n")
    print("| Scenario | Reached goal | Closest approach to a wire | Min altitude | Plan rate |")
    print("|---|---|---|---|---|")
    for name, wires in SCENARIOS.items():
        planner = MPPIPlanner(MPPIConfig(seed=0))
        t0 = time.perf_counter()
        r = simulate(planner, [0, 0, 10], [0, 40, 10], wires, steps=600)
        hz = r["steps"] / (time.perf_counter() - t0)
        clearance = "n/a" if np.isinf(r["min_clearance_m"]) else f"{r['min_clearance_m']:.2f} m"
        print(f"| {name} | {'yes' if r['reached'] else 'no'} | {clearance} | {r['min_alt_m']:.1f} m | {hz:.0f} Hz |")


if __name__ == "__main__":
    main()
