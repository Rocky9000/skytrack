"""End-to-end check of the measurement chain on synthetic flights.

    python scripts/simulate_flight.py --flights 3

Renders nadir flights over a road with potholes of known size, runs the full
pipeline (tracking, motion compensation, segmentation, depth alignment,
measurement, geotagging, de-duplication) with stand-in models, and compares the
report with ground truth. Telemetry is perturbed the way real flight logs are:
a GPS bias that stays fixed during a flight plus jitter, altitude error, and
attitude error. The rendered frames always use the true pose.

What this does *not* measure: the accuracy of the trained detector, segmenter
or depth model on real roads. Those come from RDD2022 validation and field data.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import linear_sum_assignment

from skytrack.geometry import Intrinsics, Pose
from skytrack.measure import PotholeMeasurement
from skytrack.pipeline import InspectionPipeline
from skytrack.road_sim import SimulatedDepth, SimulatedDetector, random_potholes, render, threshold_segmenter
from skytrack.severity import rule_grade

MATCH_RADIUS_M = 1.0

TELEMETRY = {
    # name: (GPS bias sigma m, GPS jitter m, altitude sigma m, attitude sigma deg)
    "RTK GPS": (0.02, 0.01, 0.03, 0.2),
    "Standard GPS": (1.0, 0.15, 0.3, 0.5),
}


def fly(seed: int, telemetry: tuple[float, float, float, float], n_holes: int = 12, length: float = 120.0,
        alt: float = 10.0, speed: float = 5.0, fps: float = 10.0) -> dict:
    rng = np.random.default_rng(seed)
    intr = Intrinsics(1100, 1100, 480, 270, 960, 540)  # ~9.1 mm/px at 10 m
    holes = random_potholes(n_holes, length, seed=seed)
    det = SimulatedDetector(intr, holes, seed=seed)
    depth = SimulatedDepth(seed=seed)
    pipe = InspectionPipeline(intr, det, threshold_segmenter, depth, origin=(17.4474, 78.3762))
    bias_sigma, jitter, alt_sigma, att_sigma = telemetry
    bias = rng.normal(0, bias_sigma, 2)
    for k, north in enumerate(np.arange(-5, length + 5, speed / fps)):
        true = Pose(0.3 * np.sin(k / 15), north, alt + 0.3 * np.sin(k / 23), yaw=np.radians(2 * np.sin(k / 30)),
                    pitch=np.radians(3 * np.sin(k / 7)), roll=np.radians(3 * np.cos(k / 9)))
        img, z = render(intr, true, holes)
        e, n = np.array([true.east, true.north]) + bias + rng.normal(0, jitter, 2)
        a = np.radians(rng.normal(0, att_sigma, 3))
        logged = Pose(e, n, true.alt + rng.normal(0, alt_sigma), true.yaw + a[0], true.pitch + a[1], true.roll + a[2])
        det.pose, depth.depth = true, z
        pipe.process(img, logged)

    found = pipe.results()
    # Associate after removing this flight's GPS offset, so a large offset is reported as geolocation
    # error rather than as missed potholes. The geolocation error below still includes it.
    dist = np.array([[np.hypot(h.east - r.east + bias[0], h.north - r.north + bias[1]) for r in found] for h in holes])
    dist = dist.reshape(len(holes), -1)
    gi, ri = linear_sum_assignment(dist) if dist.size else ([], [])
    matched = {int(g): found[r] for g, r in zip(gi, ri, strict=True) if dist[g, r] < MATCH_RADIUS_M}
    rows = []
    for i, r in matched.items():
        h, m = holes[i], r.measurement
        truth = PotholeMeasurement(h.area, 2 * h.a, 2 * h.b, h.depth, 0, h.volume, 0)
        rows.append((np.hypot(h.east - r.east, h.north - r.north), abs(m.area_m2 / h.area - 1),
                     abs(m.max_depth_m - h.depth), abs(m.volume_m3 / h.volume - 1), r.grade == rule_grade(truth)))
    missed = [i for i in range(len(holes)) if i not in matched]
    return {"gt": len(holes), "found": len(matched), "extra": len(found) - len(matched), "rows": rows,
            "missed": [holes[i] for i in missed]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flights", type=int, default=3)
    a = p.parse_args()
    print(f"{a.flights} flights x 12 potholes, 120 m of road at 10 m altitude, 5 m/s, 10 fps\n")
    print("| Telemetry | Recall | Extra records | Geo error mean / p95 | Area error | Max-depth error | "
          "Volume error | Grade match |")
    print("|---|---|---|---|---|---|---|---|")
    for name, tel in TELEMETRY.items():
        runs = [fly(s, tel) for s in range(a.flights)]
        rows = np.array([r for run in runs for r in run["rows"]], dtype=float)
        gt, found = sum(r["gt"] for r in runs), sum(r["found"] for r in runs)
        print(f"| {name} | {found / gt:.3f} | {sum(r['extra'] for r in runs)} | "
              f"{rows[:, 0].mean():.2f} / {np.percentile(rows[:, 0], 95):.2f} m | {100 * rows[:, 1].mean():.1f} % | "
              f"{1000 * rows[:, 2].mean():.1f} mm | {100 * rows[:, 3].mean():.1f} % | {rows[:, 4].mean():.2f} |")


if __name__ == "__main__":
    main()
