"""Run the inspection pipeline on a flight video with its telemetry log.

    skytrack inspect flight.mp4 telemetry.csv --detector yolov8s-pothole.onnx \
        --segmenter deeplabv3p.onnx --depth depth_anything_v2_metric.onnx \
        --focal-mm 4.5 --sensor-width-mm 6.17 --out report

``telemetry.csv`` needs one row per video frame with the columns
``frame,lat,lon,alt_agl,yaw_deg,pitch_deg,roll_deg`` (yaw clockwise from north).
Writes ``report.geojson`` (open it in QGIS or geojson.io) and ``report.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .geometry import Intrinsics, Pose, geodetic_to_enu
from .pipeline import InspectionPipeline, PipelineConfig


def read_telemetry(path: Path) -> dict[int, dict[str, float]]:
    with open(path, newline="") as f:
        return {int(r["frame"]): {k: float(v) for k, v in r.items() if k != "frame"} for r in csv.DictReader(f)}


def inspect(a: argparse.Namespace) -> None:
    import cv2

    from .detectors import OnnxDetector
    from .models import OnnxDepth, OnnxSegmenter

    tel = read_telemetry(a.telemetry)
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise OSError(f"cannot open {a.video}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    intr = Intrinsics.from_sensor(a.focal_mm, a.sensor_width_mm, w, h)
    first = tel[min(tel)]
    origin = (first["lat"], first["lon"])
    pipe = InspectionPipeline(
        intr, OnnxDetector(a.detector, a.imgsz, conf=0.1), OnnxSegmenter(a.segmenter), OnnxDepth(a.depth), origin,
        PipelineConfig(tile=a.tile, depth_kind=a.depth_kind, road_class=a.road_class),
    )
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in tel:
            t = tel[i]
            e, n = geodetic_to_enu(t["lat"], t["lon"], *origin)
            pipe.process(frame, Pose(e, n, t["alt_agl"], *np.radians([t["yaw_deg"], t["pitch_deg"], t["roll_deg"]])))
        i += 1
    cap.release()

    results = pipe.results()
    a.out.with_suffix(".geojson").write_text(json.dumps(pipe.geojson(), indent=1))
    with open(a.out.with_suffix(".csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "lat", "lon", "grade", "priority", "area_m2", "length_m", "width_m", "max_depth_m",
                     "volume_m3", "views"])
        for r in results:
            m = r.measurement
            wr.writerow([r.track_id, f"{r.lat:.7f}", f"{r.lon:.7f}", r.grade, r.priority,
                         f"{m.area_m2:.3f}", f"{m.length_m:.2f}", f"{m.width_m:.2f}", f"{m.max_depth_m:.3f}",
                         f"{m.volume_m3:.4f}", r.views])
    print(f"{i} frames, {len(results)} potholes -> {a.out.with_suffix('.geojson')}")


def main() -> None:
    p = argparse.ArgumentParser(prog="skytrack", description="Drone pothole inspection")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inspect", help=__doc__.splitlines()[0])
    s.add_argument("video", type=Path)
    s.add_argument("telemetry", type=Path)
    s.add_argument("--detector", required=True, help="YOLOv8 ONNX trained on RDD2022 (pothole = class 0)")
    s.add_argument("--segmenter", required=True, help="DeepLabv3+ ONNX, pothole = class 1")
    s.add_argument("--depth", required=True, help="Depth Anything V2 ONNX")
    s.add_argument("--depth-kind", choices=["metric", "relative"], default="metric")
    s.add_argument("--focal-mm", type=float, required=True)
    s.add_argument("--sensor-width-mm", type=float, required=True)
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--tile", type=int, default=None, help="sliced inference tile size, for high-altitude flights")
    s.add_argument("--road-class", default="local", choices=["arterial", "collector", "local"])
    s.add_argument("--out", type=Path, default=Path("report"))
    a = p.parse_args()
    inspect(a)


if __name__ == "__main__":
    main()
