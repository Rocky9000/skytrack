"""Frame-by-frame pothole inspection: detect, track, measure, geotag, grade.

Per frame
    1. Detect potholes (YOLO, optionally with sliced inference for high altitude).
    2. Track them with ByteTrack so a pothole seen in 20 consecutive frames is
       counted once. Low-score detections keep tracks alive through glare.
    3. When a track is fully in view and closer to the image centre than any
       earlier view (least perspective distortion), segment the crop, estimate
       depth, and measure it. This caps the expensive models at a few calls per
       pothole instead of one per frame.
    4. Geotag the mask centroid by intersecting its pixel ray with the road.

At the end, records closer than ``merge_radius_m`` are merged, which catches a
pothole that left the frame and was picked up again under a new track id.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .boxes import Detections
from .geometry import Intrinsics, Pose, enu_to_geodetic, ground_hit, project
from .measure import PotholeMeasurement, align_depth, measure_pothole
from .severity import priority, rule_grade
from .tiling import sliced_predict
from .tracking import ByteTracker


@dataclass
class PotholeRecord:
    track_id: int
    frame: int
    east: float
    north: float
    lat: float
    lon: float
    measurement: PotholeMeasurement
    grade: int
    priority: float
    views: int = 1
    center_dist: float = field(default=np.inf, repr=False)

    def to_feature(self) -> dict:
        return {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(self.lon, 7), round(self.lat, 7)]},
            "properties": {"id": self.track_id, "frame": self.frame, "grade": self.grade,
                           "priority": self.priority, "views": self.views, **self.measurement.to_dict()},
        }


@dataclass
class PipelineConfig:
    pad_frac: float = 0.6        # crop padding around the box, as a fraction of its size; needs road around it
    edge_margin_px: int = 8      # skip views where the box touches the frame edge
    max_measurements: int = 3    # per track
    merge_radius_m: float = 1.0
    min_mask_px: int = 30
    tile: int | None = None      # sliced inference tile size; None = full frame only
    road_class: str = "local"
    depth_kind: str = "metric"   # "metric" or "relative"; see measure.align_depth


class InspectionPipeline:
    def __init__(self, intr: Intrinsics, detector: Callable[[np.ndarray], Detections],
                 segmenter: Callable[[np.ndarray], np.ndarray], depth: Callable[[np.ndarray], np.ndarray],
                 origin: tuple[float, float], cfg: PipelineConfig | None = None, tracker: ByteTracker | None = None):
        self.intr, self.detector, self.segmenter, self.depth = intr, detector, segmenter, depth
        self.lat0, self.lon0 = origin
        self.cfg = cfg or PipelineConfig()
        self.tracker = tracker or ByteTracker(high_thresh=0.5, low_thresh=0.1, new_track_thresh=0.5, max_lost=10)
        self.records: dict[int, PotholeRecord] = {}
        self._measured: dict[int, int] = {}
        self.frame = 0
        self._prev_pose: Pose | None = None

    def _crop(self, box: np.ndarray) -> tuple[int, int, int, int] | None:
        c, W, H = self.cfg, self.intr.width, self.intr.height
        x0, y0, x1, y1 = box
        if x0 < c.edge_margin_px or y0 < c.edge_margin_px or x1 > W - c.edge_margin_px or y1 > H - c.edge_margin_px:
            return None
        pw, ph = (x1 - x0) * c.pad_frac + 10, (y1 - y0) * c.pad_frac + 10
        return int(max(0, x0 - pw)), int(max(0, y0 - ph)), int(min(W, x1 + pw)), int(min(H, y1 + ph))

    def _measure(self, frame: np.ndarray, pose: Pose, crop: tuple[int, int, int, int]):
        x0, y0, x1, y1 = crop
        img = frame[y0:y1, x0:x1]
        mask = self.segmenter(img) > 0.5
        if mask.sum() < self.cfg.min_mask_px:
            return None
        ci = self.intr.crop(x0, y0, x1 - x0, y1 - y0)
        v, u = np.mgrid[y0:y1, x0:x1]
        _, expected = ground_hit(self.intr, pose, u + 0.5, v + 0.5)  # depth of a flat road at each pixel
        road = ~mask
        if hasattr(self.depth, "offset"):  # simulation hook: SimulatedDepth needs to know where the crop is
            self.depth.offset = (y0, x0)
        try:
            metric = align_depth(self.depth(img), expected, road, kind=self.cfg.depth_kind)
            m = measure_pothole(mask, metric, ci)
        except ValueError:  # includes DepthAlignmentError
            return None
        mv, mu = np.nonzero(mask)
        ground, _ = ground_hit(self.intr, pose, x0 + mu.mean() + 0.5, y0 + mv.mean() + 0.5)
        return m, ground

    def process(self, frame: np.ndarray, pose: Pose) -> list[int]:
        """Process one frame. Returns the ids of tracks measured in this frame."""
        self.frame += 1
        dets = sliced_predict(frame, self.detector, self.cfg.tile) if self.cfg.tile else self.detector(frame)
        if self._prev_pose is not None:
            prev = self._prev_pose

            def warp(px: np.ndarray) -> np.ndarray:  # previous pixel -> road point -> current pixel
                return project(self.intr, pose, ground_hit(self.intr, prev, px[:, 0], px[:, 1])[0])[0]

            self.tracker.compensate(warp)
        self._prev_pose = pose
        measured = []
        centre = np.array([self.intr.cx, self.intr.cy])
        for t in self.tracker.update(dets):
            dist = float(np.linalg.norm((t.box[:2] + t.box[2:]) / 2 - centre))
            rec = self.records.get(t.id)
            if self._measured.get(t.id, 0) >= self.cfg.max_measurements or (rec and dist >= rec.center_dist - 20):
                continue
            crop = self._crop(t.box)
            if crop is None:
                continue
            out = self._measure(frame, pose, crop)
            if out is None:
                continue
            m, ground = out
            lat, lon = enu_to_geodetic(ground[0], ground[1], self.lat0, self.lon0)
            grade = rule_grade(m)
            views = rec.views + 1 if rec else 1
            self.records[t.id] = PotholeRecord(t.id, self.frame, float(ground[0]), float(ground[1]), float(lat),
                                               float(lon), m, grade, priority(grade, self.cfg.road_class), views, dist)
            self._measured[t.id] = self._measured.get(t.id, 0) + 1
            measured.append(t.id)
        return measured

    def results(self) -> list[PotholeRecord]:
        """De-duplicated records, most urgent first."""
        merged: list[PotholeRecord] = []
        for r in sorted(self.records.values(), key=lambda r: r.center_dist):
            near = next((m for m in merged if np.hypot(m.east - r.east, m.north - r.north) < self.cfg.merge_radius_m),
                        None)
            if near is None:
                merged.append(r)
            else:
                near.views += r.views
        return sorted(merged, key=lambda r: (-r.priority, -r.measurement.volume_m3))

    def geojson(self) -> dict:
        return {"type": "FeatureCollection", "features": [r.to_feature() for r in self.results()]}
