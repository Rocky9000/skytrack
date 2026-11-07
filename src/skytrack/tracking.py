"""Multi-object trackers, implemented from scratch.

``ByteTracker`` follows ByteTrack (Zhang et al., 2022): Kalman motion model,
Hungarian matching on IoU, and a *second* association pass that uses
low-confidence detections to keep tracks alive through partial occlusion
instead of discarding them. ``IoUTracker`` is a greedy, motion-free
baseline used to show what each of those pieces buys.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import kalman
from .boxes import Detections, iou_matrix


class TrackState(Enum):
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    REMOVED = 3


@dataclass
class Track:
    id: int
    mean: np.ndarray
    cov: np.ndarray
    score: float
    cls: int
    start_frame: int
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    frames_lost: int = 0
    history: list = field(default_factory=list)

    @property
    def box(self) -> np.ndarray:
        return kalman.xyah_to_xyxy(self.mean)

    def predict(self) -> None:
        if self.state != TrackState.CONFIRMED:
            self.mean[7] = 0.0  # freeze height velocity while unobserved
        self.mean, self.cov = kalman.predict(self.mean, self.cov)

    def update(self, box: np.ndarray, score: float, cls: int) -> None:
        self.mean, self.cov = kalman.update(self.mean, self.cov, kalman.xyxy_to_xyah(box))
        self.score, self.cls = score, cls
        self.hits += 1
        self.frames_lost = 0
        self.state = TrackState.CONFIRMED


@dataclass
class TrackOutput:
    id: int
    box: np.ndarray
    score: float
    cls: int


def _match(cost: np.ndarray, max_cost: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment, rejecting pairs costlier than ``max_cost``."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    pairs = [(r, c) for r, c in zip(rows, cols, strict=True) if cost[r, c] <= max_cost]
    matched_r = {r for r, _ in pairs}
    matched_c = {c for _, c in pairs}
    return (
        pairs,
        [r for r in range(cost.shape[0]) if r not in matched_r],
        [c for c in range(cost.shape[1]) if c not in matched_c],
    )


class ByteTracker:
    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_iou: float = 0.2,
        second_match_iou: float = 0.5,
        max_lost: int = 30,
    ):
        """
        Args:
            high_thresh: detections above this take part in the first association.
            low_thresh: detections in [low, high) are only used to extend existing tracks.
                Set equal to ``high_thresh`` to disable the second pass (plain SORT behaviour).
            new_track_thresh: minimum score to start a new track.
            match_iou: minimum IoU for the first association.
            second_match_iou: minimum IoU when matching low-score detections.
            max_lost: frames a track may go unmatched before it is deleted.
        """
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_iou = match_iou
        self.second_match_iou = second_match_iou
        self.max_lost = max_lost
        self.frame = 0
        self.tracks: list[Track] = []
        self._ids = itertools.count(1)

    def compensate(self, warp) -> None:
        """Move every track by known camera motion before the next ``update``.

        ``warp`` maps pixel coordinates (N, 2) in the previous frame to the current one. On a
        drone the ground moves tens of pixels per frame, far more than a constant-velocity
        model learns in time for a short-lived track, so IoU matching fails without this.
        """
        for t in self.tracks:
            x0, y0, x1, y1 = t.box
            corners = np.asarray(warp(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)))
            if not np.isfinite(corners).all():
                continue
            lo, hi = corners.min(0), corners.max(0)
            t.mean[:4] = kalman.xyxy_to_xyah(np.r_[lo, hi])

    def update(self, dets: Detections) -> list[TrackOutput]:
        self.frame += 1
        high = dets[dets.scores >= self.high_thresh]
        low = dets[(dets.scores >= self.low_thresh) & (dets.scores < self.high_thresh)]

        for t in self.tracks:
            t.predict()

        confirmed = [t for t in self.tracks if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
        tentative = [t for t in self.tracks if t.state == TrackState.TENTATIVE]

        # 1. High-score detections vs confirmed + recently lost tracks.
        cost = 1.0 - iou_matrix([t.box for t in confirmed], high.boxes)
        pairs, un_tracks, un_high = _match(cost, 1.0 - self.match_iou)
        for ti, di in pairs:
            confirmed[ti].update(high.boxes[di], float(high.scores[di]), int(high.classes[di]))

        # 2. Low-score detections rescue tracks that were visible last frame.
        remaining = [confirmed[i] for i in un_tracks if confirmed[i].state == TrackState.CONFIRMED]
        lost_already = [confirmed[i] for i in un_tracks if confirmed[i].state == TrackState.LOST]
        cost = 1.0 - iou_matrix([t.box for t in remaining], low.boxes)
        pairs, un_rem, _ = _match(cost, 1.0 - self.second_match_iou)
        for ti, di in pairs:
            remaining[ti].update(low.boxes[di], float(low.scores[di]), int(low.classes[di]))
        for t in [remaining[i] for i in un_rem] + lost_already:
            t.state = TrackState.LOST
            t.frames_lost += 1

        # 3. Tentative tracks need a second consecutive high-score hit to be confirmed.
        high_left = high[np.asarray(un_high, dtype=np.int64)]
        cost = 1.0 - iou_matrix([t.box for t in tentative], high_left.boxes)
        pairs, un_tent, un_high2 = _match(cost, 0.7)
        for ti, di in pairs:
            tentative[ti].update(high_left.boxes[di], float(high_left.scores[di]), int(high_left.classes[di]))
        for i in un_tent:
            tentative[i].state = TrackState.REMOVED

        # 4. Start new tracks from confident leftovers.
        for di in un_high2:
            if high_left.scores[di] >= self.new_track_thresh:
                mean, cov = kalman.initiate(kalman.xyxy_to_xyah(high_left.boxes[di]))
                t = Track(next(self._ids), mean, cov, float(high_left.scores[di]),
                          int(high_left.classes[di]), self.frame)
                if self.frame == 1:
                    t.state = TrackState.CONFIRMED
                self.tracks.append(t)

        for t in self.tracks:
            if t.state == TrackState.LOST and t.frames_lost > self.max_lost:
                t.state = TrackState.REMOVED
        self.tracks = [t for t in self.tracks if t.state != TrackState.REMOVED]

        return [
            TrackOutput(t.id, t.box, t.score, t.cls)
            for t in self.tracks
            if t.state == TrackState.CONFIRMED and t.frames_lost == 0
        ]


class IoUTracker:
    """Greedy frame-to-frame IoU matching with no motion model (a common naive baseline)."""

    def __init__(self, iou_thresh: float = 0.3, score_thresh: float = 0.5, max_lost: int = 1):
        self.iou_thresh, self.score_thresh, self.max_lost = iou_thresh, score_thresh, max_lost
        self.tracks: dict[int, tuple[np.ndarray, int, int]] = {}  # id -> (box, cls, frames_lost)
        self._ids = itertools.count(1)

    def update(self, dets: Detections) -> list[TrackOutput]:
        dets = dets[dets.scores >= self.score_thresh]
        ids = list(self.tracks)
        ious = iou_matrix([self.tracks[i][0] for i in ids], dets.boxes)
        used_t, used_d, out = set(), set(), []
        for flat in np.argsort(-ious, axis=None):
            ti, di = np.unravel_index(flat, ious.shape)
            if ious[ti, di] < self.iou_thresh:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            self.tracks[ids[ti]] = (dets.boxes[di], int(dets.classes[di]), 0)
            out.append(TrackOutput(ids[ti], dets.boxes[di], float(dets.scores[di]), int(dets.classes[di])))
        for ti, tid in enumerate(ids):
            if ti not in used_t:
                box, cls, lost = self.tracks[tid]
                if lost + 1 > self.max_lost:
                    del self.tracks[tid]
                else:
                    self.tracks[tid] = (box, cls, lost + 1)
        for di in range(len(dets)):
            if di not in used_d:
                tid = next(self._ids)
                self.tracks[tid] = (dets.boxes[di], int(dets.classes[di]), 0)
                out.append(TrackOutput(tid, dets.boxes[di], float(dets.scores[di]), int(dets.classes[di])))
        return out
