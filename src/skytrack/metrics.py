"""Detection (COCO-style mAP) and tracking (CLEAR-MOT, IDF1) metrics from scratch."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .boxes import Detections, iou_matrix

COCO_IOUS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO 101-point interpolated AP."""
    if recall.size == 0:
        return 0.0
    envelope = np.maximum.accumulate(precision[::-1])[::-1]  # best precision at recall >= r
    idx = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(np.mean([envelope[i] if i < len(envelope) else 0.0 for i in idx]))


def _ap_for_class(preds, gts, cls: int, thr: float) -> float | None:
    n_gt = sum(int((g[1] == cls).sum()) for g in gts)
    if n_gt == 0:
        return None
    records = []  # (score, image index, box)
    for img, p in enumerate(preds):
        m = p.classes == cls
        records += [(float(s), img, b) for s, b in zip(p.scores[m], p.boxes[m], strict=True)]
    records.sort(key=lambda r: -r[0])
    gt_boxes = [g[0][g[1] == cls] for g in gts]
    taken = [np.zeros(len(b), bool) for b in gt_boxes]
    tp = np.zeros(len(records))
    for k, (_, img, box) in enumerate(records):
        if len(gt_boxes[img]) == 0:
            continue
        ious = iou_matrix(box[None], gt_boxes[img])[0]
        ious[taken[img]] = -1.0
        j = int(np.argmax(ious))
        if ious[j] >= thr:
            taken[img][j] = True
            tp[k] = 1.0
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(records) + 1)
    return average_precision(recall, precision)


def detection_map(
    preds: list[Detections], gts: list[tuple[np.ndarray, np.ndarray]], num_classes: int
) -> dict:
    """mAP@0.5 and mAP@0.5:0.95 over classes that appear in the ground truth.

    ``gts[i]`` is ``(boxes (M, 4), classes (M,))`` for image ``i``.
    """
    per_thr = {}
    per_class_50 = {}
    for thr in COCO_IOUS:
        aps = {c: _ap_for_class(preds, gts, c, float(thr)) for c in range(num_classes)}
        aps = {c: a for c, a in aps.items() if a is not None}
        per_thr[float(thr)] = float(np.mean(list(aps.values()))) if aps else 0.0
        if thr == 0.5:
            per_class_50 = aps
    return {"map50": per_thr[0.5], "map50_95": float(np.mean(list(per_thr.values()))), "ap50_per_class": per_class_50}


Frame = dict[int, np.ndarray]  # object id -> xyxy box


def mot_metrics(gt: list[Frame], hyp: list[Frame], iou_thr: float = 0.5) -> dict:
    """CLEAR-MOT (MOTA, ID switches) and identity metrics (IDF1).

    MOTA follows Bernardin & Stiefelhagen (2008): correspondences from the
    previous frame are kept while IoU stays above threshold, the rest are
    solved with the Hungarian algorithm. IDF1 follows Ristani et al. (2016):
    a single global one-to-one assignment of GT tracks to hypothesis tracks.
    """
    if len(gt) != len(hyp):
        raise ValueError("gt and hyp must have the same number of frames")
    fn = fp = idsw = n_gt = n_hyp = matches = 0
    last_match: dict[int, int] = {}  # gt id -> hyp id it was last matched to
    overlap_counts: dict[tuple[int, int], int] = defaultdict(int)
    gt_len: dict[int, int] = defaultdict(int)
    hyp_len: dict[int, int] = defaultdict(int)

    for g, h in zip(gt, hyp, strict=True):
        n_gt += len(g)
        n_hyp += len(h)
        for gid in g:
            gt_len[gid] += 1
        for hid in h:
            hyp_len[hid] += 1
        g_ids, h_ids = list(g), list(h)
        ious = iou_matrix([g[i] for i in g_ids], [h[i] for i in h_ids])
        for a, gid in enumerate(g_ids):
            for b, hid in enumerate(h_ids):
                if ious[a, b] >= iou_thr:
                    overlap_counts[(gid, hid)] += 1

        pairs: list[tuple[int, int]] = []
        free_g, free_h = set(range(len(g_ids))), set(range(len(h_ids)))
        for a, gid in enumerate(g_ids):  # keep last frame's correspondences if still valid
            hid = last_match.get(gid)
            if hid in h:
                b = h_ids.index(hid)
                if b in free_h and ious[a, b] >= iou_thr:
                    pairs.append((a, b))
                    free_g.discard(a)
                    free_h.discard(b)
        fg, fh = sorted(free_g), sorted(free_h)
        if fg and fh:
            sub = ious[np.ix_(fg, fh)]
            rows, cols = linear_sum_assignment(-sub)
            pairs += [(fg[r], fh[c]) for r, c in zip(rows, cols, strict=True) if sub[r, c] >= iou_thr]

        for a, b in pairs:
            gid, hid = g_ids[a], h_ids[b]
            if gid in last_match and last_match[gid] != hid:
                idsw += 1
            last_match[gid] = hid
        matches += len(pairs)
        fn += len(g_ids) - len(pairs)
        fp += len(h_ids) - len(pairs)

    gids, hids = sorted(gt_len), sorted(hyp_len)
    idtp = 0
    if gids and hids:
        m = np.array([[overlap_counts.get((g, h), 0) for h in hids] for g in gids])
        rows, cols = linear_sum_assignment(-m)
        idtp = int(m[rows, cols].sum())
    return {
        "mota": 1.0 - (fn + fp + idsw) / n_gt if n_gt else 0.0,
        "idf1": 2 * idtp / (n_gt + n_hyp) if (n_gt + n_hyp) else 0.0,
        "id_switches": idsw,
        "false_negatives": fn,
        "false_positives": fp,
        "gt_tracks": len(gids),
        "hyp_tracks": len(hids),
    }
