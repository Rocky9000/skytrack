"""Small-object detection and multi-object tracking for aerial (drone) video."""

from .boxes import Detections, batched_nms, iou_matrix, nms
from .metrics import detection_map, mot_metrics
from .tiling import make_tiles, sliced_predict
from .tracking import ByteTracker, IoUTracker

__all__ = [
    "ByteTracker",
    "Detections",
    "IoUTracker",
    "batched_nms",
    "detection_map",
    "iou_matrix",
    "make_tiles",
    "mot_metrics",
    "nms",
    "sliced_predict",
]
__version__ = "0.1.0"
