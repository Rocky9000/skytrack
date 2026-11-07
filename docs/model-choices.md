# Model choices

Why each model in the system was picked, what it is bad at, and what was ruled out. The onboard computer is a Jetson Orin NX, so most decisions come down to accuracy per millisecond on that board.

Latency figures here are design targets for the Jetson after TensorRT export, not measurements from this repository. Numbers measured in this repo are in the main README.

## Overview

| Task | Runs on | Model | Runs how often |
|---|---|---|---|
| Pothole detection | Jetson, downward camera | YOLOv8s, fine-tuned on RDD2022 | Every frame |
| Pothole outline | Jetson | DeepLabv3+ (ResNet-50) | Only on confirmed detections |
| Pothole depth | Jetson | Depth Anything V2, small | Only on confirmed detections |
| Wire detection | Jetson, forward stereo pair | YOLOv8-seg fine-tuned on TTPLA, plus CREStereo disparity | Every frame |
| Avoidance planning | Jetson CPU | MPPI (no learned weights) | 50 Hz or faster |
| Severity grading | Server | EfficientNet-B4 classifier | Once per pothole, on the full-resolution crop |

## 1. Pothole detection: YOLOv8

A single-stage detector gives a box in one forward pass, which is what a moving drone needs. After INT8 TensorRT export the target is 30 to 60 FPS on the Orin NX. It fine-tunes well on RDD2022 (about 47,000 road images from six countries, India and Japan among them), and the same family has a segmentation head if box-level output turns out to be too coarse.

Weak spots: shadows, patches and cracks get confused with potholes; small potholes disappear at high altitude (sliced inference in `tiling.py` helps); wet, reflective asphalt lowers recall. Indian roads need their own labelled images on top of RDD2022.

| Ruled out | Reason |
|---|---|
| Faster R-CNN | Two-stage; only a few FPS on edge hardware |
| SSD MobileNet | Faster, but misses small potholes |
| EfficientDet-D4 | Similar accuracy for considerably more compute |
| RT-DETR | Strong accuracy, but Jetson deployment tooling was not mature enough when this was designed |

## 2. Wire detection: YOLOv8-seg + CREStereo

Cables are the most dangerous obstacle for a low-flying drone: one or two pixels wide, low contrast, and invisible to LiDAR. A segmentation model fine-tuned on TTPLA (power lines and towers seen from the air) finds the wire pixels, and CREStereo gives dense disparity on the forward stereo pair. `wires.py` turns the two into 3D segments by fitting disparity along each wire, since per-pixel stereo on a thin wire is mostly holes.

Weak spots: backlit sky and cluttered backgrounds; TTPLA is small (about 1,100 images); stereo calibration drifts; with a short baseline the depth is only trustworthy to roughly 15 m (the tests in `test_wires.py` show the error growing with range).

| Ruled out | Reason |
|---|---|
| LiDAR alone | Beam spacing misses thin cables entirely |
| Time-of-flight depth | Same problem, plus sunlight noise outdoors |
| RAFT-Stereo | Slightly more accurate, but did not fit the latency budget |
| Ultrasonic | Short range, and blind to thin horizontal wires |

## 3. Pothole outline: DeepLabv3+

YOLO gives a box; area and volume need the actual boundary. DeepLabv3+ runs on the crop around a confirmed detection only, so its roughly 8 FPS speed does not matter. Atrous convolutions keep spatial detail, which helps with ragged pothole edges. A MobileNetV2 backbone is the fallback if the Jetson is short on compute.

Weak spots: needs pixel-level labels, which are expensive; road pixels vastly outnumber pothole pixels, so class weighting matters.

| Ruled out | Reason |
|---|---|
| SAM / SAM 2 | Far too heavy for the Jetson; useful offline for speeding up annotation |
| U-Net | Good in narrow domains, weaker on varied outdoor scenes |
| SegFormer-B5 | Excellent quality, too slow on edge |
| Mask R-CNN | Instance masks are not needed; a semantic pothole class is enough |

## 4. Depth: Depth Anything V2 (small)

Gives depth from the single downward camera, so no extra sensor is needed for the road. The small variant fits the Jetson's memory. It handles low-texture asphalt better than MiDaS.

Monocular depth is never exactly right at a new altitude, so `measure.align_depth` corrects it against the depth the drone's pose predicts for the flat road around each pothole. With the metric variant only a shift is fitted. With the relative variant both scale and shift are fitted, which only works when the road depth varies across the crop; over a flat road seen straight down the scale cannot be recovered and the code refuses rather than guess.

Weak spots: accuracy falls off above about 10 m altitude; wet roads cause depth errors; no temporal consistency between frames.

| Ruled out | Reason |
|---|---|
| MiDaS v3.1 | Superseded by Depth Anything on outdoor scenes |
| DPT | Too slow on the Jetson |
| LiDAR | Too sparse to resolve a shallow pothole from 10 m |
| Structured light | Washed out by sunlight |

## 5. Avoidance planning: MPPI

MPPI samples hundreds of acceleration sequences, rolls each one through a point-mass model of the drone, scores them against the goal and the wire segments, and averages them weighted by cost. There is nothing to train, it copes with uncertainty in where the wire is, and the output is a velocity setpoint PX4 or ArduPilot accept over MAVLink.

Weak spots: depends on a reasonable dynamics model; wind gusts invalidate a plan; the cost weights need tuning; it can squeeze through a gap exactly as wide as the safety margin allows (see the planner scenarios in the README).

| Ruled out | Reason |
|---|---|
| RRT* | Too slow to replan reactively in flight |
| A* on a grid | Discrete paths, expensive to replan at speed |
| Deep RL policy | Hard to guarantee behaviour on wire layouts it has not seen; not acceptable on flying hardware |
| PID + potential fields | Oscillates near obstacles, no lookahead |

## 6. Severity grading: EfficientNet-B4

Runs on the server on the full-resolution crop the drone uploads, so there is no compute limit. It grades 1 to 5 for the repair queue, and sees things the depth map does not, such as exposed base layer or standing water. The graded records are meant to feed a PostGIS table that drives a city-wide maintenance map.

Until there are enough expert-graded crops to train it, `severity.rule_grade` grades from measured depth and area, using the FHWA depth bands. Those rule-based grades are also a starting point for labelling.

Weak spots: severity labels are subjective and annotators disagree; ground-truth grades are expensive to collect.

| Ruled out | Reason |
|---|---|
| ResNet-50 | Less accurate than EfficientNet-B4 at similar compute |
| VGG-16 | Large and outdated |
| ViT-B | Needs far more data than any road severity dataset has |
| ConvNeXt-B | Much larger for a marginal gain on this task |
