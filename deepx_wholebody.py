"""Whole-body pose (RTMW, 133 points) on the DeepX NPU, for a video: the DeepX YOLO pose model finds the climber, RTMW
runs on a crop around them. Runs on the Pi without rtmlib: the crop and the SimCC decode are rtmlib's, written out.

    python3 deepx_wholebody.py VIDEO DET_MODEL_DIR RTMW.dxnn OUT.npz
saves per frame: the climber box (x1 y1 x2 y2 conf, NaN if none), keypoints [133, 2] and scores [133]; prints timing.
"""
import sys
import time

import cv2
import numpy as np
from dx_engine import InferenceEngine
from ultralytics import YOLO

W, H = 192, 256             # RTMW input
SPLIT = 2.0                 # SimCC bins per pixel
PAD = 1.25                  # rtmlib's box padding

video, det_dir, rtmw_path, out = sys.argv[1:5]
det = YOLO(det_dir, task="pose")
rtmw = InferenceEngine(rtmw_path)


def crop(img, box):
    """rtmlib's top_down_affine with no rotation: the padded box, widened to the model's aspect ratio, scaled to WxH."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    bw, bh = (box[2] - box[0]) * PAD, (box[3] - box[1]) * PAD
    if bw > bh * W / H:
        bh = bw * H / W
    else:
        bw = bh * W / H
    s = W / bw
    M = np.float32([[s, 0, W / 2 - cx * s], [0, s, H / 2 - cy * s]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR), np.array([cx, cy]), np.array([bw, bh])


def decode(sx, sy, center, scale):
    sx, sy = sx.reshape(133, -1), sy.reshape(133, -1)
    locs = np.stack([sx.argmax(1), sy.argmax(1)], -1).astype(np.float32) / SPLIT
    score = (sx.max(1) + sy.max(1)) / 2
    return locs / [W, H] * scale + center - scale / 2, score


cap = cv2.VideoCapture(video)
ok, f = cap.read()
for _ in range(5):
    det.predict(f, imgsz=640, conf=0.25, verbose=False)
    rtmw.run([np.zeros((1, H, W, 3), np.uint8)])
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
boxes, kps, scores, t_det, t_crop, t_npu, t_dec = [], [], [], [], [], [], []
prev = None
t_all = time.perf_counter()
while True:
    ok, f = cap.read()
    if not ok:
        break
    t0 = time.perf_counter()
    r = det.predict(f, imgsz=640, conf=0.25, verbose=False)[0]
    t1 = time.perf_counter()
    b = r.boxes.xyxy.cpu().numpy(); c = r.boxes.conf.cpu().numpy()
    if not len(b):
        boxes.append(np.full(5, np.nan)); kps.append(np.full((133, 2), np.nan)); scores.append(np.zeros(133))
        continue
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    if prev is not None:          # follow the climber: nearest to last frame's box
        i = int(np.argmin(np.linalg.norm((b[:, :2] + b[:, 2:]) / 2 - (prev[:2] + prev[2:]) / 2, axis=1) - 1e-3 * area))
    else:
        i = int(np.argmax(area))
    prev = b[i]
    img, center, scale = crop(f, b[i])
    x = np.ascontiguousarray(img[None])        # the NPU takes uint8 NHWC; normalisation is in the model
    t2 = time.perf_counter()
    o = rtmw.run([x])
    t3 = time.perf_counter()
    sx, sy = sorted((np.asarray(a) for a in o), key=np.size)       # x has W*SPLIT bins, y H*SPLIT
    k, s = decode(sx, sy, center, scale)
    t4 = time.perf_counter()
    boxes.append(np.r_[b[i], c[i]]); kps.append(k); scores.append(s)
    t_det.append(t1 - t0); t_crop.append(t2 - t1); t_npu.append(t3 - t2); t_dec.append(t4 - t3)
wall = time.perf_counter() - t_all
np.savez(out, boxes=np.array(boxes), kps=np.array(kps), scores=np.array(scores))
ms = lambda t: f"{1e3 * np.median(t):.0f}"
print(f"{rtmw_path}: {len(boxes)} frames in {wall:.1f}s = {len(boxes) / wall:.1f} fps; detect {ms(t_det)} ms, "
      f"crop {ms(t_crop)} ms, RTMW on NPU {ms(t_npu)} ms, decode {ms(t_dec)} ms; output shapes {[np.asarray(a).shape for a in o]}")
