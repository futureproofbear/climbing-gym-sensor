"""Whole-body pose (RTMW, 133 points) for a video on the NPU (DeepX or Hailo, npu.py): YOLO11s-pose finds the climber,
RTMW runs on a crop around them. Runs on the Pi without rtmlib: the crop and the SimCC decode are rtmlib's, written out.

    python3 wholebody_video.py VIDEO DET_MODEL RTMW_MODEL OUT.npz [--npu deepx|hailo]
DET_MODEL: the DeepX YOLO folder or a .hef; RTMW_MODEL: a .dxnn or a .hef (its CPU head *_tail.onnx next to it).
Saves per frame: the climber box (x1 y1 x2 y2 conf, NaN if none), keypoints [133, 2] and scores [133]; prints timing.
"""
import argparse
import time

import cv2
import numpy as np

from npu import RTMW_H as H, RTMW_W as W, Detector, WholeBody

SPLIT = 2.0                 # SimCC bins per pixel
PAD = 1.25                  # rtmlib's box padding


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
    locs = np.stack([sx.argmax(1), sy.argmax(1)], -1).astype(np.float32) / SPLIT
    score = (sx.max(1) + sy.max(1)) / 2
    return locs / [W, H] * scale + center - scale / 2, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("det")
    ap.add_argument("rtmw")
    ap.add_argument("out")
    ap.add_argument("--npu", help="deepx or hailo (default: whichever is present)")
    a = ap.parse_args()
    det, rtmw = Detector(a.det, a.npu), WholeBody(a.rtmw, a.npu)

    cap = cv2.VideoCapture(a.video)
    ok, f = cap.read()
    for _ in range(5):                                   # warm-up
        det.infer(f)
        rtmw.run(np.zeros((H, W, 3), np.uint8))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    boxes, kps, scores, t_det, t_crop, t_npu, t_dec = [], [], [], [], [], [], []
    prev = None
    t_all = time.perf_counter()
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        b, _ = det.infer(f)
        t1 = time.perf_counter()
        if not len(b):
            boxes.append(np.full(5, np.nan)); kps.append(np.full((133, 2), np.nan)); scores.append(np.zeros(133))
            continue
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        if prev is not None:          # follow the climber: nearest to last frame's box
            i = int(np.argmin(np.linalg.norm((b[:, :2] + b[:, 2:4]) / 2 - (prev[:2] + prev[2:4]) / 2, axis=1)
                              - 1e-3 * area))
        else:
            i = int(np.argmax(area))
        prev = b[i]
        img, center, scale = crop(f, b[i])
        t2 = time.perf_counter()
        sx, sy = rtmw.run(img)
        t3 = time.perf_counter()
        k, s = decode(sx, sy, center, scale)
        t4 = time.perf_counter()
        boxes.append(b[i]); kps.append(k); scores.append(s)
        t_det.append(t1 - t0); t_crop.append(t2 - t1); t_npu.append(t3 - t2); t_dec.append(t4 - t3)
    wall = time.perf_counter() - t_all
    np.savez(a.out, boxes=np.array(boxes), kps=np.array(kps), scores=np.array(scores))
    ms = lambda t: f"{1e3 * np.median(t):.0f}"
    print(f"{det.npu}: {len(boxes)} frames in {wall:.1f}s = {len(boxes) / wall:.1f} fps; detect {ms(t_det)} ms, "
          f"crop {ms(t_crop)} ms, RTMW {ms(t_npu)} ms, decode {ms(t_dec)} ms")


if __name__ == "__main__":
    main()
