"""Score DeepX whole-body keypoints (from wholebody_video.py) against the same RTMW model unquantised (ONNX Runtime on the laptop), given
the same climber boxes, so what differs is what the chip's 8-bit quantisation costs. Errors are % of trunk length.

    deepx_wholebody_compare.py VIDEO RTMW.onnx DEEPX.npz [more DEEPX.npz with their onnx: --pair ONNX NPZ]
"""
import argparse

import cv2
import numpy as np
from rtmlib import RTMPose

GROUPS = {"body": range(0, 17), "feet": range(17, 23), "face": range(23, 91), "hands": range(91, 133)}
THR = 0.3

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--pair", nargs=2, action="append", metavar=("ONNX", "NPZ"), required=True)
a = ap.parse_args()

cap = cv2.VideoCapture(a.video); frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)

for onnx, npz in a.pair:
    z = np.load(npz)
    model = RTMPose(onnx, model_input_size=(192, 256), backend="onnxruntime", device="cuda")
    err = {g: [] for g in GROUPS}; agree = {g: [] for g in GROUPS}
    for f, b, k, s in zip(frames, z["boxes"], z["kps"], z["scores"]):
        if np.isnan(b[0]):
            continue
        rk, rs = model(f, bboxes=[b[:4]]); rk, rs = rk[0], rs[0]
        trunk = np.linalg.norm((rk[5] + rk[6]) / 2 - (rk[11] + rk[12]) / 2)
        if trunk < 20:
            continue
        e = np.linalg.norm(rk - k, axis=1) / trunk
        for g, idx in GROUPS.items():
            idx = np.array(idx)
            vis = rs[idx] > THR
            err[g] += list(e[idx][vis & (s[idx] > THR)])
            agree[g] += list((s[idx] > THR) == vis)
    print(f"{npz}: DeepX vs same model on GPU, median error (% trunk) / within 5% / visibility agrees: " + "; ".join(
        f"{g} {100*np.median(v):.1f}% / {np.mean(np.array(v) < .05):.0%} / {np.mean(agree[g]):.0%}" for g, v in err.items()))
