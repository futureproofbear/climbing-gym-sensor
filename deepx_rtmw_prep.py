"""Prepare an RTMW whole-body ONNX for the DeepX compiler (DX-COM).

Problem: DX-COM calibrates from image files, and RTMW expects person crops normalised with ImageNet mean and std.
Folding that normalisation into the graph means the chip takes plain 0-255 crops and the calibration loader needs only
a transpose. The batch dimension is fixed to 1 (one person per call).

    deepx_rtmw_prep.py RTMW.onnx OUT_DIR VIDEO[@start_s] ... [--n 150]
writes OUT_DIR/<name>_norm.onnx, OUT_DIR/calib/*.png (192x256 crops from rtmlib's own affine) and OUT_DIR/config.json.
"""
import argparse
import json
import os

import cv2
import numpy as np
import onnx
from onnx import helper, numpy_helper
from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine
from ultralytics import YOLO

MEAN = np.array([123.675, 116.28, 103.53], np.float32).reshape(1, 3, 1, 1)
STD = np.array([58.395, 57.12, 57.375], np.float32).reshape(1, 3, 1, 1)
DET = os.environ.get("DET_WEIGHTS", os.path.join(os.environ.get("MODELS_DIR", "models"), "yolo11m-pose.pt"))

ap = argparse.ArgumentParser()
ap.add_argument("onnx")
ap.add_argument("out")
ap.add_argument("videos", nargs="+")
ap.add_argument("--n", type=int, default=150)
a = ap.parse_args()
os.makedirs(f"{a.out}/calib", exist_ok=True)

m = onnx.load(a.onnx)
inp = m.graph.input[0]
dims = inp.type.tensor_type.shape.dim
dims[0].ClearField("dim_param"); dims[0].dim_value = 1
h, w = dims[2].dim_value, dims[3].dim_value
for o in m.graph.output:
    o.type.tensor_type.shape.dim[0].ClearField("dim_param"); o.type.tensor_type.shape.dim[0].dim_value = 1
raw = inp.name + "_raw"
m.graph.initializer.extend([numpy_helper.from_array(MEAN, "norm_mean"), numpy_helper.from_array(1 / STD, "norm_istd")])
for node in m.graph.node:
    node.input[:] = [inp.name + "_n" if x == inp.name else x for x in node.input]
m.graph.node.insert(0, helper.make_node("Sub", [inp.name, "norm_mean"], [inp.name + "_c"]))
m.graph.node.insert(1, helper.make_node("Mul", [inp.name + "_c", "norm_istd"], [inp.name + "_n"]))
onnx.checker.check_model(m)
name = os.path.basename(a.onnx).replace(".onnx", "")
onnx.save(m, f"{a.out}/{name}_norm.onnx")

det = YOLO(DET)
per = a.n // len(a.videos) + 1
k = 0
for v in a.videos:
    path, start = (v.rsplit("@", 1) + [0])[:2]
    cap = cv2.VideoCapture(path)
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(float(start) * fps)
    got = 0
    for i in np.linspace(first, total - 1, per * 2).astype(int):     # spare frames for ones with nobody in
        if got >= per:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            continue
        r = det.predict(f, imgsz=1280, conf=0.5, verbose=False)[0]
        if not len(r.boxes):
            continue
        b = r.boxes.xyxy.cpu().numpy()
        b = b[np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))]
        c, s = bbox_xyxy2cs(b, padding=1.25)
        crop, _ = top_down_affine((w, h), s, c, f)
        cv2.imwrite(f"{a.out}/calib/{k:04d}.png", crop)
        got += 1; k += 1
json.dump({"inputs": {inp.name: [1, 3, h, w]}, "calibration_num": k, "calibration_method": "ema",
           "default_loader": {"dataset_path": os.path.abspath(f"{a.out}/calib"), "file_extensions": ["png"],
                              "preprocessings": [{"transpose": {"axis": [2, 0, 1]}}, {"expandDim": {"axis": 0}}]}},
          open(f"{a.out}/config.json", "w"), indent=1)
print(f"{name}: input {inp.name} 1x3x{h}x{w}, {k} calibration crops")
