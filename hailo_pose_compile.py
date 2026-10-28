"""Compile YOLO11 pose ONNX to a Hailo-8 .hef. Run where the Hailo Dataflow Compiler is installed (x86 Linux).

The graph is cut at the 9 head convs (box / class / keypoints at strides 8, 16, 32);
DFL decoding, sigmoid and NMS run on the host, as in the Hailo Model Zoo yolov8_pose.
With --a16-keypoints the 3 keypoint convs (51 channels) keep 16-bit outputs: at 8 bits their wide range leaves a
keypoint step of ~5 px at 640 (16 px on a 1080p frame), enough to break the gym's hand-stillness test.
Usage: python hailo_pose_compile.py yolo11s-pose.onnx /path/to/calib/images [--a16-keypoints] [--name NAME]
"""
import argparse
import glob

import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner

SIZE = 640
END_NODES = [f"/model.23/{head}.{i}/{head}.{i}.2/Conv" for i in range(3) for head in ("cv2", "cv3", "cv4")]

ap = argparse.ArgumentParser()
ap.add_argument("onnx")
ap.add_argument("calib")
ap.add_argument("--a16-keypoints", action="store_true")
ap.add_argument("--name")
a = ap.parse_args()
NAME = a.name or a.onnx.rsplit("/", 1)[-1].replace(".onnx", "").replace("-", "_")


def letterbox(path):
    img = Image.open(path).convert("RGB")
    s = SIZE / max(img.size)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)
    canvas = Image.new("RGB", (SIZE, SIZE), (114, 114, 114))
    canvas.paste(img, ((SIZE - img.width) // 2, (SIZE - img.height) // 2))
    return np.asarray(canvas, dtype=np.float32)


files = sorted(glob.glob(f"{a.calib}/*.jpg"))
calib = np.stack([letterbox(f) for f in files])
print(f"calibration set: {calib.shape}", flush=True)

runner = ClientRunner(hw_arch="hailo8")
runner.translate_onnx_model(a.onnx, NAME, start_node_names=["images"], end_node_names=END_NODES,
                            net_input_shapes={"images": [1, 3, SIZE, SIZE]})
wide = []
if a.a16_keypoints:
    layers = runner.get_hn_dict()["layers"]
    outs = [p for l in layers.values() if l["type"] == "output_layer" for p in l["input"]]
    wide = [n.split("/")[-1] for n in outs if layers[n]["output_shapes"][0][-1] == 51]
    print("16-bit keypoint layers:", wide, flush=True)
runner.load_model_script(
    "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n"
    + "".join(f"quantization_param({w}, precision_mode=a16_w16)\n" for w in wide) +
    f"model_optimization_config(calibration, batch_size=8, calibset_size={len(files)})\n")
runner.optimize(calib)
hef = runner.compile()
with open(f"{NAME}.hef", "wb") as f:
    f.write(hef)
print(f"saved {NAME}.hef ({len(hef) / 1e6:.1f} MB)")
