"""Compile RTMW whole-body pose (133 points, SimCC heads) for the Hailo-8 with the Hailo Dataflow Compiler.

Problem: the gym's whole-body pose runs on the DeepX NPU; a Hailo-8 unit needs the same model as a .hef. RTMW's head
turns the feature maps into 133 keypoint tokens and runs MLPs and a gated attention unit over them. The compiler
rejects that part (token-major matrix products, and the attention's offset-scale). So the model is cut after the
backbone's two final 7x7 convolutions (their ReLUs: [133, 8, 6] and [133, 16, 12]); the chip runs everything up
to there, and the token head (~100 M multiply-adds per person) runs on the CPU as a small ONNX file. Stages, each
saved so a later one can be rerun alone:
  translate  ONNX -> .har (batch fixed to 1), cut at the two convs; writes the CPU head as *_tail.onnx
  optimize   quantise to 8 bits on calibration crops (BGR person crops, as the runtime feeds them); ImageNet mean/std
             are a normalisation layer on the chip, so the input is plain 0-255
  compile    -> .hef
  check      the quantised chip part in the SDK emulator plus the CPU tail, against the float model, on crops not
             used for calibration: keypoint error in pixels of the 192x256 crop

Status (RTMW dw-x-l, 28 Oct 2026, DFC 3.34): the cut model translates, and in float the emulator plus the CPU head
match ONNX exactly. Quantised to 8 bits it fails: the chip's feature maps keep only 3-5 dB SNR against float, and
keypoints are off by a median ~80 px in a 192x256 crop (optimisation level 1 or 2, 16-bit output layers or not). The
DeepX quantises the same model within ~1% of trunk. The compile itself needs more than WSL's 15 GB of memory.

    python hailo_rtmw_compile.py RTMW.onnx CALIB_DIR OUT_DIR [--stages translate,optimize,compile,check] [--level 1]
CALIB_DIR: *.png crops 192x256 (deepx_rtmw_prep.py writes them). Run where the DFC is installed (x86 Linux / WSL).
"""
import argparse
import glob
import os

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from hailo_sdk_client import ClientRunner, InferenceContext

W, H = 192, 256
MEAN = [123.675, 116.28, 103.53]
STD = [58.395, 57.12, 57.375]
SPLIT = 2.0                 # SimCC bins per pixel
HOLDOUT = 24                # crops kept out of calibration for the check
MAPS = [(8, 6), (16, 12)]    # the two 133-channel maps at the cut, coarse first, for a 256x192 input


def cut_points(m):
    """The ReLUs after the head's two 133-channel convolutions, in graph order (coarse map first):
    (node names, {map size: output tensor})."""
    weights = {i.name: list(i.dims) for i in m.graph.initializer}
    convs = [n.output[0] for n in m.graph.node if n.op_type == "Conv" and weights.get(n.input[1], [0])[0] == 133]
    relus = [n for n in m.graph.node if n.op_type == "Relu" and n.input[0] in convs]
    assert len(relus) == 2, f"expected 2 head ReLUs, found {len(relus)}"
    return [n.name for n in relus], {hw: n.output[0] for hw, n in zip(MAPS, relus)}


def crops(calib_dir):
    files = sorted(glob.glob(f"{calib_dir}/*.png"))
    x = np.stack([cv2.resize(cv2.imread(f), (W, H)) for f in files]).astype(np.float32)   # BGR, NHWC
    return x[:-HOLDOUT], x[-HOLDOUT:]


def decode(sx, sy):
    """SimCC argmax -> (n, 133, 2) crop pixels."""
    return np.stack([sx.argmax(-1), sy.argmax(-1)], -1) / SPLIT


def tail(sess, outs):
    """Chip outputs (NHWC, 133 channels, any order) through the CPU head -> (simcc_x, simcc_y)."""
    names = {tuple(i.shape[2:]): i.name for i in sess.get_inputs()}
    feeds = {}
    for o in outs:
        feeds[names[o.shape[1:3]]] = np.ascontiguousarray(o.transpose(0, 3, 1, 2), dtype=np.float32)
    xs, ys = [], []
    for i in range(len(next(iter(feeds.values())))):
        sx, sy = sess.run(["simcc_x", "simcc_y"], {k: v[i:i + 1] for k, v in feeds.items()})
        xs.append(sx[0]); ys.append(sy[0])
    return np.stack(xs), np.stack(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("calib")
    ap.add_argument("out")
    ap.add_argument("--stages", default="translate,optimize,compile,check")
    ap.add_argument("--level", type=int, default=1, help="DFC optimization_level (0-4; 2+ wants a GPU)")
    ap.add_argument("--a16", default="outputs", help="layers kept at 16 bits: 'outputs' (the convs feeding the CPU head), 'dw' (all depthwise convs"
                    "), 'none', or comma-separated layer names")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(a.onnx).split("_simcc")[0].replace("-", "_")
    har = {s: f"{a.out}/{name}_{s}.har" for s in ("parsed", "quantized")}
    stages = a.stages.split(",")
    calib, hold = crops(a.calib)

    if "translate" in stages:
        m = onnx.load(a.onnx)
        m.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
        onnx.save(m, f"{a.out}/{name}_b1.onnx")
        end_nodes, tail_in = cut_points(m)
        print("cut after", end_nodes, flush=True)
        tail_path = f"{a.out}/{name}_tail.onnx"
        onnx.utils.extract_model(f"{a.out}/{name}_b1.onnx", tail_path, list(tail_in.values()),
                                 ["simcc_x", "simcc_y"], check_model=False)
        t = onnx.load(tail_path)
        for v in t.graph.input:                   # the cut tensors carry no shape in the source graph
            hw = next(hw for hw, n in tail_in.items() if n == v.name)
            v.type.tensor_type.ClearField("shape")
            for d in (1, 133, *hw):
                v.type.tensor_type.shape.dim.add().dim_value = d
        onnx.save(t, tail_path)
        runner = ClientRunner(hw_arch="hailo8")
        runner.translate_onnx_model(f"{a.out}/{name}_b1.onnx", name, end_node_names=end_nodes,
                                    net_input_shapes={"input": [1, 3, H, W]})
        runner.save_har(har["parsed"])
        print("translated:", har["parsed"], "and the CPU tail", f"{a.out}/{name}_tail.onnx", flush=True)

    if "optimize" in stages:
        runner = ClientRunner(har=har["parsed"])
        layers = runner.get_hn_dict()["layers"]
        if a.a16 == "outputs":
            wide = [p for n, l in layers.items() if l["type"] == "output_layer" for p in l["input"]]
        elif a.a16 == "dw":                        # depthwise convs: the noisiest layers at 8 bits (analyze-noise)
            wide = [n for n, l in layers.items() if l["type"] == "dw"]
        else:
            wide = [] if a.a16 == "none" else a.a16.split(",")
        wide = [w.split("/")[-1] for w in wide]
        print("16-bit layers:", wide, flush=True)
        runner.load_model_script(
            f"normalization1 = normalization({MEAN}, {STD})\n"
            + "".join(f"quantization_param({w}, precision_mode=a16_w16)\n" for w in wide) +
            f"model_optimization_flavor(optimization_level={a.level}, compression_level=0)\n"
            f"model_optimization_config(calibration, batch_size=8, calibset_size={len(calib)})\n")
        runner.optimize(calib)
        runner.save_har(har["quantized"])
        print("quantised:", har["quantized"], flush=True)

    if "compile" in stages:
        runner = ClientRunner(har=har["quantized"])
        hef = runner.compile()
        path = f"{a.out}/{name}_256x192.hef"
        open(path, "wb").write(hef)
        print(f"compiled: {path} ({len(hef) / 1e6:.1f} MB)", flush=True)

    if "check" in stages:
        full = ort.InferenceSession(f"{a.out}/{name}_b1.onnx", providers=["CPUExecutionProvider"])
        cut = ort.InferenceSession(f"{a.out}/{name}_tail.onnx", providers=["CPUExecutionProvider"])
        norm = ((hold - np.array(MEAN, np.float32)) / np.array(STD, np.float32)).transpose(0, 3, 1, 2)
        ref = [np.concatenate(o) for o in zip(*(full.run(["simcc_x", "simcc_y"], {"input": x[None]}) for x in norm))]
        runner = ClientRunner(har=har["quantized"])
        with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
            q = tail(cut, runner.infer(ctx, hold))
        d = np.linalg.norm(decode(*ref) - decode(*q), axis=-1)            # (n, 133) pixels in the crop
        groups = {"body": range(0, 17), "feet": range(17, 23), "face": range(23, 91), "hands": range(91, 133)}
        print(f"quantised vs float on {len(hold)} held-out crops, median keypoint error in crop pixels "
              f"(crop {W}x{H}; a trunk is ~100 px): " +
              ", ".join(f"{g} {np.median(d[:, list(r)]):.1f}" for g, r in groups.items()), flush=True)


if __name__ == "__main__":
    main()
