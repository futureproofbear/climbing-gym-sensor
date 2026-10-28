"""The gym's two pose models on either M.2 accelerator, a DeepX DX-M1 or a Hailo-8, behind one interface.

Problem: the gym unit should not depend on one chip vendor. Both chips run the same two models: YOLO11s-pose finds
people (boxes and 17 body points per person) and RTMW gives 133 whole-body points on a crop around the climber.
  Detector(...).infer(bgr)   -> boxes [N, 5] (x1 y1 x2 y2 conf), keypoints [N, 17, 3], in frame pixels
  WholeBody(...).run(crop)   -> SimCC (x [133, 384], y [133, 512]) for a 192x256 uint8 BGR crop
Which chip: NPU=deepx|hailo, else whichever device is present (/dev/dxrt0, /dev/hailo0).

DeepX: the YOLO model folder through Ultralytics; RTMW a .dxnn with ImageNet normalisation folded in (deepx_rtmw_prep.py).
Hailo: .hef files. YOLO is cut at its 9 head convolutions; box (DFL), class and keypoint decoding and NMS run here, as
in the Hailo Model Zoo. RTMW is cut after its backbone (hailo_rtmw_compile.py); its token head runs on the CPU from a
small ONNX file (onnxruntime). Both Hailo models share one device through the HailoRT round-robin scheduler.
"""
import os

import cv2
import numpy as np

RTMW_W, RTMW_H = 192, 256


def which():
    choice = os.environ.get("NPU")
    if choice:
        return choice
    if os.path.exists("/dev/dxrt0"):
        return "deepx"
    if os.path.exists("/dev/hailo0"):
        return "hailo"
    raise RuntimeError("no NPU found (/dev/dxrt0 or /dev/hailo0); set NPU=deepx|hailo")


_vdevice = None


def hailo_vdevice():
    """One VDevice with the round-robin scheduler, so the detector and RTMW share the Hailo-8."""
    global _vdevice
    if _vdevice is None:
        from hailo_platform import HailoSchedulingAlgorithm, VDevice
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        _vdevice = VDevice(params)
    return _vdevice


class HailoNet:
    """A .hef on the shared VDevice: uint8 NHWC in, float32 outputs by name."""

    def __init__(self, hef_path):
        from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams,
                                    InputVStreamParams, OutputVStreamParams)
        hef = HEF(hef_path)
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng = hailo_vdevice().configure(hef, cfg)[0]
        info = hef.get_input_vstream_infos()[0]
        self.input_name, self.shape = info.name, tuple(info.shape)          # (H, W, C)
        self.pipe = InferVStreams(ng, InputVStreamParams.make(ng, format_type=FormatType.UINT8),
                                  OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32))
        self.pipe.__enter__()

    def __call__(self, x):
        return self.pipe.infer({self.input_name: np.ascontiguousarray(x)})


class Detector:
    def __init__(self, model, npu=None, conf=0.25, iou=0.7):
        self.npu, self.conf, self.iou = npu or which(), conf, iou
        if self.npu == "deepx":
            from ultralytics import YOLO
            self.yolo = YOLO(model, task="pose")
        else:
            self.net = HailoNet(model)
            self.size = self.net.shape[0]

    def infer(self, bgr):
        if self.npu == "deepx":
            r = self.yolo.predict(bgr, imgsz=640, conf=self.conf, iou=self.iou, verbose=False)[0]
            k = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((0, 17, 3))
            return np.c_[r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()], k
        h, w = bgr.shape[:2]
        s = self.size / max(h, w)
        nh, nw = round(h * s), round(w * s)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        boxes, scores, kpts = self._decode(self.net(canvas[None, ..., ::-1]))      # the model takes RGB
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / s
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / s
        kpts[..., 0] = (kpts[..., 0] - left) / s
        kpts[..., 1] = (kpts[..., 1] - top) / s
        return np.c_[boxes, scores], kpts

    def _decode(self, out):
        """YOLOv8/11 pose head per stride: box (64 = 4 sides x 16 DFL bins), class (1), keypoints (51)."""
        by_hw = {}
        for v in out.values():
            v = v[0]
            by_hw.setdefault(v.shape[0], {})[v.shape[2]] = v
        bins = np.arange(16, dtype=np.float32)
        B, S, K = [], [], []
        for hw in sorted(by_hw, reverse=True):
            stride, heads = self.size // hw, by_hw[hw]
            cls = heads[1].reshape(-1)
            if cls.min() < 0:                                     # logits, unless the chip applied the sigmoid
                cls = 1 / (1 + np.exp(-cls))
            keep = cls >= self.conf
            if not keep.any():
                continue
            ys, xs = np.divmod(np.nonzero(keep)[0], hw)
            d = heads[64].reshape(-1, 4, 16)[keep]
            d = np.exp(d - d.max(-1, keepdims=True))
            d = (d / d.sum(-1, keepdims=True)) @ bins                 # l, t, r, b in cells
            cx, cy = xs + 0.5, ys + 0.5
            B.append(np.stack([cx - d[:, 0], cy - d[:, 1], cx + d[:, 2], cy + d[:, 3]], 1) * stride)
            k = heads[51].reshape(-1, 17, 3)[keep].copy()
            k[..., 0] = (k[..., 0] * 2.0 + xs[:, None]) * stride
            k[..., 1] = (k[..., 1] * 2.0 + ys[:, None]) * stride
            k[..., 2] = 1 / (1 + np.exp(-k[..., 2]))
            S.append(cls[keep]); K.append(k)
        if not B:
            return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 17, 3))
        boxes, scores, kpts = np.concatenate(B), np.concatenate(S), np.concatenate(K)
        xywh = np.c_[boxes[:, :2], boxes[:, 2:] - boxes[:, :2]]
        idx = np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), self.conf, self.iou), int).reshape(-1)
        return boxes[idx], scores[idx], kpts[idx]


class WholeBody:
    """RTMW on a 192x256 BGR uint8 crop -> SimCC x [133, 384], y [133, 512]."""

    def __init__(self, model, npu=None, head=None):
        """model: .dxnn (DeepX) or .hef (Hailo); head: the Hailo build's *_tail.onnx (default: next to the .hef)."""
        self.npu = npu or which()
        if self.npu == "deepx":
            from dx_engine import InferenceEngine
            self.engine = InferenceEngine(model)
        else:
            import onnxruntime as ort
            self.net = HailoNet(model)
            head = head or model.rsplit("_256x192", 1)[0] + "_tail.onnx"
            self.head = ort.InferenceSession(head, providers=["CPUExecutionProvider"])
            self.head_in = {tuple(i.shape[2:]): i.name for i in self.head.get_inputs()}   # (h, w) -> tensor

    def run(self, crop):
        if self.npu == "deepx":
            o = self.engine.run([np.ascontiguousarray(crop[None])])
            sx, sy = sorted((np.asarray(a) for a in o), key=np.size)
        else:
            feeds = {self.head_in[v.shape[1:3]]: np.ascontiguousarray(v.transpose(0, 3, 1, 2), dtype=np.float32)
                     for v in self.net(crop[None]).values()}
            sx, sy = self.head.run(["simcc_x", "simcc_y"], feeds)
        return sx.reshape(133, -1), sy.reshape(133, -1)
