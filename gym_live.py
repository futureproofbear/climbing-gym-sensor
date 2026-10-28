"""Gym pattern of life on the Pi: DeepX pose on a video or camera stream, climbs and counts out, no images out.

    python3 gym_live.py SOURCE MAP_DIR OUT_DIR [--model models/yolo11s-pose_deepx_model]
SOURCE is a video file or a stream URL (rtsp://...). MAP_DIR is gym_map.py's output for this camera view.
Appends each finished climb to OUT_DIR/climbs.jsonl as it ends and rewrites OUT_DIR/summary.json every minute.
A file is read at its own frame rate's timestamps; a stream uses the wall clock.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

from gym_pol import GymPoL, WallMap

ap = argparse.ArgumentParser()
ap.add_argument("source")
ap.add_argument("map_dir")
ap.add_argument("out")
ap.add_argument("--model", default=os.path.join(os.environ.get("MODELS_DIR", "models"), "yolo11s-pose_deepx_model"))
ap.add_argument("--deploy", action="store_true", help="suppress per-minute counts below MIN_GROUP")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

model = YOLO(a.model, task="pose")
cap = cv2.VideoCapture(a.source)
is_file = os.path.exists(a.source)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
pol = GymPoL(WallMap(a.map_dir), fps)
written, last_summary, i, t0 = 0, 0.0, 0, time.time()
times = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    t = i / fps if is_file else time.time() - t0
    s = time.perf_counter()
    r = model.predict(f, imgsz=640, conf=0.25, verbose=False)[0]
    boxes = np.c_[r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()]
    kps = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((0, 17, 3))
    pol.update(t, boxes, kps)
    times.append(time.perf_counter() - s)
    if len(pol.climbs) > written:
        with open(f"{a.out}/climbs.jsonl", "a") as fh:
            for c in pol.climbs[written:]:
                fh.write(json.dumps({k: v for k, v in c.items() if k != "path"}) + "\n")
        written = len(pol.climbs)
    if t - last_summary >= 60:
        json.dump(pol.summary(a.deploy), open(f"{a.out}/summary.json", "w"), indent=1)
        last_summary = t
    i += 1
pol.close(i / fps if is_file else time.time() - t0)
with open(f"{a.out}/climbs.jsonl", "a") as fh:
    for c in pol.climbs[written:]:
        fh.write(json.dumps({k: v for k, v in c.items() if k != "path"}) + "\n")
s = pol.summary(a.deploy)
json.dump(s, open(f"{a.out}/summary.json", "w"), indent=1)
print(f"{i} frames, {1e3 * np.median(times):.0f} ms each ({1 / np.median(times):.1f} fps); {len(pol.climbs)} climbs; "
      f"routes {json.dumps(s['routes'])}")
