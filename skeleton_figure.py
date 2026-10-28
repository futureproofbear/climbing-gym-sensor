"""A climb drawn as a stick figure on a plain canvas: hold shapes from the map, the climber's skeleton from pose.

Problem: showing what the sensor sees without showing a person or the place. This draws only what the unit keeps or
could keep (hold outlines and body points), never a video frame: holds are grey shapes on white, the lit problem in
its colours, the climber a skeleton coloured as in the DeepX preview.

    python3 skeleton_figure.py MAP_DIR POSE.npz OUT.png [--gif OUT.gif] [--from 15 --to 33] [--fps 30]
POSE.npz: deepx_wholebody.py's output (one climber per frame) or gym_pol.py's format (the box nearest the last one is
followed). OUT.png: the climb at POSES moments, earlier ones fainter, with the hand paths. --gif: an animation.
"""
import argparse
import json

import cv2
import numpy as np

SCALE = 0.5
KPT = 0.3
SMOOTH = 15                 # frames in the running median of the hand paths
POSES = 4                   # skeletons in the still image
BODY = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
FACE = range(5)             # nose, eyes, ears: drawn as one head
FEET = [(15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)]   # whole-body pose only (points 17-22)
BONE, FOOT, JOINT = (0, 200, 255), (80, 220, 80), (40, 40, 220)       # BGR, as in the DeepX preview video
HEAD_FILL = (170, 235, 255)
ROLE = {"start": (80, 190, 80), "hand": (230, 170, 40), "finish": (200, 70, 200), "foot": (40, 140, 240)}


def load_pose(path):
    """Per frame: body points (x, y, score) of the one climber, [23, 3] with feet from whole-body pose else [17, 3],
    or None."""
    z = np.load(path, allow_pickle=True)
    if "scores" in z:
        return [None if np.isnan(b[0]) else np.c_[k[:23], s[:23]] for b, k, s in zip(z["boxes"], z["kps"], z["scores"])]
    out, last = [], None
    for b, k in zip(z["boxes"], z["kps"]):
        b, k = np.asarray(b).reshape(-1, 5), np.asarray(k).reshape(-1, 17, 3)
        if not len(b):
            out.append(None)
            continue
        c = (b[:, :2] + b[:, 2:4]) / 2
        i = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))) if last is None else \
            int(np.argmin(np.linalg.norm(c - last, axis=1)))
        last = c[i]
        out.append(k[i])
    return out


def base(map_dir):
    """White canvas with every hold as a light grey shape and the lit problem's holds in their colours."""
    m = json.load(open(f"{map_dir}/map.json"))
    ids = cv2.imread(f"{map_dir}/holds.png", cv2.IMREAD_UNCHANGED)
    W, H = m["frame"]
    img = np.full((int(H * SCALE), int(W * SCALE), 3), 255, np.uint8)
    small = cv2.resize(ids, img.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    img[small > 0] = (225, 225, 225)
    for h in m["holds"]:
        if h.get("role"):
            img[small == h["id"]] = ROLE[h["role"]]
    pts = sorted((int(x), y) for x, y in m["floor_y"].items() if y is not None)
    cv2.polylines(img, [(np.array(pts) * SCALE).astype(np.int32)], False, (190, 190, 190), 2, cv2.LINE_AA)
    y0 = max(0, int((min(m["wall_top_y"].values()) - 0.05 * H) * SCALE))      # crop to the wall, with a margin
    y1 = min(img.shape[0], int((max(y for _, y in pts) + 0.12 * H) * SCALE))
    return img, (y0, y1)


def head(img, k, p):
    """A filled circle over the visible face points, sized from the shoulders, with a neck to the shoulders."""
    face = [i for i in FACE if k[i, 2] > KPT]
    if not face:
        return
    c = p[face].mean(0)
    shoulders = k[5, 2] > KPT and k[6, 2] > KPT
    r = int(max(9, 0.32 * np.linalg.norm(p[5] - p[6]) if shoulders else 0, 0.7 * np.ptp(p[face], 0).max()))
    if shoulders:
        neck = (p[5] + p[6]) / 2
        d = neck - c
        start = c + d / max(np.linalg.norm(d), 1) * r       # from the head's edge, not its centre
        cv2.line(img, tuple(start.astype(int)), tuple(neck.astype(int)), BONE, 3, cv2.LINE_AA)
    cv2.circle(img, tuple(c.astype(int)), r, HEAD_FILL, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(c.astype(int)), r, BONE, 3, cv2.LINE_AA)


def skeleton(img, k):
    """Bones amber, feet green, joints red: the DeepX preview's look, with the face as a round head."""
    p = (k[:, :2] * SCALE).astype(int)
    for links, colour in ((BODY, BONE), (FEET, FOOT)):
        for i, j in links:
            if j < len(k) and k[i, 2] > KPT and k[j, 2] > KPT:
                cv2.line(img, tuple(p[i]), tuple(p[j]), colour, 3, cv2.LINE_AA)
    head(img, k, p)
    for i in range(5, len(k)):
        if k[i, 2] > KPT:
            cv2.circle(img, tuple(p[i]), 4, JOINT, -1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map_dir")
    ap.add_argument("pose")
    ap.add_argument("out")
    ap.add_argument("--gif")
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=1e9)
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()

    pose = load_pose(a.pose)
    frames = [i for i in range(len(pose)) if a.t0 <= i / a.fps <= a.t1 and pose[i] is not None]
    img, (y0, y1) = base(a.map_dir)
    for wrist in (9, 10):                              # hand paths, a running median over SMOOTH frames (raw points jitter)
        pts = np.array([pose[i][wrist, :2] for i in frames if pose[i][wrist, 2] > KPT])
        if len(pts) > SMOOTH:
            pts = np.array([np.median(pts[max(0, j - SMOOTH // 2):j + SMOOTH // 2 + 1], axis=0) for j in range(len(pts))])
            cv2.polylines(img, [(pts * SCALE).astype(np.int32)], False, (200, 200, 200), 2, cv2.LINE_AA)
    # moments evenly spread in height gained, not in time (a climber can spend half the climb low on the wall)
    high = np.minimum.accumulate([np.nanmin([pose[i][w, 1] if pose[i][w, 2] > KPT else np.nan for w in (9, 10)] + [1e9])
                                  for i in frames])
    picks = [frames[int(np.argmax(high <= q))] for q in np.linspace(high[0], high[-1], POSES)]
    for n, i in enumerate(picks):                      # earlier moments fainter
        layer = img.copy()
        skeleton(layer, pose[i])
        alpha = 0.3 + 0.7 * n / (len(picks) - 1)
        img = cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)
    cv2.imwrite(a.out, img[y0:y1])

    if a.gif:
        from PIL import Image
        clean = base(a.map_dir)[0]
        step = max(1, int(round(a.fps / 10)))
        shots = []
        for i in range(frames[0], frames[-1] + 1, step):
            f = clean.copy()
            if pose[i] is not None:
                skeleton(f, pose[i])
            f = cv2.resize(f[y0:y1], None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
            shots.append(Image.fromarray(f[..., ::-1]).quantize(colors=128, method=Image.Quantize.MEDIANCUT,
                                                                dither=Image.Dither.NONE))
        shots[0].save(a.gif, save_all=True, append_images=shots[1:], duration=100, loop=0, optimize=True)
    print(f"{len(frames)} frames with a climber; wrote {a.out}" + (f" and {a.gif}" if a.gif else ""))


if __name__ == "__main__":
    main()
