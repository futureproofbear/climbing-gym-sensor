"""Per-view setup for an LED training board (Kilter / Tension / Moon style): the board, its holds, and the lit problem.

Problem: gym_map.py guesses routes from hold colour, which fails when a colour is reused, and needs SAM on a GPU. On an
LED board every hold is the same pale colour on a dark board, and the problem being climbed is lit: start, hand,
finish (and foot) holds each glow in their own colour. So the map needs no model and runs on the Pi's CPU.

  1. empty board: median of frames from a stretch with nobody on it: --empty T0-T1 (seconds), or --pose POSE.npz
     (wholebody_video.py's output for this video: its longest run with nobody detected), else the whole video
  2. board = the largest dark, unsaturated region, holds filled in; floor line = its bottom edge, top = its top edge
  3. holds = pale blobs inside the board
  4. lit = bright, saturated blobs inside the board; role from hue (LIT_HUES); each joins the hold it touches
     (the glow sits on the hold's rim) or, with none, is a hold of its own
  5. the lit holds are one group, "lit problem"; its finish hold is what topping means in gym_pol

    python3 board_map.py VIDEO OUT_DIR [--empty 0.5-9.5 | --pose POSE.npz]
writes wall.jpg (empty board), holds.png (uint16 hold ids, 0 = none), map.json (gym_map.py's format, plus each
hold's "role"), map_preview.jpg
"""
import argparse
import json
import os

import cv2
import numpy as np

DARK_V, DARK_S = 110, 70    # board: HSV value and saturation below these
LIT_S, LIT_V = 100, 150     # lit: saturation and value above these
MIN_HOLD = 4e-5             # share of the frame area below which a blob is a bolt hole or noise
MAX_HOLD = 4e-3             # ... and above which it is not a hold (wall, ceiling, mat)
RING, RING_DARK = 6, 0.6    # a hold: at least this share of a RING px band around it is dark board
LINK = 2.0                  # holds closer than this x the median nearest-neighbour gap are one board
MIN_LIT = 1.5e-5
LIT_HUES = [(35, 90, "start"), (90, 125, "hand"), (125, 170, "finish"), (5, 35, "foot")]   # OpenCV hue 0-180


def empty_board(video, span, every=10):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i0, i1 = (int(span[0] * fps), min(n, int(span[1] * fps))) if span else (0, n)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    stack = []
    for i in range(i0, i1):
        ok, f = cap.read()
        if not ok:
            break
        if (i - i0) % every == 0:
            stack.append(f)
    return np.median(np.stack(stack), 0).astype(np.uint8), len(stack)


def main_cluster(keep, cen):
    """The labels in the largest group of holds linked by gaps under LINK x the median nearest-neighbour distance: the
    board's holds are packed, stray pale blobs (a sign, a ceiling corner) stand apart."""
    p = cen[keep]
    d = np.hypot(*(p[:, None] - p[None]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    adj = d < LINK * np.median(d.min(1))
    comp = -np.ones(len(keep), int)
    for s in range(len(keep)):
        if comp[s] >= 0:
            continue
        comp[s], todo = s, [s]
        while todo:
            for j in np.nonzero(adj[todo.pop()] & (comp < 0))[0]:
                comp[j] = s
                todo.append(j)
    big = np.bincount(comp).argmax()
    return [k for k, c in zip(keep, comp) if c == big]


def empty_span(pose, fps):
    """Seconds of the longest run of frames with nobody detected, from a pose file (wholebody_video.py: NaN boxes;
    gym_pol.py's format: empty box lists)."""
    z = np.load(pose, allow_pickle=True)
    empty = [bool(np.isnan(np.asarray(b, float)).any()) if np.asarray(b).ndim == 1 else len(b) == 0
             for b in z["boxes"]] + [False]
    best, start = (0, 0), None
    for i, e in enumerate(empty):
        if e and start is None:
            start = i
        elif not e and start is not None:
            best, start = max(best, (start, i), key=lambda r: r[1] - r[0]), None
    if best[1] - best[0] < fps:
        raise SystemExit(f"no stretch of 1 s with nobody on the board in {pose}; pass --empty")
    return best[0] / fps, best[1] / fps


def board_holds(hsv):
    """Hold-sized pale blobs ringed by dark board (a dark ceiling or a pale wall alone is neither), and the board as the
    convex hull of them: (hold label image, list of labels, board mask)."""
    H, W = hsv.shape[:2]
    dark = (hsv[..., 2] < DARK_V) & (hsv[..., 1] < DARK_S)
    n, lab, st, cen = cv2.connectedComponentsWithStats((~dark & (hsv[..., 1] < LIT_S)).astype(np.uint8))
    keep = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if not MIN_HOLD * H * W <= area <= MAX_HOLD * H * W:
            continue
        x0, y0, x1, y1 = max(0, x - RING), max(0, y - RING), min(W, x + w + RING), min(H, y + h + RING)
        m = lab[y0:y1, x0:x1] == i
        ring = cv2.dilate(m.astype(np.uint8), np.ones((2 * RING + 1,) * 2, np.uint8)).astype(bool) & ~m
        if dark[y0:y1, x0:x1][ring].mean() > RING_DARK:
            keep.append(i)
    keep = main_cluster(keep, cen)
    pts = np.argwhere(np.isin(lab, keep))[:, ::-1].astype(np.int32)
    board = np.zeros((H, W), np.uint8)
    cv2.drawContours(board, [cv2.convexHull(pts)], -1, 1, -1)
    return lab, keep, board.astype(bool)


def role(hue):
    return next((r for lo, hi, r in LIT_HUES if lo <= hue < hi), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--empty", help="T0-T1: seconds with nobody on the board")
    ap.add_argument("--pose", help="pose file for this video: the empty board is its longest run with nobody detected")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.empty:
        span = tuple(float(t) for t in a.empty.split("-"))
    elif a.pose:
        span = empty_span(a.pose, cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS) or 30.0)
        print(f"empty board: {span[0]:.1f}-{span[1]:.1f} s")
    else:
        span = None
    img, n = empty_board(a.video, span)
    cv2.imwrite(f"{a.out}/wall.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, W = img.shape[:2]
    lab_p, keep, board = board_holds(hsv)
    lit = cv2.dilate(board.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) \
        & (hsv[..., 1] > LIT_S) & (hsv[..., 2] > LIT_V)

    ids = np.zeros((H, W), np.uint16)
    info = []
    for i in keep:
        ids[lab_p == i] = len(info) + 1
        info.append({"id": len(info) + 1, "role": None})

    n_l, lab_l, st_l, _ = cv2.connectedComponentsWithStats(lit.astype(np.uint8))
    near = cv2.dilate(ids, np.ones((15, 15), np.uint8))   # the glow is on the rim: look just outside the hold
    for i in range(1, n_l):
        if st_l[i, cv2.CC_STAT_AREA] < MIN_LIT * H * W:
            continue
        m = lab_l == i
        r = role(float(np.median(hsv[..., 0][m])))
        if r is None:
            continue
        hit = near[m][near[m] > 0]
        if hit.size:
            hid = int(np.bincount(hit).argmax())
        else:
            info.append({"id": len(info) + 1, "role": None})
            hid = len(info)
        ids[m & ((ids == 0) | (ids == hid))] = hid
        info[hid - 1]["role"] = r

    for h in info:
        ys, xs = np.nonzero(ids == h["id"])
        h.update({"x": float(xs.mean()), "y": float(ys.mean()), "area": int(len(xs)),
                  "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], "volume": False,
                  "group": 0 if h["role"] else None})
    lit_holds = [h for h in info if h["role"]]
    finish = min((h for h in lit_holds if h["role"] == "finish"), key=lambda h: h["y"], default=None)
    groups = [{"group": 0, "name": "lit problem", "holds": len(lit_holds),
               "top_y": float(min(h["y"] for h in lit_holds)) if lit_holds else None,
               "finish": finish["id"] if finish else None,
               "roles": {r: sum(h["role"] == r for h in lit_holds) for _, _, r in LIT_HUES}}]

    cols = np.nonzero(board.any(0))[0]
    floor = {int(x): int(H - 1 - np.argmax(board[::-1, x])) for x in cols[::8]}
    top = {int(x): int(np.argmax(board[:, x])) for x in cols[::8]}
    json.dump({"frame": [W, H], "frames_used": n, "floor_y": floor, "wall_top_y": top, "groups": groups,
               "holds": info}, open(f"{a.out}/map.json", "w"), indent=1)
    cv2.imwrite(f"{a.out}/holds.png", ids)

    prev = (0.45 * img).astype(np.uint8)
    rcol = {"start": (80, 220, 80), "hand": (255, 200, 0), "finish": (220, 60, 220), "foot": (0, 140, 255), None: (150, 150, 150)}
    for h in info:
        cnts, _ = cv2.findContours((ids == h["id"]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(prev, cnts, -1, rcol[h["role"]], 4 if h["role"] else 1)
    cnts, _ = cv2.findContours(board.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(prev, cnts, -1, (0, 200, 255), 2)
    if finish:
        cv2.circle(prev, (int(finish["x"]), int(finish["y"])), 45, rcol["finish"], 3)
    y = 40
    for r, c in groups[0]["roles"].items():
        cv2.putText(prev, f"{r}: {c}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, rcol[r], 2)
        y += 38
    cv2.imwrite(f"{a.out}/map_preview.jpg", prev, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(info)} holds, {len(lit_holds)} lit: " + ", ".join(f"{r} {c}" for r, c in groups[0]["roles"].items())
          + f"; finish hold {groups[0]['finish']}")


if __name__ == "__main__":
    main()
