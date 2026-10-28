"""Per-wall setup for gym pattern of life: the empty wall, where the floor is, and the holds grouped into routes.

Problem: counting what happens on a bouldering wall (which route, how high, topped or not) needs to know where the
holds are and which belong together. In commercial gyms a route is set in one hold colour, so routes are hold-colour
groups. This runs once per camera view, from a video of normal use (people are removed), not per climb.

  1. empty wall: masked median of frames, people boxes (detector conf >= 0.1) left out; computed strip by strip
  2. segment everything (SAM 2.1) on the empty wall: the floor mat is the mask covering most of the bottom rows,
     the wall the largest remaining one; the smaller masks inside the wall are holds
  3. small holds SAM misses (screw-ons): pixels inside the wall far from the wall's own colour (Lab), as connected
     components not already covered
     Blobs touching a hold (its shadow) and long thin ones (panel seams) are dropped.
  4. shapes larger than VOLUME of the wall area are volumes: shared by every route, not a route colour
  5. hold colour = median Lab of its most saturated half (the lit side, not the shadowed one); clustered with
     k-means on lightness and hue x chroma; k picked by silhouette score

    python gym_map.py VIDEO OUT_DIR [--every 30]
writes wall.jpg (empty wall), holds.png (uint16 hold ids, 0 = none), map.json, map_preview.jpg
"""
import argparse
import json
import os

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ultralytics import SAM, YOLO

MODELS = os.environ.get("MODELS_DIR", "models")
DET = os.environ.get("DET_WEIGHTS", f"{MODELS}/yolo11m-pose.pt")
SAM_WEIGHTS = os.environ.get("SAM_WEIGHTS", f"{MODELS}/sam2.1_b.pt")
VOLUME = 0.02               # share of the wall area above which a shape is a volume (big jugs are ~0.5-1.5%)
MIN_HOLD = 1.5e-5           # share of the frame area below which a blob is a bolt hole or noise
COLOUR_DIST = 28            # Lab distance from the local wall colour that counts as "not wall"
K_RANGE = range(4, 11)      # route colours to try


def empty_wall(video, every):
    det = YOLO(DET)
    cap = cv2.VideoCapture(video)
    stack, masks = [], []
    for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            continue
        m = np.zeros(f.shape[:2], bool)
        for b in det.predict(f, imgsz=1280, conf=0.1, verbose=False)[0].boxes.xyxy.cpu().numpy():
            w, h = b[2] - b[0], b[3] - b[1]
            m[max(0, int(b[1] - .15 * h)):int(b[3] + .15 * h), max(0, int(b[0] - .15 * w)):int(b[2] + .15 * w)] = True
        stack.append(f); masks.append(m)
    S, M = np.array(stack), np.array(masks)
    wall = np.zeros(S.shape[1:], np.uint8); seen = np.zeros(S.shape[1:3], bool)
    for y in range(0, S.shape[1], 64):           # strip by strip: a whole-frame float stack runs WSL out of memory
        s = S[:, y:y + 64].astype(np.float32); s[M[:, y:y + 64]] = np.inf
        s.sort(axis=0); cnt = (~M[:, y:y + 64]).sum(0)
        k = np.clip((cnt - 1) // 2, 0, None)[None, ..., None]
        med = np.take_along_axis(s, np.broadcast_to(k, (1,) + s.shape[1:]), axis=0)[0]
        med[cnt == 0] = 0
        wall[y:y + 64] = med.astype(np.uint8); seen[y:y + 64] = cnt > 0
    return wall, seen, len(S)


def regions(masks, shape):
    """Floor mat, wall (filled, holds included) and the hold-sized masks inside it."""
    H, W = shape
    bottom = [m[int(0.9 * H):].mean() for m in masks]
    mat = masks[int(np.argmax(bottom))]
    rest = [i for i in range(len(masks)) if masks[i] is not mat]
    wall_i = max(rest, key=lambda i: masks[i].sum())
    cnts, _ = cv2.findContours(masks[wall_i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wall = np.zeros((H, W), np.uint8)
    cv2.drawContours(wall, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
    wall = wall.astype(bool) & ~mat
    holds = [masks[i] for i in rest if i != wall_i and (masks[i] & wall).sum() > 0.8 * masks[i].sum()
             and masks[i].sum() < 0.05 * wall.sum()]
    return mat, wall, holds


def small_holds(img, wall, taken):
    """Blobs inside the wall far from the wall's local colour, not already a hold."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = lab.copy(); bg[~wall | taken] = np.nan
    k = 61                                                  # local wall colour: median over a wide window
    small = cv2.resize(np.nan_to_num(bg, nan=-1), None, fx=0.125, fy=0.125, interpolation=cv2.INTER_NEAREST)
    valid = (small >= 0).all(-1)
    local = np.zeros_like(small)
    for c in range(3):
        ch = np.where(valid, small[..., c], np.median(small[..., c][valid]))
        local[..., c] = cv2.medianBlur(ch.astype(np.uint8), k // 8 * 2 + 1)
    local = cv2.resize(local, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = np.linalg.norm(lab - local, axis=-1) > COLOUR_DIST
    diff &= wall & ~cv2.dilate(taken.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)   # shadows
    diff = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(diff)
    min_px = MIN_HOLD * img.shape[0] * img.shape[1]
    return [lbl == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]


def thin(m):
    """Long and thin: a panel seam or wall edge, not a hold."""
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (_, _), (w, h), _ = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    return max(w, h) > 5 * max(min(w, h), 1)


def hold_colour(lab_px):
    """Median Lab of the more saturated half of a hold's pixels: its lit side."""
    chroma = np.hypot(lab_px[:, 1] - 128., lab_px[:, 2] - 128.)
    return np.median(lab_px[chroma >= np.median(chroma)], axis=0)


def colour_name(lab):
    L, a, b = lab[0] * 100 / 255, lab[1] - 128, lab[2] - 128
    chroma, hue = np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360
    if chroma < 12:
        return "black" if L < 30 else "grey" if L < 75 else "white"
    names = [(20, "pink"), (45, "red"), (70, "orange"), (100, "yellow"), (160, "green"), (300, "blue"),
             (340, "purple"), (355, "pink"), (360, "red")]
    name = next(n for h, n in names if hue < h)
    return ("dark " if L < 25 and name != "purple" else "") + name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    img, seen, n = empty_wall(a.video, a.every)
    cv2.imwrite(f"{a.out}/wall.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cache = f"{a.out}/sam_masks.npz"                # SAM takes ~4 min on the wall: kept for re-runs
    if os.path.exists(cache):
        masks = list(np.load(cache)["m"])
    else:
        masks = list(SAM(SAM_WEIGHTS)(img, verbose=False)[0].masks.data.cpu().numpy().astype(bool))
        np.savez_compressed(cache, m=np.array(masks))
    mat, wall, holds = regions(masks, img.shape[:2])
    taken = np.any(holds, axis=0) if holds else np.zeros(img.shape[:2], bool)
    extra = small_holds(img, wall, taken)
    holds = sorted([m for m in holds + extra if not thin(m)], key=lambda m: -m.sum())
    n_extra = sum(any(m is e for e in extra) for m in holds)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    ids = np.zeros(img.shape[:2], np.uint16)
    info = []
    wall_px = wall.sum()
    for i, m in enumerate(holds, 1):
        m = m & (ids == 0)
        if not m.any():
            continue
        ids[m] = i
        ys, xs = np.nonzero(m)
        info.append({"id": i, "x": float(xs.mean()), "y": float(ys.mean()), "area": int(m.sum()),
                     "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                     "lab": hold_colour(lab[m].astype(float)).tolist(), "volume": bool(m.sum() > VOLUME * wall_px)})

    route_holds = [h for h in info if not h["volume"]]
    X = np.array([[0.5 * h["lab"][0], h["lab"][1] - 128, h["lab"][2] - 128] for h in route_holds])
    best = max(K_RANGE, key=lambda k: silhouette_score(X, KMeans(k, n_init=10, random_state=0).fit_predict(X)))
    km = KMeans(best, n_init=10, random_state=0).fit(X)
    groups = []
    for g in range(best):
        members = [h for h, l in zip(route_holds, km.labels_) if l == g]
        centre = np.median([h["lab"] for h in members], axis=0)
        groups.append({"group": g, "name": colour_name(centre), "lab": centre.tolist(), "holds": len(members),
                       "top_y": float(min(h["y"] for h in members))})
        for h in members:
            h["group"] = g
    for h in info:
        h.setdefault("group", None)

    cols = np.nonzero(wall.any(0))[0]
    floor = {int(x): int(np.argmax(mat[:, x])) if mat[:, x].any() else None for x in range(0, img.shape[1], 8)}
    top = {int(x): int(np.argmax(wall[:, x])) for x in cols[::8]}
    json.dump({"frame": [img.shape[1], img.shape[0]], "frames_used": n, "unseen": float(1 - seen.mean()),
               "floor_y": floor, "wall_top_y": top, "groups": groups, "holds": info, "k_silhouette": best},
              open(f"{a.out}/map.json", "w"), indent=1)
    cv2.imwrite(f"{a.out}/holds.png", ids)

    prev = (0.45 * img).astype(np.uint8)
    rng = np.random.default_rng(1)
    gcol = {g["group"]: tuple(int(v) for v in rng.integers(60, 255, 3)) for g in groups}
    for h in info:
        m = (ids == h["id"]).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        col = (255, 255, 255) if h["volume"] else gcol[h["group"]]
        cv2.drawContours(prev, cnts, -1, col, 2 if h["volume"] else 3)
        if not h["volume"]:
            cv2.putText(prev, str(h["group"]), (int(h["x"]) - 6, int(h["y"]) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        col, 2)
    pts = [(x, y) for x, y in floor.items() if y is not None]
    cv2.polylines(prev, [np.array(pts, np.int32)], False, (0, 255, 0), 3)
    cnts, _ = cv2.findContours(wall.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(prev, cnts, -1, (0, 200, 255), 2)
    y = 40
    for g in groups:
        cv2.putText(prev, f"{g['group']}: {g['name']} ({g['holds']})", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    gcol[g["group"]], 2)
        y += 38
    cv2.imwrite(f"{a.out}/map_preview.jpg", prev, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(info)} holds ({sum(h['volume'] for h in info)} volumes, {n_extra} from colour), "
          f"{best} colour groups: " + ", ".join(f"{g['name']} {g['holds']}" for g in groups))


if __name__ == "__main__":
    main()
