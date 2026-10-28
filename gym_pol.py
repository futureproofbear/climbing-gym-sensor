"""Gym pattern of life: climbs, routes and wall use from per-frame pose, against a wall map from gym_map.py.

Problem: a gym wants to know how its walls are used (which routes are tried and topped, how long climbs take, how
busy the wall is over the day) without filming who is climbing. Pose (boxes + 17 keypoints per person) is enough, and
it is what the DeepX NPU produces on the Pi. This turns it into climbs and counts; no image leaves this step.

Per frame, for each tracked person (the footfall Tracker, tracker.py):
  where       feet = the lowest visible ankle (else knee, else box bottom); height = feet above the floor line at
              that x, as a share of the wall's height there. People whose box centre is outside the wall region
              (e.g. someone standing behind the top of the wall) are not counted.
  on the wall height > ON_WALL for HOLD_S; back on the floor below OFF_WALL for HOLD_S (hysteresis); on an LED
              board (board_map.py) a hand on a lit start hold also counts, so a sit start begins the climb
  contacts    a hand (wrist pushed out along the forearm) or foot still for STILL_S is on the nearest hold within
              REACH of the frame width; the hold map is holds.png
A climb runs from leaving the floor to returning to it. Duplicate boxes of one person are dropped; a track lost on
the wall is continued by a new track that appears on the wall nearby within LINK_S. Per climb:
  route       the colour group with most distinct holds touched, a hand hold counting twice a foot hold (feet often
              use any nearby hold; volumes and unmatched contacts don't vote), with the share of the vote that agrees
  high point  highest hand, as a share of the wall height
  topped      both hands on the group's highest hold, or one hand there for TOP_S
  end         dropped off (feet fall > DROP wall heights within DROP_S) or climbed down; lost if the track ends on
              the wall and nobody continues it; in progress at the end of the input
Aggregates per minute: people on the wall and in front of it, climbs started, tops; per route: attempts, tops,
median duration; a coarse grid of hand contacts.

    python3 gym_pol.py MAP_DIR POSE.npz OUT_DIR [--fps 30]
POSE.npz: per frame `boxes` [N, 5] (x1 y1 x2 y2 conf) and `kps` [N, 17, 3], object arrays; or deepx_wholebody.py's
output (one climber box per frame, NaN if none, 133 points + `scores`), whose first 17 points are the body.
writes climbs.jsonl, summary.json, pol_preview.jpg (the empty wall with each climb's holds and hand paths)
"""
import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np

from tracker import Tracker

ON_WALL, OFF_WALL = 0.06, 0.03     # share of wall height: feet this far above the floor line are on the wall
HOLD_S = 0.5                       # s a change of state must hold
STILL, STILL_S = 0.12, 0.4         # wall heights / s and s: a hand or foot this still for this long is on a hold
                                   # (loose enough for raw NPU keypoints, which jitter by 20-40 px on a 1080p frame)
REACH = 0.02                       # share of frame width: a contact this close to a hold is on it
TOP_S = 1.0                        # s a hand must stay on the route's highest hold to count as topped
KPT = 0.3
DUP = 0.6                          # a box overlapping a higher-scored one this much (IoU or containment) is a duplicate
LOST_S = 0.5                       # s unseen before a climb is set aside as lost
MIN_CLIMB_S = 1.5                  # shorter climbs with no hold touched are not counted (a box flickering on the wall)
DROP = 0.3                         # wall heights the feet fall within DROP_S at the end: dropped off, not climbed down
DROP_S = 0.7
LINK_S, LINK_DIST = 10.0, 1.5      # a lost climb is continued by a new on-wall track within this long and this close
GRID = (8, 6)                      # contact grid over the wall, columns x rows
MIN_GROUP = 3                      # per-minute counts below this are reported as 0 when --deploy (k-anonymity)


def per_second_max(rows):
    """Busiest second in a minute, each second's count its median over frames (one flickering box doesn't count)."""
    by = defaultdict(list)
    for sec, n in rows:
        by[sec].append(n)
    return int(max(np.median(v) for v in by.values()))


class WallMap:
    def __init__(self, d):
        m = json.load(open(f"{d}/map.json"))
        self.W, self.H = m["frame"]
        xs = sorted(int(x) for x, y in m["floor_y"].items() if y is not None)
        self.floor = np.interp(np.arange(self.W), xs, [m["floor_y"][str(x)] for x in xs])
        xs = sorted(int(x) for x in m["wall_top_y"])
        self.top = np.interp(np.arange(self.W), xs, [m["wall_top_y"][str(x)] for x in xs])
        self.x0, self.x1 = xs[0], xs[-1]
        self.wall_h = float(np.median((self.floor - self.top)[self.x0:self.x1 + 1]))   # px, for speeds
        self.ids = cv2.imread(f"{d}/holds.png", cv2.IMREAD_UNCHANGED)
        self.holds = {h["id"]: h for h in m["holds"]}
        self.groups = {g["group"]: g for g in m["groups"]}
        self.top_hold = {}                         # group -> its highest hold
        for h in m["holds"]:
            g = h["group"]
            if g is not None and (g not in self.top_hold or h["y"] < self.holds[self.top_hold[g]]["y"]):
                self.top_hold[g] = h["id"]
        for g in m["groups"]:
            if g.get("finish"):                    # an LED board's lit finish hold (board_map.py)
                self.top_hold[g["group"]] = g["finish"]
        self.start_holds = {h["id"] for h in m["holds"] if h.get("role") == "start"}
        self.lit = {h["id"] for h in m["holds"] if h.get("role")}
        self.wall_img = cv2.imread(f"{d}/wall.jpg")

    def height(self, x, y):
        """Share of the wall height a point is above the floor line (0 = floor, 1 = top of the wall)."""
        xi = int(np.clip(x, 0, self.W - 1))
        return (self.floor[xi] - y) / max(self.floor[xi] - self.top[xi], 1.0)

    def inside(self, x, y):
        return self.x0 <= x <= self.x1 and -0.05 <= self.height(x, y) <= 1.0

    def hold_at(self, x, y):
        r = int(REACH * self.W)
        xi, yi = int(x), int(y)
        win = self.ids[max(0, yi - r):yi + r + 1, max(0, xi - r):xi + r + 1]
        if not win.any():
            return None
        ys, xs = np.nonzero(win)
        d = np.hypot(xs + max(0, xi - r) - x, ys + max(0, yi - r) - y)
        return int(win[ys[np.argmin(d)], xs[np.argmin(d)]])


def dedupe(boxes):
    """{index: box} without duplicate detections of one person (the NPU's NMS leaves some)."""
    keep = {}
    for i in np.argsort([-b[4] for b in boxes]):
        b = boxes[i]
        dup = False
        for q in keep.values():
            x0, y0 = max(b[0], q[0]), max(b[1], q[1])
            x1, y1 = min(b[2], q[2]), min(b[3], q[3])
            inter = max(0, x1 - x0) * max(0, y1 - y0)
            ab, aq = (b[2] - b[0]) * (b[3] - b[1]), (q[2] - q[0]) * (q[3] - q[1])
            if inter / (ab + aq - inter) > DUP or inter / max(min(ab, aq), 1) > 0.85:
                dup = True
                break
        if not dup:
            keep[int(i)] = b
    return keep


def feet_point(k, box):
    for idx in ((15, 16), (13, 14)):
        v = [k[i] for i in idx if k[i, 2] > KPT]
        if v:
            p = max(v, key=lambda q: q[1])
            return float(p[0]), float(p[1])
    return float((box[0] + box[2]) / 2), float(box[3])


def hand_points(k):
    """Hands: wrist pushed out 35% of the forearm (the hold is in the palm, past the wrist)."""
    out = {}
    for name, (e, w) in {"left hand": (7, 9), "right hand": (8, 10)}.items():
        if k[w, 2] > KPT:
            p = k[w, :2] + (0.35 * (k[w, :2] - k[e, :2]) if k[e, 2] > KPT else 0)
            out[name] = (float(p[0]), float(p[1]))
    for name, a in {"left foot": 15, "right foot": 16}.items():
        if k[a, 2] > KPT:
            out[name] = (float(k[a, 0]), float(k[a, 1]))
    return out


class Person:
    def __init__(self):
        self.on_wall = False
        self.since = None              # time the candidate state change started
        self.climb = None
        self.limbs = defaultdict(list)  # limb -> recent (t, x, y)
        self.held = {}                 # limb -> (hold id, since t)
        self.last = None               # (t, cx, cy, box height) last seen
        self.feet = []                 # (t, feet height) over the last 2 s


class GymPoL:
    def __init__(self, wall_map, fps):
        self.m, self.fps = wall_map, fps
        self.tracker = Tracker(frame=(wall_map.W, wall_map.H), dist_thr=1.0)
        self.people = {}
        self.climbs = []
        self.lost = []                 # ((t, cx, cy, box height) last seen, climb) for tracks lost on the wall
        self.minutes = defaultdict(lambda: {"on_wall": [], "in_front": [], "starts": 0, "tops": 0})
        self.grid = np.zeros(GRID[::-1], int)

    def update(self, t, boxes, kps):
        boxes = dedupe(boxes)
        kps = [kps[i] for i in boxes]
        boxes = [np.asarray(b) for b in boxes.values()]
        current = self.tracker.update([b[:4] for b in boxes], [b[4] for b in boxes], t, kps)
        on, front = 0, 0
        minute = self.minutes[int(t // 60)]
        for tr in current:
            k, box = np.asarray(tr.kpts), tr.box
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if not self.m.inside(cx, cy):
                continue
            fx, fy = feet_point(k, box)
            h = self.m.height(fx, fy)
            p = self.people.get(tr.uid)
            if p is None:
                p = self.people[tr.uid] = Person()
                if h > ON_WALL:                     # first seen on the wall: a climber whose track was lost
                    p.on_wall = True
                    p.climb = self._adopt(t, cx, cy, box[3] - box[1])
                    if p.climb is None:
                        p.climb = self._new_climb(tr.uid, t)
                        p.climb["seen_from"] = "wall"
            p.last = (t, cx, cy, box[3] - box[1])
            p.feet.append((t, h))
            while p.feet and p.feet[0][0] < t - 2.0:
                p.feet.pop(0)
            if not p.on_wall:
                want = h > ON_WALL or self._on_start(k)
            else:                                   # low on a board the feet are at floor level: a held hand keeps it
                want = h > OFF_WALL or self._on_start(k) or (bool(self.m.start_holds)
                                                              and any("hand" in l for l in p.held))
            if want != p.on_wall:
                p.since = p.since if p.since is not None else t
                if t - p.since >= HOLD_S:
                    start, p.on_wall, p.since = p.since, want, None
                    if want:
                        p.climb = self._new_climb(tr.uid, start)
                    else:
                        recent = [fh for ft, fh in p.feet if start - DROP_S <= ft <= start]
                        how = "dropped off" if recent and max(recent) - min(recent) > DROP else "climbed down"
                        self._finish(p, start, how)
            else:
                p.since = None
            on += p.on_wall
            front += not p.on_wall
            if p.on_wall and p.climb is not None:
                self._contacts(p, k, t, minute)
        for uid, p in list(self.people.items()):     # tracks lost on the wall: keep the climb for LINK_S
            if p.climb is not None and t - p.last[0] > LOST_S:
                self.lost.append((p.last, p.climb))
                p.climb = None
        for item in list(self.lost):
            if t - item[0][0] > LINK_S:
                self.lost.remove(item)
                self._finish_climb(item[1], item[0][0], "lost")
        minute["on_wall"].append((int(t), on))
        minute["in_front"].append((int(t), front))

    def _on_start(self, k):
        """A hand on a lit start hold (LED board maps only): a sit start, feet still at floor level, is on the wall."""
        return any(self.m.hold_at(x, y) in self.m.start_holds
                   for limb, (x, y) in hand_points(k).items() if "hand" in limb) if self.m.start_holds else False

    @staticmethod
    def _new_climb(uid, t):
        return {"person": uid, "start": round(t, 2), "holds": {}, "high": 0.0, "topped": False, "path": []}

    def _adopt(self, t, cx, cy, bh):
        """A lost climb near where a new on-wall track appeared, if any (nearest, within LINK_DIST box heights)."""
        best = None
        for item in self.lost:
            (lt, lx, ly, lh) = item[0]
            d = np.hypot(cx - lx, cy - ly) / max(bh, lh, 1.0)
            if d < LINK_DIST and (best is None or d < best[0]):
                best = (d, item)
        if best is None:
            return None
        self.lost.remove(best[1])
        best[1][1].setdefault("relinked", 0)
        best[1][1]["relinked"] += 1
        return best[1][1]

    def _contacts(self, p, k, t, minute):
        c = p.climb
        pts = hand_points(k)
        for limb, (x, y) in pts.items():
            hist = p.limbs[limb]
            hist.append((t, x, y))
            while hist and hist[0][0] < t - STILL_S:
                hist.pop(0)
            if "hand" in limb:
                c["high"] = max(c["high"], round(float(min(self.m.height(x, y), 1.0)), 3))
                c["path"].append((limb[0], round(x), round(y)))
            if len(hist) < 0.6 * STILL_S * self.fps:
                continue
            xs, ys = np.array([q[1] for q in hist]), np.array([q[2] for q in hist])
            if np.hypot(np.ptp(xs), np.ptp(ys)) / self.m.wall_h / STILL_S > STILL:
                p.held.pop(limb, None)
                continue
            hid = self.m.hold_at(float(np.median(xs)), float(np.median(ys)))
            if hid is None:
                continue
            if p.held.get(limb, (None,))[0] != hid:
                p.held[limb] = (hid, t)
                c["holds"].setdefault(str(hid), []).append(limb)
                if "hand" in limb:
                    gx = int(np.clip(x / self.m.W * GRID[0], 0, GRID[0] - 1))
                    gy = int(np.clip((1 - self.m.height(x, y)) * GRID[1], 0, GRID[1] - 1))
                    self.grid[gy, gx] += 1
            hold = self.m.holds[hid]
            other = p.held.get("right hand" if limb == "left hand" else "left hand", (None,))[0]
            if ("hand" in limb and not c["topped"] and hold["group"] is not None
                    and self.m.top_hold.get(hold["group"]) == hid
                    and (other == hid or t - p.held[limb][1] >= TOP_S)):
                c["topped"] = True
                c["topped_group"] = hold["group"]
                c["topped_at"] = round(t, 2)

    def _finish(self, p, t, how):
        if p.climb is not None:
            self._finish_climb(p.climb, t, how)
        p.climb = None
        p.held.clear()

    def _finish_climb(self, c, t, how):
        c["end"], c["status"] = round(t, 2), how
        c["duration"] = round(c["end"] - c["start"], 2)
        if c["duration"] < MIN_CLIMB_S and not c["holds"]:
            return
        votes = defaultdict(int)
        for hid, limbs in c["holds"].items():          # hands choose the route; feet often use any nearby hold
            g = self.m.holds[int(hid)]["group"]
            if g is not None:
                votes[g] += 2 if any("hand" in l for l in limbs) else 1
        if votes:
            g = max(votes, key=votes.get)
            c["route"] = {"group": g, "colour": self.m.groups[g]["name"],
                          "agree": round(votes[g] / sum(votes.values()), 2), "holds_touched": len(c["holds"])}
        else:
            c["route"] = None
        if self.m.lit:                                  # LED board: how much of the lit problem was used
            c["lit_used"] = sorted(int(h) for h in c["holds"] if int(h) in self.m.lit)
            c["lit_skipped"] = sorted(self.m.lit - set(c["lit_used"]))
        if c["topped"] and c["route"] and c["topped_group"] != c["route"]["group"]:
            c["topped"] = False                          # topped some other colour's highest hold: not this route
        self.climbs.append(c)

    def close(self, t):
        for p in self.people.values():
            self._finish(p, t, "in progress")
        for last, c in self.lost:
            self._finish_climb(c, last[0], "lost")
        self.lost = []

    def summary(self, deploy=False):
        for c in self.climbs:
            self.minutes[int(c["start"] // 60)]["starts"] += 1
            if c["topped"]:
                self.minutes[int(c["topped_at"] // 60)]["tops"] += 1
        mins = {}
        for mi, v in sorted(self.minutes.items()):
            row = {"on_wall_max": per_second_max(v["on_wall"]), "in_front_max": per_second_max(v["in_front"]),
                   "climbs_started": v["starts"], "tops": v["tops"]}
            if deploy:
                row = {key: (val if val >= MIN_GROUP else 0) for key, val in row.items()}
            mins[mi] = row
        routes = defaultdict(lambda: {"attempts": 0, "tops": 0, "durations": []})
        for c in self.climbs:
            key = c["route"]["colour"] + f" ({c['route']['group']})" if c["route"] else "unknown"
            routes[key]["attempts"] += 1
            routes[key]["tops"] += int(c["topped"])
            routes[key]["durations"].append(c["duration"])
        return {"minutes": mins,
                "routes": {k: {"attempts": v["attempts"], "tops": v["tops"],
                               "median_duration_s": float(np.median(v["durations"]))} for k, v in routes.items()},
                "climbs": len(self.climbs), "hand_contact_grid": self.grid.tolist()}

    def preview(self, path):
        img = (0.5 * self.m.wall_img).astype(np.uint8)
        for c in self.climbs:
            for hid, limbs in c["holds"].items():
                m = (self.m.ids == int(hid)).astype(np.uint8)
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                col = (60, 220, 255) if any("hand" in l for l in limbs) else (80, 220, 80)
                cv2.drawContours(img, cnts, -1, col, 4)
            for side, col in (("l", (255, 200, 120)), ("r", (120, 160, 255))):     # each hand's path, smoothed
                pts = np.array([q[1:] for q in c["path"] if q[0] == side], np.float32)
                if len(pts) > 15:
                    pts = np.stack([np.convolve(pts[:, i], np.ones(15) / 15, "valid") for i in (0, 1)], 1)
                    cv2.polylines(img, [pts[::3].astype(np.int32)], False, col, 2, cv2.LINE_AA)
        x = np.arange(self.m.W)
        cv2.polylines(img, [np.c_[x, self.m.floor].astype(np.int32)], False, (0, 255, 0), 2)
        cv2.polylines(img, [np.c_[x, self.m.floor - ON_WALL * (self.m.floor - self.m.top)].astype(np.int32)],
                      False, (0, 160, 0), 1)
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map_dir")
    ap.add_argument("pose")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--deploy", action="store_true", help="suppress per-minute counts below MIN_GROUP")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    z = np.load(a.pose, allow_pickle=True)
    boxes, kps = z["boxes"], z["kps"]
    if "scores" in z:                                   # deepx_wholebody.py: one climber, 133 points
        kps = [np.c_[k[:17], s[:17]] for k, s in zip(kps, z["scores"])]
        kps = [k if not np.isnan(b[0]) else np.zeros((0, 17, 3)) for b, k in zip(boxes, kps)]
        boxes = [b if not np.isnan(b[0]) else np.zeros((0, 5)) for b in boxes]
    pol = GymPoL(WallMap(a.map_dir), a.fps)
    for i, (b, k) in enumerate(zip(boxes, kps)):
        pol.update(i / a.fps, np.asarray(b).reshape(-1, 5), np.asarray(k).reshape(-1, 17, 3))
    pol.close(len(boxes) / a.fps)
    with open(f"{a.out}/climbs.jsonl", "w") as fh:
        for c in pol.climbs:
            fh.write(json.dumps({k: v for k, v in c.items() if k != "path"}) + "\n")
    s = pol.summary(a.deploy)
    json.dump(s, open(f"{a.out}/summary.json", "w"), indent=1)
    pol.preview(f"{a.out}/pol_preview.jpg")
    for c in pol.climbs:
        r = c["route"]
        print(f"climb {c['person']}: {c['start']:.1f}-{c['end']:.1f}s ({c['status']}), high point {c['high']:.0%}, "
              f"{len(c['holds'])} holds, route {r['colour'] + ' (' + str(r['group']) + ')' if r else '?'}"
              f"{', agree ' + format(r['agree'], '.0%') if r else ''}, topped {c['topped']}")
    print(json.dumps({k: v for k, v in s.items() if k != "hand_contact_grid"}))


if __name__ == "__main__":
    main()
