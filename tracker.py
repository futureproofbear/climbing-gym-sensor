"""People tracker for gym_pol.py: each person keeps an ID from frame to frame, and a track lost for a moment is
continued when the same person reappears nearby. Taken from the footfall counter of the edge-vision garden sensor
(ai-camera/footfall.py: Track, Tracker and the settings they use), so both count people the same way.
"""
from collections import deque
import os
import time

import numpy as np

MIN_HITS = int(os.environ.get("TRACK_MIN_HITS", 3))
MIN_SECONDS = float(os.environ.get("TRACK_MIN_SECONDS", 0.5))
MAX_AGE = float(os.environ.get("TRACK_MAX_AGE", 2.0))   # seconds a track survives without a detection
REVIVE_SECONDS = float(os.environ.get("REVIVE_SECONDS", 5.0))  # a track lost this long can still be continued
REVIVE_DIST = 1.0                                            # body heights from where it was last seen / predicted
STAY_DIST = 0.5                                              # body heights: reappeared where it was (stood still)
EDGE = 0.10            # fraction of the frame width at the left / right edges where people really enter and leave
LINK_SECONDS = float(os.environ.get("LINK_SECONDS", 12.0))   # mid-scene disappearance: wait this long for a reappearance
LINK_SPEED = (0.15, 2.0)                                     # plausible walking speed across the gap, body heights / s
MAX_SPEED = 3.0        # body heights / s: no pedestrian is faster; caps the velocity one jumpy box would imply
STAND_SECONDS = float(os.environ.get("STAND_SECONDS", 3.0))  # must stay put at least this long to be standing
STAND_MOVE = float(os.environ.get("STAND_MOVE", 0.4))        # body heights of movement still counted as standing
RUN_SPEED = float(os.environ.get("RUN_SPEED", 1.2))          # net body heights / s above which = running
LIVE_WINDOW = 4.0                                            # seconds of history behind the live label
LABEL_HOLD = float(os.environ.get("LABEL_HOLD", 1.0))        # a new live label must hold this long before it shows
HIP_MIN = 0.5                                                # hip keypoint confidence for the movement anchor
KPT_MIN = 0.6                                                # keypoint confidence for the sitting check
SIT_ASPECT = 2.0                                             # sitting box: height < 2x width (standing ~2.5-3.5)
ACTIVITIES = ("walking", "running", "standing", "sitting")
REID_MEMORY_S = float(os.environ.get("REID_MEMORY_S", 300))     # deploy mode: re-ID vectors kept this long, in memory
RETENTION_DAYS = float(os.environ.get("RETENTION_DAYS", 7))     # deploy mode: per-person detail kept this long
MIN_GROUP = int(os.environ.get("MIN_GROUP", 3))                 # deploy mode: smaller counts per bucket reported as 0



def head_pose(k, conf=0.4):
    """Head and neck from COCO-17 keypoints, which have no neck or head-centre point.

    Returns {"neck": [x, y] | None, "head": [x, y] | None, "head_r": radius px, "facing": label, "yaw": deg | None}:
      neck    midpoint of the shoulders
      head    mean of the visible face points (nose, eyes, ears); radius from the ear / eye spread
      facing  "camera" (nose and both eyes), "away" (ears but no nose or eyes), "left" / "right" (profile:
              which side of the ears the nose is, in image directions), or None when the face isn't visible
      yaw     approximate head turn, -90 (image left) .. +90 (image right), 0 = straight at the camera
    """
    if k is None:
        return {"neck": None, "head": None, "head_r": None, "facing": None, "yaw": None}
    k = np.asarray(k, float)
    vis = k[:, 2] >= conf
    neck = ((k[5, :2] + k[6, :2]) / 2).round(1).tolist() if vis[5] and vis[6] else None
    face = [i for i in range(5) if vis[i]]
    head = k[face, :2].mean(0).round(1).tolist() if face else None
    spread = float(np.ptp(k[face, 0])) if len(face) > 1 else None
    sh = float(abs(k[5, 0] - k[6, 0])) if vis[5] and vis[6] else None
    head_r = round(max(spread or 0, 0.5 * (sh or 0), 4.0) * 0.6, 1) if face else None
    facing, yaw = None, None
    nose, eyes, ears = vis[0], vis[1] and vis[2], [i for i in (3, 4) if vis[i]]
    if nose and len(ears) == 2:
        mid, half = (k[3, 0] + k[4, 0]) / 2, abs(k[3, 0] - k[4, 0]) / 2
        r = float(np.clip((k[0, 0] - mid) / max(half, 1.0), -1, 1))
        yaw = round(float(np.degrees(np.arcsin(r))), 0)
        facing = "camera" if abs(yaw) < 30 and eyes else ("right" if yaw > 0 else "left")
    elif nose and len(ears) == 1:
        facing = "right" if k[0, 0] > k[ears[0], 0] else "left"
        yaw = 70.0 if facing == "right" else -70.0
    elif nose and eyes:
        facing, yaw = "camera", 0.0
    elif ears and not nose and not (vis[1] or vis[2]):
        facing = "away"
    return {"neck": neck, "head": head, "head_r": head_r, "facing": facing, "yaw": yaw}


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, box, t, kpts=None):
        self.box = np.asarray(box, float)
        self.vel = np.zeros(2)         # centre velocity, px/s
        self.first_t = self.last_t = t
        self.hits = 1
        self.uid = None                # unique pedestrian number, set on confirmation
        self.kpts = kpts               # latest COCO-17 keypoints (x, y, conf) in frame pixels, or None
        self.hist = deque(maxlen=600)  # (t, cx, cy, h, sit) per update; sit = True / False / None (no clear pose)
        self.trace = deque(maxlen=600)  # (t, box, keypoints, live activity) per update, for the per-person record
        self.activity = None           # current label
        self._cand, self._cand_t = None, None   # live label waiting out LABEL_HOLD
        self.hip_off = None            # hips minus box centre, px, smoothed: fills in when the hips are hidden
        self._record(t)
        self.rowid = None              # footfall database row, for the activity / dwell update at the end

    def predicted(self, t):
        dx, dy = self.vel * (t - self.last_t)
        return self.box + [dx, dy, dx, dy]

    def update(self, box, t, kpts=None):
        box = np.asarray(box, float)
        dt = t - self.last_t
        if dt > 0:
            c_old = (self.box[:2] + self.box[2:]) / 2
            c_new = (box[:2] + box[2:]) / 2
            self.vel = 0.5 * self.vel + 0.5 * (c_new - c_old) / dt
            cap = MAX_SPEED * max(box[3] - box[1], 1.0)
            sp = float(np.hypot(*self.vel))
            if sp > cap:                    # a box that jumps (small far people jitter by ~0.5 body heights)
                self.vel *= cap / sp        # must not send the prediction off where the person isn't
        self.box, self.last_t, self.kpts = box, t, kpts
        self.hits += 1
        self._record(t)

    def _sitting_pose(self):
        """True / False when the pose is clear enough to judge, None otherwise."""
        b, k = self.box, self.kpts
        if k is None:
            return None
        for hip, knee in ((11, 13), (12, 14)):
            if k[hip, 2] >= KPT_MIN and k[knee, 2] >= KPT_MIN:
                dx, dy = k[knee, 0] - k[hip, 0], k[knee, 1] - k[hip, 1]
                squat = (b[3] - b[1]) < SIT_ASPECT * (b[2] - b[0])
                return bool(abs(dy) < 0.6 * abs(dx) and squat)   # thigh within ~30 deg of horizontal
        return None

    def _anchor(self):
        """Where the person is, for movement: the mid-hip point when both hips are clear, else the box centre
        plus the last hip offset. A box cut short by a bush or tree trunk moves its centre without the person
        moving (a woman carrying a coffee read "running" for two updates that way); the hips stay put."""
        b, k = self.box, self.kpts
        c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        if k is not None and k[11][2] >= HIP_MIN and k[12][2] >= HIP_MIN:
            hip = (np.asarray(k[11][:2], float) + np.asarray(k[12][:2], float)) / 2
            self.hip_off = hip - c if self.hip_off is None else 0.7 * self.hip_off + 0.3 * (hip - c)
            return hip
        return c if self.hip_off is None else c + self.hip_off

    def set_live_label(self, label, t):
        """Show a new live label only once it has held for LABEL_HOLD seconds (no one-update flickers)."""
        if self.activity is None or label == self.activity:
            self.activity, self._cand = label, None
        elif label != self._cand:
            self._cand, self._cand_t = label, t
        elif t - self._cand_t >= LABEL_HOLD:
            self.activity, self._cand = label, None

    def _record(self, t):
        b = self.box
        ax, ay = self._anchor()
        self.hist.append((t, ax, ay, max(b[3] - b[1], 1.0), self._sitting_pose()))
        k = None if self.kpts is None else np.round(np.asarray(self.kpts, float), 1).tolist()
        self.trace.append((round(t, 3), [round(float(v), 1) for v in b], k, self.activity, head_pose(self.kpts)))

    def label(self, window=None):
        """Activity over the last `window` seconds of the track (None = the whole track)."""
        h = [e for e in self.hist if window is None or e[0] >= self.last_t - window]
        if len(h) < 2:
            return "walking"
        dur = h[-1][0] - h[0][0]
        n = max(1, len(h) // 5)                            # average the ends to damp box jitter
        t0, x0, y0, h0 = (np.mean([e[i] for e in h[:n]]) for i in range(4))
        t1, x1, y1, h1 = (np.mean([e[i] for e in h[-n:]]) for i in range(4))
        height = float(np.median([e[3] for e in h]))
        moved = float(np.hypot(x1 - x0, y1 - y0)) / height  # net movement in body heights
        if dur > 0 and moved / max(t1 - t0, 1e-3) > RUN_SPEED and dur >= 1.0:
            return "running"
        still = dur >= STAND_SECONDS and moved < STAND_MOVE and abs(np.log(h1 / h0)) < 0.1
        if still:
            votes = [e[4] for e in h if e[4] is not None]
            if votes and sum(votes) >= 0.5 * len(votes):
                return "sitting"
            return "standing"
        return "walking"

    def overall_activity(self):
        return self.label()


class Tracker:
    """Greedy IoU / centre-distance matching with constant-velocity prediction.

    dist_thr: largest centre jump between updates, in person heights, that still matches
    (raise it for slow detectors, where people move further between detections).
    """

    def __init__(self, on_new=None, on_end=None, dist_thr=0.75, max_age=MAX_AGE, frame=None):
        self.frame = frame             # (width, height) of the frame, for the scene rule in _revivable
        self.tracks = []
        self.next_uid = 1
        self.on_new = on_new           # called with (track, t) once per unique pedestrian
        self.on_end = on_end           # called with (track,) when a confirmed track is dropped
        self.dist_thr, self.max_age = dist_thr, max_age
        self.ended = []                # confirmed tracks lost recently, kept for revival (see _revivable)

    def update(self, boxes, scores, t=None, kpts=None):
        t = time.time() if t is None else t
        boxes = [np.asarray(b, float) for b in boxes]
        kpts = [None] * len(boxes) if kpts is None else kpts
        pairs = []
        for ti, tr in enumerate(self.tracks):
            p = tr.predicted(t)
            ph = max(p[3] - p[1], 1.0)
            pc, lc = (p[:2] + p[2:]) / 2, (tr.box[:2] + tr.box[2:]) / 2
            for di, b in enumerate(boxes):
                # nearer of predicted and last-seen position: a wrong velocity estimate alone must not lose the
                # person (2026-10-22 15:03:43: one box 90 px too high, the next 0.17 s later back in place, a
                # second track and a second count)
                ov = max(iou(p, b), iou(tr.box, b))
                bc = (b[:2] + b[2:]) / 2
                dist = min(np.linalg.norm(bc - pc), np.linalg.norm(bc - lc)) / max(ph, b[3] - b[1], 1.0)
                if ov > 0.1 or dist < self.dist_thr:
                    pairs.append((ov - 0.5 * dist, ti, di))
        used_t, used_d = set(), set()
        for _, ti, di in sorted(pairs, reverse=True):
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti), used_d.add(di)
            self.tracks[ti].update(boxes[di], t, kpts[di])
        for di, b in enumerate(boxes):
            if di not in used_d:
                self.tracks.append(Track(b, t, kpts[di]))
        alive = []
        for tr in self.tracks:
            if t - tr.last_t <= self.max_age:
                alive.append(tr)
            elif tr.uid is not None:
                self.ended.append(tr)       # held for REVIVE_SECONDS in case the same person reappears
        self.tracks = alive
        still = []
        for tr in self.ended:               # nobody took it over: the pedestrian really left
            if t - tr.last_t > self.max_age + self._hold(tr):
                if self.on_end:
                    self.on_end(tr)
            else:
                still.append(tr)
        self.ended = still
        for i, tr in enumerate(self.tracks):
            if tr.uid is None and tr.hits >= MIN_HITS and tr.last_t - tr.first_t >= MIN_SECONDS:
                old = self._revivable(tr)
                if old is not None:         # same person after a broken track: continue it, don't count again
                    if old in self.ended:
                        self.ended.remove(old)
                    else:                   # hand-over from a coasting track
                        self.tracks = [o if o is not old else None for o in self.tracks]
                    old.box, old.last_t, old.kpts, old.vel = tr.box, tr.last_t, tr.kpts, tr.vel
                    old.hits += tr.hits
                    old.hist.extend(tr.hist)
                    old.trace.extend(tr.trace)
                    old.revived = getattr(old, "revived", 0) + 1
                    self.tracks[i] = old
                    continue
                tr.uid = self.next_uid
                self.next_uid += 1
                if self.on_new:
                    self.on_new(tr, t)
        self.tracks = [tr for tr in self.tracks if tr is not None]
        current = [tr for tr in self.tracks if tr.uid is not None and tr.last_t == t]
        for tr in current:
            tr.set_live_label(tr.label(LIVE_WINDOW), t)
        return current

    def _interior(self, x):
        """True if x is away from the left / right frame edges, i.e. not where people enter or leave the view."""
        return self.frame is not None and EDGE * self.frame[0] < x < (1 - EDGE) * self.frame[0]

    def _hold(self, tr):
        """How long a lost track waits to be continued: longer if it vanished mid-scene (behind something)."""
        return LINK_SECONDS if self._interior(tr.hist[-1][1]) else REVIVE_SECONDS

    def _revivable(self, new):
        """A recently ended track the new one continues, by either rule:

        nearby   it started within REVIVE_SECONDS of the old one's last sighting, near where it was last
                 seen (allowing for its walking speed), at a similar size;
        stayed   it started within half a body height of where the old one was last seen, within the old one's
                 hold time, at any plausible size: someone standing or working in one place who bent down,
                 crouched or was hidden by leaves for a while (23 Oct count checks: a man on his phone and two
                 gardeners were each counted again);
        scene    the old track ended mid-scene, away from the frame edges where people really come and go, so
                 the person went behind something (the trees and hedge across the main path) rather than leaving.
                 The new one starts mid-scene within LINK_SECONDS, further along the old one's direction of
                 travel, at a plausible walking speed for the gap. Without this, one walker crossing the garden
                 was counted twice (2026-10-18 09:23:56 / 09:24:05: ended at x 0.67 heading left, reappeared at
                 x 0.29 5 s later, 0.6 body heights/s)."""
        nb = new.hist[0]                                    # (t, cx, cy, h, sit) at the new track's start
        best, best_d = None, None
        # hand-over: a confirmed track still alive but not seen since this one started (it stopped matching
        # within MAX_AGE, so it isn't in self.ended yet) is the same person if the new one starts where it was
        coasting = [o for o in self.tracks if o.uid is not None and o is not new and o.last_t <= nb[0]]
        for old in self.ended + coasting:
            gap = nb[0] - old.last_t
            if not (0 <= gap <= max(REVIVE_SECONDS, self._hold(old))):
                continue
            ob = old.hist[-1]
            h = max(ob[3], nb[3])
            ratio = nb[3] / ob[3]
            d_last = np.hypot(nb[1] - ob[1], nb[2] - ob[2]) / h
            if d_last < STAY_DIST and 0.4 <= ratio <= 2.5:          # stayed where it was
                d = d_last
            elif gap <= REVIVE_SECONDS and 0.7 <= ratio <= 1.4:
                px, py = ob[1] + old.vel[0] * gap, ob[2] + old.vel[1] * gap   # where it would be by now
                d = min(d_last, np.hypot(nb[1] - px, nb[2] - py) / h)
            else:
                continue
            if d < REVIVE_DIST and (best_d is None or d < best_d):
                best, best_d = old, d
        if best is not None or not self._interior(nb[1]):
            return best
        best_cost = None
        for old in self.ended:
            gap = nb[0] - old.last_t
            ob, oa = old.hist[-1], old.hist[0]
            if not (0 < gap <= LINK_SECONDS) or not self._interior(ob[1]):
                continue
            h = float(np.median([e[3] for e in old.hist]))
            if not 0.4 <= nb[3] / h <= 2.5:                  # partly hidden people have shorter boxes
                continue
            dir_old = ob[1] - oa[1]
            if abs(dir_old) < 0.3 * h:                        # old track's direction unclear
                continue
            ahead = (nb[1] - ob[1]) * np.sign(dir_old)
            if ahead <= 0:                                    # reappeared behind where it vanished
                continue
            dir_new = new.hist[-1][1] - nb[1]
            if dir_new * np.sign(dir_old) < -0.2 * h:         # new one walks the other way
                continue
            if abs((nb[2] + nb[3] / 2) - (ob[2] + ob[3] / 2)) > 0.2 * h:   # feet level: must be the same path
                continue
            speed = np.hypot(nb[1] - ob[1], nb[2] - ob[2]) / h / gap
            if not LINK_SPEED[0] <= speed <= LINK_SPEED[1]:
                continue
            old_speed = max(abs(dir_old) / h / max(ob[0] - oa[0], 0.3), 0.2)
            cost = abs(np.log(speed / old_speed)) + abs(nb[2] - ob[2]) / h
            if best_cost is None or cost < best_cost:
                best, best_cost = old, cost
        if best is not None:
            best.scene_linked = getattr(best, "scene_linked", 0) + 1
        return best
