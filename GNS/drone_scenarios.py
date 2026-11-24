"""Scenarios for the data generator and the closed-loop test: the reference the
drone flies, written as pad-center waypoints in the wall frame (distance d from
the wall, along-wall coordinates a/b) plus a push-force profile, smoothed with
min-jerk segments. Nothing here depends on the simulator.

For a fully actuated drone the reference also carries body roll/pitch targets
(add_attitude_targets), so the training data sees the drone tilted while it
holds position; the attitude is an independent input for such a drone.
"""

import math

import numpy as np

PHASES = {"free": 0, "approach": 1, "hold": 2, "slide": 3, "release": 4, "hover_near": 5}


class Reference:
    """Segments of (duration, target, phase); targets are d (pad-center distance
    from the wall along n), a/b (along-wall coordinates), F (push force, N),
    Fa/Fb (sideways push along t1/t2, N). Each segment is a min-jerk move from
    the previous target. A target with an "arc" entry (center_a, center_b,
    radius, start angle, swept angle) moves a/b along that circle instead of a
    straight line, with the same min-jerk timing."""

    def __init__(self, start, segments):
        self.start = dict(start)
        self.t, self.vals, self.phase, self.arcs = [0.0], [dict(start)], [], []
        for dur, tgt, ph in segments:
            tgt = dict(tgt)
            self.arcs.append(tgt.pop("arc", None))
            v = dict(self.vals[-1])
            v.update(tgt)
            self.t.append(self.t[-1] + dur)
            self.vals.append(v)
            self.phase.append(PHASES[ph])
        self.T = self.t[-1]

    def __call__(self, t):
        t = min(max(t, 0.0), self.T - 1e-9)
        k = int(np.searchsorted(self.t, t, side="right") - 1)
        tau = (t - self.t[k]) / (self.t[k + 1] - self.t[k])
        s = tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)
        a, b = self.vals[k], self.vals[k + 1]
        out = {key: a[key] + (b[key] - a[key]) * s for key in a}
        if self.arcs[k] is not None:
            ca, cb, radius, th0, dth = self.arcs[k]
            th = th0 + dth * s
            out["a"], out["b"] = ca + radius * math.cos(th), cb + radius * math.sin(th)
        return out, self.phase[k]


# Builds a randomized scenario. r_c = pad contact-sphere radius: a pad-center
# distance of r_c means the pad is just touching (if it is flat to the wall).
# All ranges come from gs (GenSettings). Returns the start values and the list of segments.
def make_scenario(kind, rng, r_c, duration, gs, yaw_force_cap=None):
    U = rng.uniform
    logU = lambda lo, hi: math.exp(U(math.log(lo), math.log(hi)))     # log-uniform
    start = dict(d=U(0.25, 0.6), a=U(-0.3, 0.3), b=U(-0.3, 0.3), F=0.0, Fa=0.0, Fb=0.0)
    cur, seg = dict(start), []

    def add(dur, phase, **tgt):
        seg.append((dur, tgt, phase))
        cur.update(tgt)

    def approach(d_target, speed):
        add(max(abs(cur["d"] - d_target) / speed, 0.3), "approach", d=d_target)

    add(U(0.3, 0.6), "free")                                          # settle
    if kind == "free":
        while sum(sg[0] for sg in seg) < duration:
            add(U(0.8, 2.0), "free", d=U(0.4, 1.2), a=U(-0.6, 0.6), b=U(-0.4, 0.4))
    elif kind == "hover_near":
        # Dense hovering just off the wall (gate partly open, no force), plus a
        # few farther standoffs. d is the pad-center distance, so with a yaw
        # offset a ring node can touch on the closest hovers; that is fine.
        near = r_c + U(*gs.hover_near_gap, size=gs.hover_near_count)
        far = U(*gs.hover_far_dist, size=int(rng.integers(0, gs.hover_far_max + 1)))
        for d in sorted(np.r_[near, far], reverse=True):
            add(U(0.6, 1.2), "approach", d=d)
            add(U(0.8, 1.5), "hover_near")
        add(U(0.6, 1.0), "release", d=U(0.3, 0.6))
    elif kind == "tap":
        # Impacts: fly in with no push, touch briefly, back off.
        for _ in range(int(rng.integers(1, 3))):
            approach(r_c - U(0.0, 0.01), U(*gs.tap_approach_speed))
            add(U(0.1, 0.4), "hold")
            add(U(0.5, 1.0), "release", d=U(0.15, 0.4))
    elif kind in ("push", "slide", "arc"):
        # Pushing is limited by yaw authority. Horizontal friction at the pad,
        # and the normal force when the rod is yawed off the wall normal, act
        # on the rod's ~0.4 m lever arm, and a quad's yaw torque is only
        # k_m/k_f ~ 1.6 cm times the thrust differential. The yawed-rod moment
        # also grows with the yaw error (pushing a stick against a wall), so
        # once yaw saturates it runs away. yaw_force_cap(c) is the largest push
        # that keeps both inside the yaw budget when the horizontal tangential
        # load is c * mu * F.
        slides = []
        if kind == "slide":
            for _ in range(int(rng.integers(1, gs.slide_count_max + 1))):
                speed = logU(*gs.slide_speed)
                slides.append((U(0, 2 * math.pi), min(U(*gs.slide_dist), gs.slide_max_time * speed), speed))
        # Sideways-loaded hold: (direction in the wall plane, load as a fraction of mu * F)
        side = (U(0, 2 * math.pi), U(*gs.side_load_frac)) if U() < gs.side_load_prob else None

        f_max = gs.push_force[1]
        if yaw_force_cap is not None:
            caps = [yaw_force_cap(0.0)] + [yaw_force_cap(abs(math.cos(a))) for a, _, _ in slides]
            if kind == "arc":
                caps.append(yaw_force_cap(1.0))                       # friction turns through every direction
            if side is not None:
                caps.append(yaw_force_cap(side[1] * abs(math.cos(side[0]))))
            f_max = min([f_max] + caps)
        f_min = min(gs.push_force[0], 0.5 * f_max)

        approach(r_c - U(0.0, 0.005), logU(*gs.push_approach_speed))
        add(U(0.3, 0.6), "hold", F=logU(f_min, f_max))                # ramp the push in
        add(U(0.5, 1.5), "hold", F=logU(f_min, f_max))                # vary it while holding

        # Load the friction sideways without (or barely) sliding, then unload.
        if side is not None:
            ang, frac = side
            f_t = frac * gs.mu * cur["F"]
            add(U(*gs.side_load_time), "hold", Fa=f_t * math.cos(ang), Fb=f_t * math.sin(ang))
            add(U(0.3, 0.6), "hold", Fa=0.0, Fb=0.0)

        for ang, dist, speed in slides:
            add(dist / speed, "slide", a=cur["a"] + dist * math.cos(ang),
                b=cur["b"] + dist * math.sin(ang))

        if kind == "arc":
            radius, speed = U(*gs.arc_radius), U(*gs.arc_speed)
            th0 = U(0, 2 * math.pi)                                   # angle of the current point on the circle
            dth = rng.choice([-1.0, 1.0]) * 2 * math.pi * U(*gs.arc_laps)
            ca, cb = cur["a"] - radius * math.cos(th0), cur["b"] - radius * math.sin(th0)
            end = dict(a=ca + radius * math.cos(th0 + dth), b=cb + radius * math.sin(th0 + dth))
            add(abs(dth) * radius / speed, "slide", arc=(ca, cb, radius, th0, dth), **end)

        add(U(0.3, 0.6), "release", F=0.0)
        add(U(0.6, 1.0), "release", d=U(0.2, 0.5))
    else:
        raise ValueError(kind)
    add(U(0.3, 0.6), "free")
    return start, seg


# In-plane wall axes: t1 horizontal, t2 "up the wall".
def wall_tangents(n):
    t1 = np.cross(n, [0.0, 0.0, 1.0])
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(t1, n)


# Feedforward force the controller adds for the reference r: F into the wall,
# plus the sideways load (Fa along t1, Fb along t2) of the loaded holds.
def push_force_vector(r, n, t1, t2):
    return -n * r["F"] + t1 * r["Fa"] + t2 * r["Fb"]


# Adds a random body roll/pitch target (uniform within +-max_deg) to every
# segment, starting level. Reference interpolates them with the same min-jerk
# timing as the waypoints. Returns new (start, segments).
def add_attitude_targets(start, segments, rng, max_deg):
    lim = math.radians(max_deg)
    start = dict(start, roll=0.0, pitch=0.0)
    out = [(dur, dict(tgt, roll=rng.uniform(-lim, lim), pitch=rng.uniform(-lim, lim)), ph)
           for dur, tgt, ph in segments]
    return start, out


# Desired attitude: the level heading frame R_h tilted by body roll and pitch
# (R = R_h Ry(pitch) Rx(roll)). A reference without attitude targets is level.
def reference_attitude(R_h, r):
    if "roll" not in r:
        return R_h
    cr, sr, cp, sp = math.cos(r["roll"]), math.sin(r["roll"]), math.cos(r["pitch"]), math.sin(r["pitch"])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return R_h @ Ry @ Rx


# Stretches segments whose min-jerk move would need more than a_max (m/s^2) at
# the COM. A min-jerk move of length L over time T peaks at 5.77 L / T^2. com_of
# maps a waypoint dict to the COM position it asks for (it includes the attitude
# target, which swings the COM around the pad). Arcs use at least their arc
# length. Only ever lengthens a segment. Returns the new segments.
def limit_segment_accel(start, segments, com_of, a_max):
    prev, out = dict(start), []
    for dur, tgt, ph in segments:
        cur = dict(prev)
        cur.update({k: v for k, v in tgt.items() if k != "arc"})
        length = float(np.linalg.norm(com_of(cur) - com_of(prev)))
        if tgt.get("arc") is not None:
            length = max(length, abs(tgt["arc"][2] * tgt["arc"][4]))
        out.append((max(dur, math.sqrt(5.77 * length / a_max)), tgt, ph))
        prev = cur
    return out


# Largest push (N) that keeps the pad's yaw moment inside the yaw budget when the
# horizontal tangential load is c_h * mu * F (see make_scenario). lever: pad
# center's horizontal distance from the COM; yo: the heading's yaw offset (rad).
def yaw_force_cap(yaw_budget, lever, mu, yo):
    return lambda c_h: yaw_budget / (lever * (mu * c_h + abs(math.sin(yo))) + 1e-9)
