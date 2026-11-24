"""Controllers that fly the drone for data generation and the closed-loop test.
The model never learns the controller: it only has to make the contacts we want.

  GeometricController       quad (underactuated): SE(3) controller (Lee et al.
                            2010), thrust direction sets the attitude
  FullyActuatedController   tilted-rotor drone: world force and body torque
                            tracked independently, so it holds any attitude
                            while it tracks position; bounded 6-D allocation

make_controller() picks one from DroneConfig.fully_actuated. Both are called
the same way and return rotor thrust commands (N):
    ctrl(com, v, R, w_body, p_d, v_d, a_d, F_ff, heading, R_d=None)
R_d (desired attitude) is used only by the fully actuated controller; the quad's
attitude follows from the force it needs, at the given heading.
"""

import math

import numpy as np
from scipy.optimize import linprog, lsq_linear

from drone_config import wrench_matrix


# Per-rotor thrust limits (N): a small floor so the rotors keep spinning, and the
# config's maximum (or GenSettings.t_max_over_hover x the quad's hover thrust).
def thrust_limits(cfg, gs):
    M = len(cfg.rotor_pos)
    t_max = cfg.rotor_thrust_max or gs.t_max_over_hover * (cfg.mass * cfg.gravity / M)
    return 0.05 * cfg.mass * cfg.gravity / M, t_max


def make_controller(cfg, gs, rng):
    _, t_max = thrust_limits(cfg, gs)
    if cfg.fully_actuated:
        return FullyActuatedController(cfg, rng, t_max)
    return GeometricController(cfg, rng, t_max)


# Geometric tracking controller on SE(3) with a thrust/torque mixer.
class GeometricController:

    BASE_GAINS = dict(wp=4.0, wr=15.0, wy=4.0)       # the closed-loop test randomizes around these

    def __init__(self, cfg, rng, t_max):
        self.m, self.g = cfg.mass, cfg.gravity
        self.J = np.asarray(cfg.inertia)
        c = np.asarray(cfg.com_offset)
        rp = np.asarray(cfg.rotor_pos) - c
        s = np.asarray(cfg.spin_dir, dtype=float)
        # Mixer for (near-)vertical rotor axes: [f, Mx, My, Mz] = A @ T
        self.A = np.vstack([np.ones(len(rp)), rp[:, 1], -rp[:, 0], -s * cfg.k_m / cfg.k_f])
        self.A_inv = np.linalg.pinv(self.A)
        self.t_min, self.t_max = 0.05 * self.m * self.g / len(rp), t_max

        # Gains, randomized +-20% per trajectory so the data is not tied to one tuning.
        wp = 4.0 * rng.uniform(0.8, 1.2)                     # position bandwidth, rad/s
        wr = 15.0 * rng.uniform(0.8, 1.2)                    # roll/pitch bandwidth, rad/s
        wy = 4.0 * rng.uniform(0.8, 1.2)                     # yaw bandwidth: slow, it has little authority
        self.kp, self.kv = wp ** 2, 2.0 * 0.9 * wp
        Jd = np.diag(self.J)
        bw = np.array([wr, wr, wy])
        self.kR, self.kW = Jd * bw ** 2, Jd * 2.0 * 0.9 * bw
        self.max_tilt = math.radians(35.0)
        self.gains = dict(wp=wp, wr=wr, wy=wy)

    # Returns rotor thrust commands (N).
    def __call__(self, com, v, R, w_body, p_d, v_d, a_d, F_ff, heading, R_d=None):
        e3 = np.array([0.0, 0.0, 1.0])
        F = self.m * (-self.kp * (com - p_d) - self.kv * (v - v_d) + a_d + self.g * e3) + F_ff

        # Limit tilt.
        b3 = F / np.linalg.norm(F)
        tilt = math.acos(np.clip(b3 @ e3, -1.0, 1.0))
        if tilt > self.max_tilt:
            h = b3 - (b3 @ e3) * e3
            h /= np.linalg.norm(h) + 1e-12
            b3 = math.cos(self.max_tilt) * e3 + math.sin(self.max_tilt) * h

        b2 = np.cross(b3, heading)
        b2 /= np.linalg.norm(b2)
        R_d = np.column_stack([np.cross(b2, b3), b2, b3])
        f = F @ (R @ e3)

        E = R_d.T @ R - R.T @ R_d
        e_R = 0.5 * np.array([E[2, 1], E[0, 2], E[1, 0]])
        M = -self.kR * e_R - self.kW * w_body + np.cross(w_body, self.J @ w_body)
        return self.mix(f, M)

    # Mixer with roll/pitch priority. Yaw authority on a quad is small
    # (k_m/k_f ~ 1.6 cm of lever), so roll/pitch and collective are allocated
    # first, then the largest fraction of the yaw torque that still fits is
    # added. (Dropping yaw entirely when it does not fit leaves a large yaw
    # error with no correction at all, and the drone spins up against the wall.)
    def mix(self, f, M):
        T0 = self.A_inv @ np.r_[f, M[0], M[1], 0.0]
        if T0.max() > self.t_max:
            T0 = T0 + (self.t_max - T0.max())
        elif T0.min() < self.t_min:
            T0 = T0 + (self.t_min - T0.min())
        Ty = self.A_inv @ np.r_[0.0, 0.0, 0.0, M[2]]
        s = 1.0
        for t0, ty in zip(T0, Ty):
            if ty > 1e-12:
                s = min(s, (self.t_max - t0) / ty)
            elif ty < -1e-12:
                s = min(s, (t0 - self.t_min) / -ty)
        return np.clip(T0 + max(s, 0.0) * Ty, self.t_min, self.t_max)


    # Overrides the randomized gains (the closed-loop test draws its own).
    def set_gains(self, wp, wr, wy):
        Jd = np.diag(self.J)
        bw = np.array([wr, wr, wy])
        self.kp, self.kv = wp ** 2, 2.0 * 0.9 * wp
        self.kR, self.kW = Jd * bw ** 2, Jd * 2.0 * 0.9 * bw
        self.gains = dict(wp=wp, wr=wr, wy=wy)


# Position + attitude controller for a fully actuated (tilted-rotor) drone. The
# position loop asks for a world force, the attitude loop for a body torque
# toward R_d, and the 6-D wrench [R^T F; M] is allocated to the rotors directly:
# no tilting to move. Allocation: the exact solve when it fits the thrust
# limits, otherwise bounded least squares (lsq_linear) with the torque rows
# weighted more, so attitude wins over position when the rotors saturate.
class FullyActuatedController:

    BASE_GAINS = dict(wp=4.0, wr=15.0, wy=8.0)       # yaw is not starved here: tilted rotors give it lever arms

    def __init__(self, cfg, rng, t_max):
        self.m, self.g = cfg.mass, cfg.gravity
        self.J = np.asarray(cfg.inertia)
        self.A = wrench_matrix(cfg)
        self.A_inv = np.linalg.pinv(self.A)
        M = self.A.shape[1]
        self.t_min, self.t_max = 0.05 * self.m * self.g / M, t_max
        arm = float(np.mean(np.linalg.norm(np.asarray(cfg.rotor_pos) - np.asarray(cfg.com_offset), axis=1)))
        mg = self.m * self.g
        self.row_weight = np.r_[np.full(3, 1.0 / mg), np.full(3, 3.0 / (mg * arm))]
        self.set_gains(**{k: v * rng.uniform(0.8, 1.2) for k, v in self.BASE_GAINS.items()})

    def set_gains(self, wp, wr, wy):
        Jd = np.diag(self.J)
        bw = np.array([wr, wr, wy])
        self.kp, self.kv = wp ** 2, 2.0 * 0.9 * wp
        self.kR, self.kW = Jd * bw ** 2, Jd * 2.0 * 0.9 * bw
        self.gains = dict(wp=wp, wr=wr, wy=wy)

    # Returns rotor thrust commands (N).
    def __call__(self, com, v, R, w_body, p_d, v_d, a_d, F_ff, heading, R_d=None):
        e3 = np.array([0.0, 0.0, 1.0])
        F = self.m * (-self.kp * (com - p_d) - self.kv * (v - v_d) + a_d + self.g * e3) + F_ff
        if R_d is None:
            R_d = np.column_stack([heading, np.cross(e3, heading), e3])
        E = R_d.T @ R - R.T @ R_d
        e_R = 0.5 * np.array([E[2, 1], E[0, 2], E[1, 0]])
        M = -self.kR * e_R - self.kW * w_body + np.cross(w_body, self.J @ w_body)
        return self.allocate(np.r_[R.T @ F, M])

    def allocate(self, wrench):
        T = self.A_inv @ wrench
        if T.min() >= self.t_min and T.max() <= self.t_max:
            return T
        W = self.row_weight[:, None]
        return lsq_linear(W * self.A, self.row_weight * wrench, bounds=(self.t_min, self.t_max)).x


# Largest extra body wrench along one direction the rotors can add while still
# hovering level, within the thrust limits. direction: (6,) in [F; M] body
# coordinates, e.g. [1,0,0,0,0,0] = forward force (N), [0,0,0,0,0,1] = yaw torque (N m).
def max_hover_wrench(cfg, t_min, t_max, direction):
    A = wrench_matrix(cfg)
    M = A.shape[1]
    target = np.array([0.0, 0.0, cfg.mass * cfg.gravity, 0.0, 0.0, 0.0])
    # variables [T (M), s]: A T - s direction = hover, maximize s
    res = linprog(np.r_[np.zeros(M), -1.0], A_eq=np.hstack([A, -np.asarray(direction, float)[:, None]]),
                  b_eq=target, bounds=[(t_min, t_max)] * M + [(0.0, None)])
    return float(res.x[-1]) if res.success else 0.0


# axis 0/1/2 = roll/pitch/yaw torque (N m).
def max_hover_torque(cfg, t_min, t_max, axis=2):
    return max_hover_wrench(cfg, t_min, t_max, np.eye(6)[3 + axis])


# Smallest horizontal force (N) the rotors can add at level hover over the four
# body directions +-x, +-y: the sideways authority of a drone that does not tilt.
def min_hover_side_force(cfg, t_min, t_max):
    return min(max_hover_wrench(cfg, t_min, t_max, s * np.eye(6)[k]) for k in (0, 1) for s in (1, -1))
