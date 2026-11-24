"""MuJoCo backend for the data generator and the closed-loop test: the MJCF of
the drone (core, arms, props, rod, pad spheres) built from DroneConfig, the
mass properties read back from the compiled model, and the plant that steps it
one physics substep at a time with rotor drag added and every force logged.

The generator only talks to MujocoDronePlant through reset / set_wall /
set_command / command / pose / state / finite / substep, so another simulator
(AM-Bench) can be dropped in by writing a class with the same calls.
"""

import dataclasses

import mujoco
import numpy as np

from drone_config import build_drone_graph


def _quat2mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def _mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def _v(x):
    return " ".join(f"{float(a):.6g}" for a in x)


# Builds the MJCF from the DroneConfig geometry. Mass properties are NOT taken
# from the config: MuJoCo computes them from the geoms, and they are read back
# with true_config_from_model() so training uses exactly the simulated values.
def build_xml(cfg, gs):
    g_pts = build_drone_graph(cfg, com_relative=False)
    pts = g_pts["rest_nodes"].numpy()
    pad_idx = g_pts["pad_indices"]
    tip = np.asarray(cfg.rod_tip)
    axis = tip - np.asarray(cfg.rod_base)
    axis = axis / np.linalg.norm(axis)
    pad_r = max([r for r, _ in cfg.pad_rings], default=0.0) + cfg.pad_contact_radius

    rotors = ""
    for i, (p, a) in enumerate(zip(cfg.rotor_pos, cfg.rotor_axis)):
        rotors += f"""
      <geom name="arm{i}" type="capsule" fromto="0 0 0 {_v(p)}" size="0.008" mass="{cfg.arm_mass}" rgba=".3 .3 .3 1"/>
      <geom name="prop{i}" type="cylinder" pos="{_v(p)}" zaxis="{_v(a)}" size="{cfg.prop_radius} 0.005" mass="{cfg.prop_mass}" rgba=".2 .5 .9 .5"/>
      <site name="rotor{i}" pos="{_v(p)}" zaxis="{_v(a)}"/>"""

    pads = ""
    for k, j in enumerate(pad_idx):
        pads += f"""
      <geom name="pad{k}" type="sphere" pos="{_v(pts[j])}" size="{cfg.pad_contact_radius}" mass="0.001"
            friction="{gs.mu} 0.005 0.0001" condim="3" solref="{_v(gs.pad_solref)}" rgba=".9 .2 .2 1"/>"""

    acts = ""
    for i, s in enumerate(cfg.spin_dir):
        acts += f"""
    <general name="m{i}" site="rotor{i}" gear="0 0 1 0 0 {-s * cfg.k_m / cfg.k_f:.6g}"
             dyntype="filter" dynprm="{gs.motor_tau}" gainprm="1" biastype="none"
             ctrllimited="true" ctrlrange="0 100"/>"""

    return f"""
<mujoco model="aerial_manipulator">
  <option timestep="{gs.timestep}" gravity="0 0 {-cfg.gravity}" density="1.225"
          viscosity="1.8e-5" integrator="Euler" cone="elliptic"/>
  <default><geom contype="1" conaffinity="0"/></default>
  <worldbody>
    <light pos="0 0 4"/>
    <geom name="floor" type="plane" size="5 5 .1" contype="0" conaffinity="0" rgba=".8 .8 .8 1"/>
    <!-- mocap body: MuJoCo does not re-pose static worldbody geoms when model.geom_pos changes -->
    <body name="wall_body" mocap="true" pos="0 0 -10">
      <geom name="wall" type="box" size="0.02 2 2" contype="0" conaffinity="1"
            friction="{gs.mu} 0.005 0.0001" solref="{_v(gs.wall_solref)}" rgba=".7 .6 .5 .6"/>
    </body>
    <body name="drone" pos="0 0 1">
      <freejoint/>
      <geom name="core" type="box" pos="{_v(cfg.core_pos)}" size="{_v(cfg.core_half_size)}" mass="{cfg.core_mass}" fluidshape="ellipsoid" rgba=".1 .1 .1 1"/>{rotors}
      <geom name="rod" type="capsule" fromto="{_v(cfg.rod_base)} {_v(tip - 0.012 * axis)}" size="0.006" mass="{cfg.rod_mass}"
            fluidshape="ellipsoid" rgba=".6 .6 .6 1"/>
      <geom name="sensor" type="cylinder" pos="{_v(np.asarray(cfg.rod_base) + 0.02 * axis)}" zaxis="{_v(axis)}"
            size="0.015 0.012" mass="0.05" rgba=".9 .7 .1 1"/>
      <geom name="pad_disk" type="cylinder" pos="{_v(tip - 0.002 * axis)}" zaxis="{_v(axis)}"
            size="{pad_r:.4f} 0.002" mass="0.01" contype="0" conaffinity="0" rgba=".9 .4 .4 .6"/>{pads}
    </body>
  </worldbody>
  <actuator>{acts}
  </actuator>
</mujoco>"""


# Reads the simulated mass properties back into a DroneConfig. The body frame
# of the MuJoCo drone IS the mocap frame, so body_ipos is the COM offset.
def true_config_from_model(model, cfg, gs):
    bid = model.body("drone").id
    Ri = np.zeros(9)
    mujoco.mju_quat2Mat(Ri, model.body_iquat[bid])
    Ri = Ri.reshape(3, 3)
    I = Ri @ np.diag(model.body_inertia[bid]) @ Ri.T
    return dataclasses.replace(
        cfg, mass=float(model.body_mass[bid]),
        inertia=tuple(tuple(float(x) for x in row) for row in I),
        com_offset=tuple(float(x) for x in model.body_ipos[bid]),
        dt=gs.timestep * gs.substeps)


class Crash(Exception):
    pass


# ======================================================================
# MuJoCo plant: one physics substep at a time
# ======================================================================

class MujocoDronePlant:
    """The MuJoCo drone with the forces MuJoCo does not model added by hand
    (per-rotor drag at each hub), plus force logging. Used by the data
    generator and by the closed-loop evaluation, so both run identical physics.

    substep(wind, t) applies rotor drag for the given wind, advances one MuJoCo
    timestep, and returns what acted during that step: rotor thrusts (M), aero
    force (3, MuJoCo fluid + rotor drag), contact force on the pad (3), its
    torque about the COM (3), and the per-pad-sphere contact forces (P, 3).
    Raises Crash if any drone geom other than a pad sphere touches the wall."""

    def __init__(self, model, cfg_true, gs, data=None):
        self.model, self.cfg, self.gs = model, cfg_true, gs
        self.data = data if data is not None else mujoco.MjData(model)
        self.bid = model.body("drone").id
        self.wall_gid = model.geom("wall").id
        n_pad = len(build_drone_graph(cfg_true)["pad_indices"])
        self.pad_gids = {model.geom(f"pad{k}").id: k for k in range(n_pad)}
        self.drone_gids = set(np.flatnonzero(model.geom_bodyid == self.bid).tolist())
        self.c_off = np.asarray(cfg_true.com_offset)
        self.rotor_r = np.asarray(cfg_true.rotor_pos) - self.c_off          # hubs relative to the COM, body frame
        a = np.asarray(cfg_true.rotor_axis, dtype=float)
        self.rotor_a = a / np.linalg.norm(a, axis=1, keepdims=True)
        self.qfrc_fluid = "qfrc_fluid" if hasattr(self.data, "qfrc_fluid") else "qfrc_passive"
        self._f6 = np.zeros(6)

    # COM position, attitude, body angular velocity, COM velocity (world).
    def state(self):
        d = self.data
        R = _quat2mat(d.qpos[3:7])
        w_b = d.qvel[3:6].copy()
        return d.qpos[0:3] + R @ self.c_off, R, w_b, d.qvel[0:3] + np.cross(R @ w_b, R @ self.c_off)

    def substep(self, wind, t):
        m, d, cfg, gs = self.model, self.data, self.cfg, self.gs
        com, R, w_b, v_com = self.state()

        # Per-rotor drag at each hub, from that hub's local airspeed.
        m.opt.wind[:] = wind
        omega_true = np.sqrt(np.maximum(d.act, 0.0) / cfg.k_f)
        r_hub = self.rotor_r @ R.T                                           # (M,3) lever arms, world
        a_hub = self.rotor_a @ R.T                                           # (M,3) rotor axes, world
        u_hub = wind - (v_com + np.cross(R @ w_b, r_hub))                    # local airspeed at each hub
        u_perp = u_hub - np.sum(u_hub * a_hub, axis=1, keepdims=True) * a_hub
        F_hub = cfg.mass * gs.k_rotor * omega_true[:, None] * u_perp         # (M,3)
        F_rd = F_hub.sum(0)
        d.xfrc_applied[self.bid, :3] = F_rd                                  # xfrc_applied acts at the COM,
        d.xfrc_applied[self.bid, 3:] = np.cross(r_hub, F_hub).sum(0)         # so add the hubs' moment

        mujoco.mj_step(m, d)

        # Forces used during the step just taken (MuJoCo leaves them from the
        # pre-integration forward pass).
        Fc, tc = np.zeros(3), np.zeros(3)
        pad = np.zeros((len(self.pad_gids), 3))
        for i in range(d.ncon):
            con = d.contact[i]
            if self.wall_gid not in (con.geom1, con.geom2):
                continue
            other = con.geom2 if con.geom1 == self.wall_gid else con.geom1
            if other not in self.pad_gids:
                if other in self.drone_gids:
                    raise Crash(f"{m.geom(other).name} hit the wall at t={t:.2f}s")
                continue
            mujoco.mj_contactForce(m, d, i, self._f6)
            F = con.frame.reshape(3, 3).T @ self._f6[:3]                     # force on geom2
            if other == con.geom1:
                F = -F
            Fc += F
            tc += np.cross(con.pos - com, F)
            pad[self.pad_gids[other]] += F
        return d.actuator_force.copy(), getattr(d, self.qfrc_fluid)[0:3] + F_rd, Fc, tc, pad

    # ---- what a backend provides to the generator (an AM-Bench backend would
    # implement the same calls; physics substeps stay inside the backend) ----
    def set_wall(self, n, wall_point):
        set_wall_pose(self.model, self.data, n, wall_point)

    # Starts the drone at rest with its COM at com, attitude R, and the motors
    # already producing `thrust` (N per rotor).
    def reset(self, com, R, thrust):
        d = self.data
        d.qpos[0:3] = com - R @ self.c_off
        d.qpos[3:7] = _mat2quat(R)
        d.qvel[:] = 0.0
        d.act[:] = thrust
        d.ctrl[:] = thrust
        mujoco.mj_forward(self.model, d)

    def set_command(self, thrust):
        self.data.ctrl[:] = thrust

    def command(self):
        return self.data.ctrl.copy()

    # Body-origin (mocap) position and wxyz quaternion: what gets recorded.
    def pose(self):
        return self.data.qpos[0:3].copy(), self.data.qpos[3:7].copy()

    def finite(self):
        return bool(np.all(np.isfinite(self.data.qpos)))


# Verlet alignment: x is a per-substep log (n_sub, ...); returns per-frame values
# with triangle weights over the two intervals around each frame, which is
# exactly what the second difference of the recorded positions sees.
def align_to_frames(x, S, T_frames, first=None):
    w = np.r_[np.arange(S), S - np.arange(S)] / S ** 2                     # sums to 1
    out = np.empty((T_frames,) + x.shape[1:])
    for k in range(1, T_frames):
        out[k] = np.tensordot(w, x[(k - 1) * S:(k + 1) * S], axes=1)
    out[0] = out[1] if first is None else first
    return out


# Places the wall (a mocap box) so its surface passes through wall_point with
# outward normal n. MuJoCo does not re-pose static worldbody geoms when
# model.geom_pos changes, which is why the wall lives on a mocap body.
def set_wall_pose(model, data, n, wall_point):
    t1 = np.cross(n, [0.0, 0.0, 1.0])
    t1 /= np.linalg.norm(t1)
    half = model.geom_size[model.geom("wall").id][0]
    data.mocap_pos[0] = wall_point - half * n
    data.mocap_quat[0] = _mat2quat(np.column_stack([n, t1, np.cross(n, t1)]))   # box local x = n
