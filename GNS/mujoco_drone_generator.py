"""Generate MuJoCo trajectories of the aerial manipulator touching, pushing on,
and sliding along a wall, in the format drone_data.py loads.

HOW THE DRONE IS FLOWN
The model never learns the controller. A controller is only needed so the sim
drone goes where we want it; what gets recorded is the motion plus the rotor
speeds that actually produced it, and the model takes those rotor speeds as
inputs. So any controller that reliably makes the contacts we want is fine.

  reference      a randomized scenario (free flight, near-wall hover, tap,
                 push, push-and-slide) written as pad-center waypoints in the
                 wall frame, smoothed with min-jerk segments, plus a push-force
                 profile
  controller     quad: standard geometric (SE(3)) position + attitude
                 controller (Lee et al. 2010) once per recorded frame (100 Hz),
                 with a slow yaw loop. Fully actuated drone (tilted rotors): world
                 force + body torque tracked independently, so it can hold a
                 random roll/pitch target while it tracks the pad reference
                 (drone_controllers.py). Pushing is a feedforward force into the wall on top
                 of position tracking, so the controller needs no force
                 measurement. The command is held for the whole frame, so the
                 rotor speeds over [t, t+1] depend only on the state at t (see
                 CAUSALITY below)
  mixer          quad: desired collective thrust + body torques -> 4 rotor
                 thrusts, roll/pitch first, then as much yaw as still fits.
                 Fully actuated: the 6-D body wrench -> rotor thrusts, exact
                 when it fits the thrust limits, else bounded least squares
  excitation     random band-limited noise on the rotor commands, so rotor
                 speeds are not a fixed function of the state (closed-loop
                 data is otherwise biased toward the controller's own choices)
  motors         MuJoCo first-order filter on thrust (motor lag)

PHYSICS IN THE SIM
  rotor thrust + yaw reaction: site actuators, gear = [0 0 1 0 0 -s*k_m/k_f]
  body aero:  MuJoCo ellipsoid fluid model with opt.wind (OU gusts, off by default)
  rotor drag: per rotor, k_rotor * m * omega_i * u_perp_i at that rotor's hub,
              where u_perp_i is the hub's local airspeed (wind minus hub velocity,
              including rotation) with the rotor-axis component removed. Applied
              through xfrc_applied as the summed force plus its torque about the
              COM. Same law and coefficient as the aero GNN's rotor anchor.
              (MuJoCo has no rotor drag; without it sim aero is unlike hardware.)
  pad:        one small collision sphere per pad node from build_drone_graph(),
              so the sim contact points ARE the graph's pad nodes
  NOT modeled: rotor downwash / near-wall rotor aero, rotor gyroscopics, flex

WHAT IS RECORDED (per frame, dt = timestep * substeps)
  pos, quat           body-frame origin (= mocap origin), wxyz
  rotor_speed         Verlet-aligned: sqrt of the triangle-weighted mean of
                      thrust/k_f over the two intervals around the frame, which
                      is exactly what the second difference sees (meta says
                      "aligned"). With high-rate ESC telemetry, hardware can
                      compute the same thing.
  rotor_thrust        the same aligned value as thrust (N), what AM-Bench reports
  rotor_speed_cmd     commanded speed at the frame (for a motor model later)
  wind, wall_normal, wall_point, phase (commanded scenario phase)
  labels (aligned the same way, validation only):
      F_contact, tau_contact (about the COM), F_pad (per pad sphere), F_aero
  meta                dt, gravity, mass/inertia/COM read from the compiled
                      model, k_f, k_m, mu, scenario, and generator settings

CAUSALITY
  rotor_speed[t] is aligned over the two intervals around frame t, so it
  includes the motor response during [t, t+1]. That is legitimate (it is what
  an MPC computes from its planned command), as long as the command for
  [t, t+1] was chosen from the state at t. With a faster controller (the old
  500 Hz), commands inside [t, t+1] reacted to what happened inside it, e.g. a
  contact starting there, and that reaction leaked into rotor_speed[t]. Hence
  ctrl_every = substeps. audit_checks.py verifies this.

STABILITY NOTES (learned building this; they apply to hardware too)
  - Pushing needs nose-down pitch (about atan(F/mg)). With a level rod at or
    below the COM, that pitch drops the pad below the COM, and the wall's
    normal force then pitches the drone further into the wall: a runaway once
    the moment exceeds pitch authority. With the old level rod 2 cm below the
    rotor plane, ~13% of slides ended this way; angling the rod up ~8 deg
    (DroneConfig.rod_tip z = +5 cm) removed it in 24/24 trials.
  - Yaw authority is ~1.6 cm of lever (k_m/k_f) against the rod's ~0.4 m.
    Horizontal friction, and the normal force on a yawed rod (which grows with
    the yaw error), can out-torque it, so push forces are capped from a yaw budget.

SCENARIO SETTINGS
  Every scenario range (push force, approach speeds, hover distances, slide and
  arc speeds, sideways loads, scenario mix, wind) is a GenSettings field at the
  top of this file. Each trajectory saves them in meta["gen_settings"], and the
  closed-loop test rebuilds its scenarios from that, so it always matches the data.

FILES
  drone_scenarios.py       reference waypoints, push profiles, attitude targets
  drone_controllers.py     quad and fully actuated controllers, allocation
  mujoco_drone_backend.py  MJCF, mass properties, the MuJoCo plant
  this file                GenSettings, one trajectory, the driver, the CLI
An AM-Bench port replaces only the backend (same plant calls).

Run:  python mujoco_drone_generator.py --out data/mj_drone_wind --n 1200
      python mujoco_drone_generator.py --out data/mj_hexa --n 1200 --drone fa_hexa
      python mujoco_drone_generator.py --out data/mj_drone_wind --n 1200 --wind-max 3   (override wind)
      python mujoco_drone_generator.py --out data/mj_drone --n 5 --verify
      python mujoco_drone_generator.py --replay data/mj_drone/3.pt   (viewer)
On ROAR, split with --start (array task index * n) so seeds never overlap.
"""

import argparse
import dataclasses
import json
import math
import os


import mujoco
import numpy as np
import torch

from drone_config import DroneConfig, PRESETS, build_drone_graph, hover_thrusts
from drone_scenarios import (PHASES, Reference, make_scenario, wall_tangents, push_force_vector,
                             add_attitude_targets, reference_attitude, yaw_force_cap, limit_segment_accel)
from drone_controllers import (GeometricController, FullyActuatedController, make_controller, thrust_limits,
                               max_hover_torque, min_hover_side_force)
from mujoco_drone_backend import (build_xml, true_config_from_model, Crash, MujocoDronePlant, align_to_frames,
                                  set_wall_pose, _quat2mat, _mat2quat)


@dataclasses.dataclass
class GenSettings:
    timestep: float = 0.0005
    substeps: int = 20              # dt = 0.01 s, 100 Hz recorded
    ctrl_every: int = 20            # controller once per frame (100 Hz); must equal substeps for causal data
    duration: tuple = (4.0, 7.0)    # seconds, random per trajectory
    motor_tau: float = 0.025        # s, thrust first-order lag
    t_max_over_hover: float = 2.5
    mu: float = 0.4                 # wall friction (fixed per dataset: the model learns one mu)
    k_rotor: float = 5e-5           # rotor drag / m, per rotor, per (rad/s)
    wall_solref: tuple = (0.004, 1.0)
    # Pad contact softness (MuJoCo averages the pad's and wall's solref). Equal
    # to wall_solref = rigid pad: it touches on ONE edge point unless aligned to
    # well under a degree. A larger time constant mimics a rubber pad, which
    # spreads load over several nodes. Match the real pad.
    pad_solref: tuple = (0.004, 1.0)

    # ---- wind ----
    wind_max: float = 5.0           # m/s; 70% of trajectories get a mean wind U(0, wind_max), 30% none
    gust_std: float = 0.0           # m/s, OU gusts (off)
    # The controller has no integral term, so wind holds it 1-2 cm off its target,
    # which swamps the few-mm hover gaps. True: cancel last substep's aero force
    # in the feedforward (the wind still shows in the motion and rotor speeds).
    aero_feedforward: bool = True

    # ---- excitation / wall / heading ----
    excite_max: float = 0.1         # max relative rotor-command noise
    tilted_wall_prob: float = 0
    wall_tilt_max_deg: float = 0
    yaw_offset_max_deg: float = 10.0
    yaw_torque_budget: float = 0.15 # N m of yaw torque the pad's moments may use (see make_scenario)

    # ---- scenario mix ----
    scenario_probs: tuple = (("free", 0.05), ("hover_near", 0.15), ("tap", 0.10),
                             ("push", 0.25), ("slide", 0.30), ("arc", 0.15))

    # ---- push force (push, slide, arc) ----
    push_force: tuple = (0.2, 3.0)              # N, log-uniform; the top is also capped by the yaw budget
    push_approach_speed: tuple = (0.03, 0.3)    # m/s, log-uniform

    # ---- hover_near ----
    hover_near_gap: tuple = (0.001, 0.008)      # m above touching (pad-center d = r_c + gap): where the gate is partly open
    hover_near_count: int = 2                   # hovers per trajectory in that band
    hover_far_dist: tuple = (0.012, 0.15)       # m, pad-center distance for the farther hovers
    hover_far_max: int = 1                      # 0 to this many farther hovers per trajectory

    # ---- tap ----
    tap_approach_speed: tuple = (0.05, 0.5)     # m/s

    # ---- straight slides ----
    slide_speed: tuple = (0.01, 0.2)            # m/s, log-uniform (slip gate is at 0.03 m/s)
    slide_dist: tuple = (0.05, 0.25)            # m
    slide_max_time: float = 4.0                 # s, caps the distance of slow slides
    slide_count_max: int = 2                    # 1 to this many slides per trajectory

    # ---- arc slides ----
    arc_radius: tuple = (0.05, 0.15)            # m
    arc_laps: tuple = (0.5, 1.0)                # fraction of a full circle
    arc_speed: tuple = (0.03, 0.15)             # m/s, mean speed along the arc

    # ---- sideways-loaded static holds (push, slide, arc) ----
    side_load_prob: float = 0.5                 # fraction of push/slide/arc trajectories that get one
    side_load_frac: tuple = (0.3, 1.2)          # tangential push as a fraction of mu * F (>1 creeps / breaks away)
    side_load_time: tuple = (0.5, 1.5)          # s to ramp the sideways push in

    # ---- fully actuated drones only (DroneConfig.fully_actuated) ----
    # Attitude targets: this fraction of trajectories gets a random body
    # roll/pitch target per segment (+-attitude_max_deg); the rest fly level.
    attitude_prob: float = 0.7
    attitude_max_deg: float = 6.0     # holding tilt theta costs m g sin(theta) of the ~16 N sideways authority
    # Yaw budget for the push cap, as a fraction of the most yaw torque the
    # rotors can add at hover (computed from the allocation). The quad uses
    # the fixed yaw_torque_budget above instead.
    yaw_budget_frac: float = 0.3
    # Peak COM acceleration of the reference, as a fraction of the sideways
    # force the rotors can add at level hover (/ mass). A drone that does not
    # tilt has only that much sideways authority (~2.7 m/s^2 for the hexa); the
    # quad's segment timings ask for up to ~8 m/s^2, so faster segments are
    # stretched (drone_scenarios.limit_segment_accel).
    accel_frac: float = 0.5


# Settings that differ for a drone preset (python mujoco_drone_generator.py --drone fa_hexa).
# The hexa is ~4x the quad's mass with ~16 N of sideways force at level, so it pushes harder.
PRESET_GEN_SETTINGS = {"quad": {}, "fa_hexa": dict(push_force=(0.2, 6.0))}


# Yaw budget (N m) the pad's moments may use when the push force is capped.
def yaw_budget(cfg, gs):
    if not cfg.fully_actuated:
        return gs.yaw_torque_budget
    return gs.yaw_budget_frac * max_hover_torque(cfg, *thrust_limits(cfg, gs), axis=2)


# Fully actuated drones only: random attitude targets (attitude_prob of the
# trajectories), then segments stretched to the drone's sideways authority.
# p_tip_of(r) = pad-center waypoint, tip_body = pad center from the COM (body).
# Returns (start, segments) unchanged for a quad (and draws nothing from rng).
def shape_reference(cfg, gs, s0, seg, rng, R_h, p_tip_of, tip_body):
    if not cfg.fully_actuated:
        return s0, seg
    if rng.random() < gs.attitude_prob:
        s0, seg = add_attitude_targets(s0, seg, rng, gs.attitude_max_deg)
    a_max = gs.accel_frac * min_hover_side_force(cfg, *thrust_limits(cfg, gs)) / cfg.mass
    com_of = lambda r: p_tip_of(r) - reference_attitude(R_h, r) @ tip_body
    return s0, limit_segment_accel(s0, seg, com_of, a_max)


# ======================================================================
# Simulation of one trajectory
# ======================================================================

# pulse: optional (t_start, t_end, force_xyz) external force on the drone, used
# only by audit_checks.py to show that perturbing the future leaves past rotor
# speeds unchanged.
def simulate_trajectory(model, cfg_true, gs, rng, pulse=None):
    plant = MujocoDronePlant(model, cfg_true, gs)
    S, h = gs.substeps, gs.timestep
    dt = h * S
    c_off = plant.c_off
    M = len(cfg_true.rotor_pos)
    t_hover = hover_thrusts(cfg_true)
    ctrl = make_controller(cfg_true, gs, rng)

    # ---- wall pose: random yaw, sometimes tilted ----
    psi = rng.uniform(0, 2 * math.pi)
    beta = math.radians(rng.uniform(-1, 1) * gs.wall_tilt_max_deg) if rng.random() < gs.tilted_wall_prob else 0.0
    n = np.array([-math.cos(beta) * math.cos(psi), -math.cos(beta) * math.sin(psi), math.sin(beta)])
    wall_point = np.array([0.0, 0.0, 1.5])
    t1, t2 = wall_tangents(n)
    plant.set_wall(n, wall_point)

    # ---- heading: rod (body x) points at the wall, plus a random yaw offset ----
    nh = -n.copy()
    nh[2] = 0.0
    nh /= np.linalg.norm(nh)
    yo = math.radians(rng.uniform(-1, 1) * gs.yaw_offset_max_deg)
    heading = np.array([[math.cos(yo), -math.sin(yo), 0], [math.sin(yo), math.cos(yo), 0], [0, 0, 1]]) @ nh
    R_h = np.column_stack([heading, np.cross([0, 0, 1.0], heading), [0, 0, 1.0]])
    tip_body = np.asarray(cfg_true.rod_tip) - c_off             # pad center relative to COM, body frame

    # ---- scenario ----
    kinds, probs = zip(*gs.scenario_probs)
    kind = str(rng.choice(kinds, p=np.asarray(probs) / sum(probs)))
    lever = float(np.linalg.norm(tip_body[:2]))
    yaw_cap = yaw_force_cap(yaw_budget(cfg_true, gs), lever, gs.mu, yo)
    s0, seg = make_scenario(kind, rng, cfg_true.pad_contact_radius, rng.uniform(*gs.duration), gs, yaw_cap)
    p_tip_of = lambda r: wall_point + r["d"] * n + r["a"] * t1 + r["b"] * t2
    s0, seg = shape_reference(cfg_true, gs, s0, seg, rng, R_h, p_tip_of, tip_body)
    ref = Reference(s0, seg)
    T_frames = int(ref.T / dt) + 1

    # COM reference that puts the pad center on the waypoint at the reference attitude
    def p_com_ref(t):
        r, ph = ref(t)
        R_d = reference_attitude(R_h, r)
        return p_tip_of(r) - R_d @ tip_body, push_force_vector(r, n, t1, t2), ph, R_d

    # ---- wind: mean + OU gusts ----
    w_mean = np.zeros(3)
    if rng.random() > 0.3:
        w_ang = rng.uniform(0, 2 * math.pi)
        w_mean = rng.uniform(0, gs.wind_max) * np.array([math.cos(w_ang), math.sin(w_ang), 0.0])
    gust_sig = rng.uniform(0, gs.gust_std)
    gust = np.zeros(3)
    ex_sig, ex_tau = rng.uniform(0, gs.excite_max), 0.05
    ex = np.zeros(M)

    # ---- initial state: hovering at the start reference ----
    plant.reset(p_com_ref(0.0)[0], R_h, t_hover)

    n_sub = T_frames * S
    log_thrust = np.zeros((n_sub, M))
    log_Fc, log_tc, log_Fa = np.zeros((n_sub, 3)), np.zeros((n_sub, 3)), np.zeros((n_sub, 3))
    log_pad = np.zeros((n_sub, len(plant.pad_gids), 3))
    frames = dict(pos=[], quat=[], rotor_speed_cmd=[], wind=[], phase=[])

    for j in range(n_sub):
        t = j * h
        com, R, w_b, v_com = plant.state()

        # Crash checks.
        if not plant.finite() or com[2] < 0.2 or (R @ [0, 0, 1])[2] < 0.5:
            raise Crash(f"state left the envelope at t={t:.2f}s")

        # Record the frame state (before the substeps that follow it).
        if j % S == 0:
            pos, quat = plant.pose()
            frames["pos"].append(pos)
            frames["quat"].append(quat)
            frames["rotor_speed_cmd"].append(np.sqrt(np.maximum(plant.command(), 0) / cfg_true.k_f))
            frames["wind"].append(w_mean + gust)
            frames["phase"].append(p_com_ref(t)[2])

        # Controller at its own rate.
        if j % gs.ctrl_every == 0:
            eps = 1e-3
            p_d, F_push, _, R_d = p_com_ref(t)
            v_d = (p_com_ref(t + eps)[0] - p_com_ref(t - eps)[0]) / (2 * eps)
            a_d = (p_com_ref(t + eps)[0] - 2 * p_d + p_com_ref(t - eps)[0]) / eps ** 2
            if gs.aero_feedforward and j > 0:
                F_push = F_push - log_Fa[j - 1]                        # aero force of the last substep
            T_cmd = ctrl(com, v_com, R, w_b, p_d, v_d, a_d, F_push, heading, R_d)
            hc = gs.ctrl_every * h
            ex += -ex * hc / ex_tau + ex_sig * math.sqrt(2 * hc / ex_tau) * rng.normal(size=M)
            plant.set_command(np.clip(T_cmd * (1.0 + ex), 0.0, ctrl.t_max))

        # Wind gusts, then one physics substep (rotor drag + MuJoCo + force logging).
        if pulse is not None:
            plant.data.qfrc_applied[0:3] = pulse[2] if pulse[0] <= t < pulse[1] else 0.0
        gust += -gust * h / 1.0 + gust_sig * math.sqrt(2 * h) * np.r_[rng.normal(size=2), 0.2 * rng.normal()]
        log_thrust[j], log_Fa[j], log_Fc[j], log_tc[j], log_pad[j] = plant.substep(w_mean + gust, t)

    align = lambda x: align_to_frames(x, S, T_frames)
    thrust_al = align(log_thrust)
    traj = dict(
        pos=frames["pos"], quat=frames["quat"],
        rotor_speed=np.sqrt(np.maximum(thrust_al, 0.0) / cfg_true.k_f), rotor_thrust=thrust_al,
        rotor_speed_cmd=frames["rotor_speed_cmd"], wind=frames["wind"], phase=frames["phase"],
        F_contact=align(log_Fc), tau_contact=align(log_tc), F_pad=align(log_pad), F_aero=align(log_Fa))
    # Positions in float64: float32 resolves only ~1e-7 m at 1.5 m from the
    # origin, about 1% of the per-step aero signal the model learns from.
    dtypes = {"phase": torch.long, "pos": torch.float64}
    traj = {k: torch.tensor(np.asarray(v), dtype=dtypes.get(k, torch.float32)) for k, v in traj.items()}
    traj["wall_normal"] = torch.tensor(n, dtype=torch.float32)
    traj["wall_point"] = torch.tensor(wall_point, dtype=torch.float32)
    traj["meta"] = dict(dt=dt, gravity=cfg_true.gravity, rotor_speed_hold="aligned",
                        mass=cfg_true.mass, inertia=cfg_true.inertia, com_offset=cfg_true.com_offset,
                        k_f=cfg_true.k_f, k_m=cfg_true.k_m, mu=gs.mu, k_rotor=gs.k_rotor,
                        scenario=kind, wind_mean=w_mean.tolist(), excitation=ex_sig,
                        wall_tilt_deg=math.degrees(beta), yaw_offset_deg=math.degrees(yo),
                        fully_actuated=cfg_true.fully_actuated, attitude_targets="roll" in s0,
                        aero_label_source=plant.qfrc_fluid, gen_settings=dataclasses.asdict(gs), **ctrl.gains)
    return traj


# ======================================================================
# Check: the logged forces must explain the recorded motion
# ======================================================================

# Verlet residual of the COM after removing thrust, gravity, and the logged
# contact + aero labels, in units of g. If the conventions (frame alignment,
# contact sign, COM offset, thrust axis) are right, this is ~0.
def verify_trajectory(traj, cfg):
    from force_data import quat_wxyz_to_R
    R = quat_wxyz_to_R(traj["quat"].double())
    com = traj["pos"].double() + R @ torch.tensor(cfg.com_offset, dtype=torch.float64)
    dt, m = cfg.dt, cfg.mass
    a_meas = (com[2:] - 2 * com[1:-1] + com[:-2]) / dt ** 2
    om = traj["rotor_speed"].double()[1:-1]
    axes = torch.tensor(cfg.rotor_axis, dtype=torch.float64)
    F_thr = R[1:-1] @ ((cfg.k_f * om ** 2) @ axes).unsqueeze(-1)
    F_ext = (traj["F_contact"] + traj["F_aero"]).double()[1:-1]
    g = torch.tensor([0, 0, -cfg.gravity], dtype=torch.float64)
    res = a_meas - F_thr.squeeze(-1) / m - F_ext / m - g
    in_contact = traj["F_contact"][1:-1].norm(dim=-1) > 1e-6
    out = dict(resid_all_g=float(res.norm(dim=-1).mean() / cfg.gravity),
               contact_frac=float(in_contact.float().mean()),
               normal_force_ok=bool(((traj["F_contact"] @ traj["wall_normal"]) > -1e-4).all()))
    if in_contact.any():
        out["resid_contact_g"] = float(res[in_contact].norm(dim=-1).mean() / cfg.gravity)
        out["mean_contact_N"] = float(traj["F_contact"][1:-1][in_contact].norm(dim=-1).mean())
    return out


# ======================================================================
# Driver
# ======================================================================

def generate(out, n, start=0, seed=0, cfg=None, gs=None, verify=False, verbose=True):
    cfg, gs = cfg or DroneConfig(), gs or GenSettings()
    model = mujoco.MjModel.from_xml_string(build_xml(cfg, gs))
    cfg_true = true_config_from_model(model, cfg, gs)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "drone_config.json"), "w") as f:
        json.dump(cfg_true.to_dict(), f, indent=2)

    crashes = 0
    for i in range(start, start + n):
        attempt = 0
        while True:
            rng = np.random.default_rng([seed, i, attempt])
            try:
                traj = simulate_trajectory(model, cfg_true, gs, rng)
                break
            except Crash as e:
                crashes += 1
                attempt += 1
                if verbose:
                    print(f"  traj {i} attempt {attempt}: discarded ({e})")
        torch.save(traj, os.path.join(out, f"{i}.pt"))
        if verbose:
            msg = f"traj {i}: {traj['meta']['scenario']:10s} T={traj['pos'].shape[0]}"
            if verify:
                v = verify_trajectory(traj, cfg_true)
                msg += "  " + "  ".join(f"{k}={v[k]:.4g}" if isinstance(v[k], float) else f"{k}={v[k]}"
                                        for k in v)
            print(msg, flush=True)
    if verbose:
        print(f"done: {n} trajectories, {crashes} discarded; config -> {out}/drone_config.json")
    return cfg_true


# Plays a saved trajectory back in the MuJoCo viewer (needs a display; on
# macOS run it with mjpython). Kinematic replay: it sets the recorded poses.
def replay(path, speed=1.0):
    import time
    import mujoco.viewer
    traj = torch.load(path, weights_only=False)
    with open(os.path.join(os.path.dirname(path), "drone_config.json")) as f:
        cfg = DroneConfig(**json.load(f))
    model = mujoco.MjModel.from_xml_string(build_xml(cfg, GenSettings()))
    data = mujoco.MjData(model)
    set_wall_pose(model, data, traj["wall_normal"].double().numpy(), traj["wall_point"].double().numpy())
    print(f"{path}: {traj['meta']['scenario']}, {traj['pos'].shape[0]} frames")
    with mujoco.viewer.launch_passive(model, data) as v:
        for k in range(traj["pos"].shape[0]):
            if not v.is_running():
                break
            data.qpos[0:3] = traj["pos"][k].numpy()
            data.qpos[3:7] = traj["quat"][k].numpy()
            mujoco.mj_forward(model, data)
            v.sync()
            time.sleep(cfg.dt / speed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mj_drone")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--replay", default=None, help="path to a saved .pt to play back in the viewer")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--wind-max", type=float, default=None, help="mean wind, m/s (default: GenSettings.wind_max)")
    ap.add_argument("--gust-std", type=float, default=None, help="gust std, m/s (default: GenSettings.gust_std)")
    ap.add_argument("--drone", default="quad", choices=sorted(PRESETS), help="drone preset (drone_config.PRESETS)")
    a = ap.parse_args()
    if a.replay:
        replay(a.replay, a.speed)
    else:
        gs = dataclasses.replace(GenSettings(), **PRESET_GEN_SETTINGS[a.drone])
        if a.wind_max is not None:
            gs = dataclasses.replace(gs, wind_max=a.wind_max)
        if a.gust_std is not None:
            gs = dataclasses.replace(gs, gust_std=a.gust_std)
        generate(a.out, a.n, a.start, a.seed, cfg=PRESETS[a.drone](), gs=gs, verify=a.verify)
