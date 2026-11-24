"""Define the aerial manipulator's physical description and the graph the
contact GNN runs on. DroneConfig holds the measured mass properties, the rotor
layout and thrust/yaw coefficients, the rod geometry, and the graph options.
Everything is entered in the MOCAP body frame (the frame the motion-capture
rigid body reports), and the helpers convert it to COM-relative quantities,
because the dynamics are written about the center of mass. build_drone_graph()
creates the node set (one body node, nodes along the rod, a tip node or a small
pad), node types, contact mask, and hand-wired edges.

Every default number below is a PLACEHOLDER. Replace mass, inertia, COM offset,
k_f, and k_m with measured values (or read them from the MuJoCo model for sim).

The defaults are the quad. fa_hexa_config() is the fully actuated hexarotor
(AM-Bench's FAHexa with the rod + pad); PRESETS maps names to both, and the
generator's --drone flag picks one. wrench_matrix() and hover_thrusts() work
for any rotor layout.
"""

import math
from dataclasses import dataclass, asdict

import numpy as np
import torch

import dataclasses

# Node kinds used while building the geometry.
NODE_BODY, NODE_ROD, NODE_TIP = 0, 1, 2
N_NODE_TYPES = 3

# One-hot node types for each GNN.
# Aero graph:    COM [1,0,0,0], rotor [0,1,0,0], rod [0,0,1,0], pad [0,0,0,1]
# Contact graph: COM [1,0],     pad [0,1]       (+ rod [0,0,1] only if rod_can_contact)
AERO_COM, AERO_ROTOR, AERO_ROD, AERO_PAD = 0, 1, 2, 3
N_AERO_TYPES = 4


@dataclass
class DroneConfig:
    # ---- mass properties (measure WITH the rod and sensor mounted) ----
    mass: float = 1.5                                   # kg
    # Inertia about the COM, body axes, kg m^2. Full 3x3: the rod makes it non-isotropic.
    inertia: tuple = ((0.015, 0.0, 0.0),
                      (0.0, 0.030, 0.0),
                      (0.0, 0.0, 0.035))
    # COM position in the mocap body frame (m). Mocap origin != COM unless you calibrate it.
    com_offset: tuple = (0.04, 0.0, -0.01)

    # ---- rotors (mocap body frame) ----
    rotor_pos: tuple = ((0.12, 0.12, 0.0), (-0.12, 0.12, 0.0),
                        (-0.12, -0.12, 0.0), (0.12, -0.12, 0.0))
    rotor_axis: tuple = ((0.0, 0.0, 1.0),) * 4
    spin_dir: tuple = (1, -1, 1, -1)       # +1 = rotor spins CCW about its own axis
    k_f: float = 1.0e-5                     # thrust coeff,   T   = k_f * omega^2  (N, omega in rad/s)
    k_m: float = 1.6e-7                     # yaw-moment coeff, Q = k_m * omega^2  (N m)

    # ---- rod / end effector (mocap body frame) ----
    # Angled ~8 deg up: pushing needs nose-down pitch, and a level rod at or
    # below the COM then puts the pad below the COM, where the wall's normal
    # force pitches the drone further into the wall (unstable in the MuJoCo
    # generator). Set these to the real geometry.
    rod_base: tuple = (0.10, 0.0, 0.0)
    rod_tip: tuple = (0.45, 0.0, 0.05)
    n_rod_nodes: int = 3            # rod nodes in the AERO graph (base included, tip excluded);
                                    # the contact graph has none unless rod_can_contact
    tip_type: str = "pad"           # "pad" -> center + rings; "point" -> 1 tip node
    # Pad rings as (radius m, node count), in the plane perpendicular to the rod
    # at rod_tip. More/denser rings = finer pad mesh. In MuJoCo each pad node is
    # a small collision sphere of radius pad_contact_radius, so the gate distance
    # contact_d0 should be about pad_contact_radius plus a margin.
    pad_rings: tuple = ((0.025, 8),)
    pad_contact_radius: float = 0.004
    pad_knn: int = 4                # pad-internal edges: each pad node to its k nearest pad nodes
    rod_can_contact: bool = False   # True lets rod nodes take contact (rod lying on the wall)
    edge_mode: str = "chain"        # "chain" (body-rod-tip + body->tip shortcut + pad kNN) or "full"

    # ---- physics / timing ----
    gravity: float = 9.81
    dt: float = 0.01                # recorded step (s); checked against the data's meta

    # ---- actuation ----
    # fully_actuated: the rotors can make a force in any direction without
    # tilting the body (tilted rotors), so the data generator flies it with a
    # full-wrench controller and can hold any attitude. False = a quad that
    # tilts to move (geometric controller).
    fully_actuated: bool = False
    rotor_thrust_max: float = None  # N per rotor; None -> GenSettings.t_max_over_hover x hover thrust

    # ---- MuJoCo body (sim only: the XML builders read these; the model never does) ----
    # MuJoCo computes mass, inertia, and COM from these geoms, and the generator
    # writes the result back into mass / inertia / com_offset.
    core_half_size: tuple = (0.06, 0.06, 0.025)    # m, box at core_pos
    core_pos: tuple = (0.0, 0.0, 0.0)
    core_mass: float = 0.9
    arm_mass: float = 0.04          # per arm (capsule from the body origin to the hub)
    prop_mass: float = 0.05         # per rotor (motor + prop disk at the hub)
    prop_radius: float = 0.06       # m; the prop disks collide with the wall (crash check)
    rod_mass: float = 0.06

    def to_dict(self):
        return asdict(self)


# Fully actuated hexarotor: AM-Bench's FAHexa (hexa_scorpion.urdf, fa_hexa.py),
# with its 4-DOF arm replaced by the rigid rod + pad.
#   rotors   AM-Bench positions and tilts (every rotor 30 deg off vertical).
#            AM-Bench spin dirs are +1 = CW; here +1 = CCW, so they are flipped
#            (the yaw reaction torque comes out the same).
#   thrust   0-23 N per rotor, k_m/k_f = 0.02 m (AM-Bench reaction-torque ratio).
#            k_f only sets the rad/s scale: hover is ~560 rad/s (~5400 RPM, about
#            right for the 0.152 m props).
#   mass     AM-Bench base without the arm: 5.615 kg, split into core, arms, and
#            motors + props. The core sits 3.7 cm up so the base COM is at the
#            rotor-arm mount height (z = 0). AM-Bench's base inertia (0.198, 0.198,
#            0.460) breaks the triangle inequality (Izz > Ixx + Iyy), which no real
#            body can have and MuJoCo rejects, so the inertia comes from the geoms.
#   rod      level, at the COM height, mounted where AM-Bench's arm is (x = 0.088).
#            Reaches past the front prop edge by the same fraction of the drone's
#            size as on the quad: the quad's rod tip is 0.27 m past its front prop
#            edge (x = 0.18), 1.5x that distance, so here 1.5 x 0.494 m past
#            x = 0.494 -> tip at x = 1.234 m. Mass scaled with length from the quad
#            rod (same tube); the sensor and pad are the same parts.
def fa_hexa_config():
    pos = ((0.34175, 0.1585, -0.058), (-0.0335, 0.375, -0.058), (-0.30825, 0.2165, -0.058),
           (-0.30825, -0.2165, -0.058), (-0.0335, -0.375, -0.058), (0.34175, -0.1585, -0.058))
    rpy = ((-0.464, -0.252, 0.060), (0.0, 0.524, 0.0), (0.464, -0.252, -0.060),
           (-0.464, -0.252, 0.060), (0.0, 0.524, 0.0), (0.464, -0.252, -0.060))
    k_f = 3.5e-5
    return DroneConfig(
        mass=5.85, inertia=((0.15, 0.0, 0.0), (0.0, 0.20, 0.0), (0.0, 0.0, 0.32)),  # replaced by the
        com_offset=(0.03, 0.0, 0.0),                                                # compiled MuJoCo values
        rotor_pos=pos, rotor_axis=tuple(_rpy_z_axis(*a) for a in rpy),
        spin_dir=(-1, 1, -1, 1, -1, 1),
        k_f=k_f, k_m=0.02 * k_f,
        rod_base=(0.088, 0.0, 0.0), rod_tip=(1.234, 0.0, 0.0),
        fully_actuated=True, rotor_thrust_max=23.0,
        core_half_size=(0.15, 0.10, 0.05), core_pos=(0.0, 0.0, 0.0368), core_mass=3.215,
        arm_mass=0.12, prop_mass=0.28, prop_radius=0.152, rod_mass=0.20)


# Rotor axis = body z rotated by URDF roll-pitch-yaw (R = Rz(yaw) Ry(pitch) Rx(roll)),
# the same conversion AM-Bench's rotor_layout() does.
def _rpy_z_axis(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    return (cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr)


PRESETS = {"quad": DroneConfig, "fa_hexa": fa_hexa_config}


# Body-frame wrench about the COM per newton of each rotor's thrust: (6, M),
# rows [Fx, Fy, Fz, Mx, My, Mz]. Same physics as drone_dynamics.rotor_wrench_body:
# thrust along each rotor axis, its lever arm, and the yaw reaction -spin k_m/k_f.
def wrench_matrix(cfg):
    r = np.asarray(cfg.rotor_pos, dtype=float) - np.asarray(cfg.com_offset, dtype=float)
    a = np.asarray(cfg.rotor_axis, dtype=float)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    s = np.asarray(cfg.spin_dir, dtype=float)
    moment = np.cross(r, a) - (s * cfg.k_m / cfg.k_f)[:, None] * a
    return np.vstack([a.T, moment.T])


# Rotor thrusts (N) that hover the drone level. A quad's axes all point up, so
# it is m g / M each; with tilted rotors the full 6-D wrench is solved.
def hover_thrusts(cfg):
    M = len(cfg.rotor_pos)
    if not cfg.fully_actuated:
        return np.full(M, cfg.mass * cfg.gravity / M)
    target = np.array([0.0, 0.0, cfg.mass * cfg.gravity, 0.0, 0.0, 0.0])
    return np.linalg.lstsq(wrench_matrix(cfg), target, rcond=None)[0]


# Returns the COM-relative rotor geometry and the inertia-over-mass tensor as torch tensors.
# J = I/m is the drone's version of the cube's scalar I_OVER_M.
def rotor_and_inertia_tensors(cfg):
    c = np.asarray(cfg.com_offset, dtype=np.float64)
    rotor_pos = np.asarray(cfg.rotor_pos, dtype=np.float64) - c
    axes = np.asarray(cfg.rotor_axis, dtype=np.float64)
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    J = np.asarray(cfg.inertia, dtype=np.float64) / cfg.mass

    # Mean thrust axis: the direction rotor drag acts perpendicular to.
    thrust_axis = axes.sum(0)
    thrust_axis = thrust_axis / np.linalg.norm(thrust_axis)

    f32 = lambda a: torch.tensor(a, dtype=torch.float32)
    return dict(rotor_pos=f32(rotor_pos), rotor_axis=f32(axes),
                spin_dir=f32(np.asarray(cfg.spin_dir, dtype=np.float64)),
                J_body=f32(J), thrust_axis=f32(thrust_axis))


# Builds the node set, node types, contact mask, and edges for the drone graph.
# Node 0 is the body node at the COM; rod nodes follow; then the tip center;
# then the pad-ring nodes. Positions are COM-relative unless com_relative=False,
# in which case they are in the mocap/body frame (what the MuJoCo XML needs).
def build_drone_graph(cfg, com_relative=True):
    c = np.asarray(cfg.com_offset, dtype=np.float64)
    base = np.asarray(cfg.rod_base, dtype=np.float64)
    tip = np.asarray(cfg.rod_tip, dtype=np.float64)

    nodes, types = [c.copy()], [NODE_BODY]

    # Rod nodes, base included, tip excluded.
    for s in np.linspace(0.0, 1.0, cfg.n_rod_nodes, endpoint=False):
        nodes.append(base + s * (tip - base))
        types.append(NODE_ROD)

    # Tip / pad center.
    tip_index = len(nodes)
    nodes.append(tip)
    types.append(NODE_TIP)

    # Pad rings in the plane perpendicular to the rod.
    pad = [tip_index]
    if cfg.tip_type == "pad":
        axis = (tip - base) / np.linalg.norm(tip - base)
        tmp = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(axis, tmp)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        for radius, count in cfg.pad_rings:
            for k in range(count):
                ang = 2.0 * np.pi * k / count
                pad.append(len(nodes))
                nodes.append(tip + radius * (np.cos(ang) * u + np.sin(ang) * v))
                types.append(NODE_TIP)
    elif cfg.tip_type != "point":
        raise ValueError(f"tip_type must be 'point' or 'pad', got {cfg.tip_type!r}")

    nodes = np.asarray(nodes)
    types = np.asarray(types)
    N = len(nodes)

    # Edges. At this node count kNN over the whole body buys nothing, so the
    # body/rod part is hand-wired and only the pad uses kNN (among pad nodes).
    pairs = set()
    if cfg.edge_mode == "full":
        pairs = {(i, j) for i in range(N) for j in range(N) if i != j}
    elif cfg.edge_mode == "chain":
        chain = list(range(tip_index + 1))                  # body, rod..., tip center
        for a, b in zip(chain[:-1], chain[1:]):
            pairs |= {(a, b), (b, a)}
        pairs |= {(0, tip_index), (tip_index, 0)}           # shortcut: body state reaches the pad in one hop
        if len(pad) > 1:
            P = nodes[pad]
            dist = np.linalg.norm(P[:, None] - P[None], axis=-1)
            for i in range(len(pad)):
                for j in np.argsort(dist[i])[1:cfg.pad_knn + 1]:
                    pairs |= {(pad[i], pad[j]), (pad[j], pad[i])}
    else:
        raise ValueError(f"edge_mode must be 'chain' or 'full', got {cfg.edge_mode!r}")
    edge_index = np.array(sorted(pairs)).T

    # Which nodes may carry a contact force.
    contact_mask = types == NODE_TIP
    if cfg.rod_can_contact:
        contact_mask |= types == NODE_ROD

    if com_relative:
        nodes = nodes - c

    return dict(rest_nodes=torch.tensor(nodes, dtype=torch.float32),
                node_type=torch.tensor(types, dtype=torch.long),
                contact_mask=torch.tensor(contact_mask),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                tip_index=tip_index, pad_indices=pad)


# ======================================================================
# The two graphs the model actually runs on
# ======================================================================

# Contact graph: COM hub + pad nodes (rod nodes only if the rod may touch).
# Edges: COM <-> pad center, each pad node <-> its pad_knn nearest pad nodes.
def build_contact_graph(cfg):
    n_rod = cfg.n_rod_nodes if cfg.rod_can_contact else 0
    g = build_drone_graph(dataclasses.replace(cfg, n_rod_nodes=n_rod))
    kind = g["node_type"]
    onehot_index = torch.where(kind == NODE_BODY, 0, torch.where(kind == NODE_TIP, 1, 2))
    n_types = 3 if n_rod > 0 else 2
    g["onehot"] = torch.nn.functional.one_hot(onehot_index, n_types).float()
    return g


# Aero graph: COM, one node per rotor, rod nodes, one pad node at the rod tip.
# Edges (bidirectional): every rotor <-> COM; COM <-> rod_1 <-> ... <-> rod_n <-> pad;
# each front rotor (ahead of the COM along the rod direction) <-> its nearest
# rod node, so information about its wash can reach the rod.
# All positions COM-relative, body frame.
def build_aero_graph(cfg):
    c = np.asarray(cfg.com_offset, dtype=np.float64)
    rotors = np.asarray(cfg.rotor_pos, dtype=np.float64) - c
    axes = np.asarray(cfg.rotor_axis, dtype=np.float64)
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    base = np.asarray(cfg.rod_base, dtype=np.float64) - c
    tip = np.asarray(cfg.rod_tip, dtype=np.float64) - c
    rod_axis = (tip - base) / np.linalg.norm(tip - base)

    nodes, types = [np.zeros(3)], [AERO_COM]
    rotor_idx = []
    for p in rotors:
        rotor_idx.append(len(nodes))
        nodes.append(p)
        types.append(AERO_ROTOR)
    rod_idx = []
    for s in np.linspace(0.0, 1.0, cfg.n_rod_nodes, endpoint=False):
        rod_idx.append(len(nodes))
        nodes.append(base + s * (tip - base))
        types.append(AERO_ROD)
    pad_idx = len(nodes)
    nodes.append(tip)
    types.append(AERO_PAD)
    nodes = np.asarray(nodes)

    pairs = set()
    for i in rotor_idx:
        pairs |= {(0, i), (i, 0)}
    chain = [0] + rod_idx + [pad_idx]
    for a, b in zip(chain[:-1], chain[1:]):
        pairs |= {(a, b), (b, a)}
    if rod_idx:
        for i in rotor_idx:
            if nodes[i] @ rod_axis > 0:
                j = rod_idx[int(np.argmin(np.linalg.norm(nodes[rod_idx] - nodes[i], axis=1)))]
                pairs |= {(i, j), (j, i)}
    edge_index = np.array(sorted(pairs)).T

    t = torch.tensor(types, dtype=torch.long)
    return dict(rest_nodes=torch.tensor(nodes, dtype=torch.float32),
                node_type=t, onehot=torch.nn.functional.one_hot(t, N_AERO_TYPES).float(),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                rotor_idx=rotor_idx, rod_idx=rod_idx, pad_idx=pad_idx,
                rotor_axis=torch.tensor(axes, dtype=torch.float32),
                rod_axis=torch.tensor(rod_axis, dtype=torch.float32))
