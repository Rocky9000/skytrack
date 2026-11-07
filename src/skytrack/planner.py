"""MPPI (Model Predictive Path Integral) planner for flying around wires.

Each control step samples K noisy acceleration sequences around the current
plan, rolls them out through a point-mass model, scores each trajectory, and
averages the samples with weights ``exp(-cost / lambda)``. Nothing is learned:
the behaviour comes from the dynamics model and the cost terms, so it can be
reasoned about and tuned on the bench, which matters for a safety layer.

Wires come from ``wires.py`` as 3D segments in the world frame. The planner
outputs a velocity setpoint; ``velocity_setpoint_ned`` packs it for a PX4
``SET_POSITION_TARGET_LOCAL_NED`` message.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MPPIConfig:
    horizon: int = 30           # steps
    dt: float = 0.05            # s, so the horizon looks 1.5 s ahead
    samples: int = 512
    noise_std: float = 2.5      # m/s^2 per axis
    temperature: float = 1.0    # lambda
    max_accel: float = 4.0      # m/s^2
    max_speed: float = 5.0      # m/s
    min_alt: float = 2.0        # m above ground
    safety_radius: float = 1.0  # m; closer than this to a wire is a collision
    w_goal: float = 1.0
    w_terminal: float = 10.0
    w_control: float = 0.02
    w_obstacle: float = 50.0
    w_collision: float = 1e4
    seed: int = 0


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points p (..., 3) to each segment (M, 3)-(M, 3). Returns (..., M)."""
    ab = b - a
    ap = p[..., None, :] - a
    t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.linalg.norm(p[..., None, :] - closest, axis=-1)


@dataclass
class MPPIPlanner:
    cfg: MPPIConfig = field(default_factory=MPPIConfig)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.cfg.seed)
        self.plan = np.zeros((self.cfg.horizon, 3))

    def reset(self) -> None:
        self.plan[:] = 0.0

    def _rollout(self, pos: np.ndarray, vel: np.ndarray, accel: np.ndarray) -> np.ndarray:
        c = self.cfg
        k = accel.shape[0]
        traj = np.empty((k, c.horizon, 3))
        p = np.broadcast_to(pos, (k, 3)).copy()
        v = np.broadcast_to(vel, (k, 3)).copy()
        for t in range(c.horizon):
            v = v + accel[:, t] * c.dt
            speed = np.linalg.norm(v, axis=1, keepdims=True)
            v = np.where(speed > c.max_speed, v / np.maximum(speed, 1e-9) * c.max_speed, v)
            p = p + v * c.dt
            traj[:, t] = p
        return traj

    def _cost(self, traj: np.ndarray, goal: np.ndarray, wires: np.ndarray | None) -> np.ndarray:
        c = self.cfg
        cost = c.w_goal * np.linalg.norm(traj - goal, axis=-1).sum(1) * c.dt
        cost += c.w_terminal * np.linalg.norm(traj[:, -1] - goal, axis=-1)
        low = np.clip(c.min_alt - traj[..., 2], 0, None)
        cost += c.w_collision * (low > 0).sum(1) + c.w_obstacle * (low**2).sum(1)
        if wires is not None and len(wires):
            d = point_segment_distance(traj, wires[:, 0], wires[:, 1]).min(-1)  # (K, T)
            cost += c.w_collision * (d < c.safety_radius).sum(1)
            cost += c.w_obstacle * np.exp(-(d - c.safety_radius) / 0.5).sum(1) * c.dt
        return cost

    def step(self, pos, vel, goal, wires: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """One planning cycle.

        Args:
            pos, vel: current state in the world frame (3,).
            goal: target position (3,).
            wires: (M, 2, 3) wire segments, or None.

        Returns:
            ``(accel, trajectory)``: the acceleration to apply now and the
            nominal trajectory of the updated plan (horizon, 3).
        """
        c = self.cfg
        pos, vel, goal = (np.asarray(x, np.float64) for x in (pos, vel, goal))
        wires = None if wires is None else np.asarray(wires, np.float64).reshape(-1, 2, 3)

        eps = self.rng.normal(0.0, c.noise_std, size=(c.samples, c.horizon, 3))
        eps[0] = 0.0  # always evaluate the current plan itself
        accel = np.clip(self.plan[None] + eps, -c.max_accel, c.max_accel)
        cost = self._cost(self._rollout(pos, vel, accel), goal, wires)
        # Control cost from the path-integral derivation: lambda * u^T Sigma^-1 eps
        cost += c.w_control * c.temperature * (self.plan[None] * (accel - self.plan[None])).sum((1, 2)) / c.noise_std**2

        w = np.exp(-(cost - cost.min()) / c.temperature)
        w /= w.sum()
        self.plan = np.clip((w[:, None, None] * accel).sum(0), -c.max_accel, c.max_accel)

        u = self.plan[0].copy()
        nominal = self._rollout(pos, vel, self.plan[None])[0]
        self.plan = np.roll(self.plan, -1, axis=0)  # warm start for the next cycle
        self.plan[-1] = self.plan[-2]
        return u, nominal


def simulate(planner: MPPIPlanner, start, goal, wires=None, steps: int = 400, tol: float = 0.5) -> dict:
    """Closed-loop run with perfect state feedback. Returns the flown path and summary stats."""
    c = planner.cfg
    pos, vel, goal = np.asarray(start, float), np.zeros(3), np.asarray(goal, float)
    path = [pos.copy()]
    for _ in range(steps):
        u, _ = planner.step(pos, vel, goal, wires)
        vel = vel + u * c.dt
        speed = np.linalg.norm(vel)
        if speed > c.max_speed:
            vel *= c.max_speed / speed
        pos = pos + vel * c.dt
        path.append(pos.copy())
        if np.linalg.norm(pos - goal) < tol and np.linalg.norm(vel) < 0.5:
            break
    path = np.asarray(path)
    clearance = np.inf
    if wires is not None and len(wires):
        w = np.asarray(wires, float).reshape(-1, 2, 3)
        clearance = float(point_segment_distance(path, w[:, 0], w[:, 1]).min())
    return {
        "path": path,
        "reached": bool(np.linalg.norm(path[-1] - goal) < tol),
        "steps": len(path) - 1,
        "min_clearance_m": clearance,
        "min_alt_m": float(path[:, 2].min()),
    }


# PX4 SET_POSITION_TARGET_LOCAL_NED type_mask: ignore position, acceleration, yaw and yaw rate.
VELOCITY_ONLY_TYPE_MASK = 0b110111000111  # 3527
MAV_FRAME_LOCAL_NED = 1


def velocity_setpoint_ned(vel_enu: np.ndarray) -> dict:
    """Fields for a SET_POSITION_TARGET_LOCAL_NED velocity setpoint (ENU -> NED)."""
    e, n, u = (float(x) for x in vel_enu)
    return {"coordinate_frame": MAV_FRAME_LOCAL_NED, "type_mask": VELOCITY_ONLY_TYPE_MASK,
            "vx": n, "vy": e, "vz": -u}
