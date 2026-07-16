"""Cartesian waypoint subdivision and inverse-kinematics path planning."""
from __future__ import annotations

import math

import mujoco
import numpy as np

from mujoco_xarm6.production_demo.models import JointPath, PathEvent
from mujoco_xarm6.sim.xarm_sim_api import rpy_to_mat


class JointPathPlanner:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

    def _solve(self, arm, joint_addrs: np.ndarray, seed_q: np.ndarray, point_m: np.ndarray, yaw_deg: float) -> np.ndarray:
        original = self.data.qpos[joint_addrs].copy()
        self.data.qpos[joint_addrs] = seed_q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        solved = arm._solve_ik(point_m, rpy_to_mat(180.0, 0.0, yaw_deg))
        self.data.qpos[joint_addrs] = original
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        if solved is None:
            raise RuntimeError(f"IK failed for {arm.tcp_site_name} at {point_m.tolist()}, yaw={yaw_deg}")
        return solved

    @staticmethod
    def cartesian_samples(start: np.ndarray, target: np.ndarray, start_yaw: float, target_yaw: float) -> list[tuple[np.ndarray, float]]:
        distance = float(np.linalg.norm(target - start))
        yaw_delta = ((target_yaw - start_yaw + 180.0) % 360.0) - 180.0
        count = max(1, int(math.ceil(distance / 0.075)), int(math.ceil(abs(yaw_delta) / 22.5)))
        return [
            (start + (target - start) * (index / count), start_yaw + yaw_delta * (index / count))
            for index in range(1, count + 1)
        ]

    def build(
        self,
        name: str,
        arm,
        joint_addrs: np.ndarray,
        dof_addrs: np.ndarray,
        start_q: np.ndarray,
        targets: list[tuple[np.ndarray, float]],
        speed: float,
        events: list[PathEvent] | None = None,
    ) -> JointPath:
        points = [start_q.copy()]
        current_q = start_q.copy()
        previous_tcp = self.data.site_xpos[arm.tcp_site_id].copy()
        previous_yaw = 0.0
        for target, yaw in targets:
            for point, point_yaw in self.cartesian_samples(previous_tcp, target, previous_yaw, yaw):
                current_q = self._solve(arm, joint_addrs, current_q, point, point_yaw)
                points.append(current_q)
            previous_tcp = target.copy()
            previous_yaw = yaw
        durations = [max(0.025, float(np.max(np.abs(end - start))) / max(speed, 1e-6)) for start, end in zip(points, points[1:])]
        return JointPath(name, joint_addrs, dof_addrs, points, durations, events or [])
