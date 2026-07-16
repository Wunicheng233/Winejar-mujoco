from __future__ import annotations

import time
from dataclasses import dataclass, field

import mujoco
import numpy as np


def smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


@dataclass
class AnimationClock:
    model: mujoco.MjModel
    data: mujoco.MjData
    viewer: object | None
    realtime: bool
    speed_scale: float
    held_joint_addrs: np.ndarray | None = None
    held_dof_addrs: np.ndarray | None = None
    held_joint_values: np.ndarray | None = None
    extra_diagnostics: dict = field(default_factory=dict)

    def hold_joints(self, qpos_addrs: np.ndarray, values: np.ndarray, dof_addrs: np.ndarray):
        self.held_joint_addrs = qpos_addrs.copy()
        self.held_dof_addrs = dof_addrs.copy()
        self.held_joint_values = values.copy()
        self.data.qpos[self.held_joint_addrs] = self.held_joint_values
        self.data.qvel[self.held_dof_addrs] = 0
        mujoco.mj_forward(self.model, self.data)

    def clear_joint_hold(self):
        self.held_joint_addrs = None
        self.held_dof_addrs = None
        self.held_joint_values = None

    def step(self, steps: int = 1, after_step=None):
        for _ in range(max(1, steps)):
            mujoco.mj_step(self.model, self.data)
            if self.held_joint_addrs is not None and self.held_dof_addrs is not None and self.held_joint_values is not None:
                self.data.qpos[self.held_joint_addrs] = self.held_joint_values
                self.data.qvel[self.held_dof_addrs] = 0
                mujoco.mj_forward(self.model, self.data)
            if after_step is not None:
                after_step()
            if self.viewer is not None:
                self.viewer.sync()
            if self.realtime:
                time.sleep(float(self.model.opt.timestep) / max(self.speed_scale, 1e-6))
