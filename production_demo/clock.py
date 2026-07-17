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
    realtime_render_hz: float = 60.0
    held_joint_addrs: np.ndarray | None = None
    held_dof_addrs: np.ndarray | None = None
    held_joint_values: np.ndarray | None = None
    extra_diagnostics: dict = field(default_factory=dict)
    _wall_start: float | None = field(default=None, init=False, repr=False)
    _sim_start: float | None = field(default=None, init=False, repr=False)
    _next_render_time: float | None = field(default=None, init=False, repr=False)

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

    def reset_timing(self):
        """Start a fresh wall-clock schedule after a MuJoCo data reset."""
        self._wall_start = None
        self._sim_start = None
        self._next_render_time = None

    def _sync_viewer(self):
        if self.viewer is None:
            return
        if not self.realtime:
            self.viewer.sync()
            return
        if self._next_render_time is None or self.data.time + 1e-12 >= self._next_render_time:
            self.viewer.sync()
            frame_period = 1.0 / max(self.realtime_render_hz, 1.0)
            self._next_render_time = self.data.time + frame_period

    def _pace_realtime(self):
        if not self.realtime:
            return
        now = time.monotonic()
        if self._wall_start is None or self._sim_start is None:
            self._wall_start = now
            self._sim_start = float(self.data.time)
            return
        target_wall_time = self._wall_start + (float(self.data.time) - self._sim_start) / max(self.speed_scale, 1e-6)
        remaining = target_wall_time - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)

    def step(self, steps: int = 1, after_step=None):
        for _ in range(max(1, steps)):
            mujoco.mj_step(self.model, self.data)
            if self.held_joint_addrs is not None and self.held_dof_addrs is not None and self.held_joint_values is not None:
                self.data.qpos[self.held_joint_addrs] = self.held_joint_values
                self.data.qvel[self.held_dof_addrs] = 0
                mujoco.mj_forward(self.model, self.data)
            if after_step is not None:
                after_step()
            self._sync_viewer()
            self._pace_realtime()
