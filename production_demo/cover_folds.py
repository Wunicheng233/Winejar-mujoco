"""Synchronized folding for the preloaded lotus leaf and white paper."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_xarm6.production_demo.clock import smoothstep


@dataclass
class CoverFoldTransition:
    joint_ids: tuple[int, ...]
    start_angles: np.ndarray
    target_angles: np.ndarray
    duration_s: float
    elapsed_s: float = 0.0


class CoverFoldController:
    """Hold and animate the four edge flaps of each jar's two cover layers."""

    SIDES = ("east", "west", "north", "south")

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.joint_ids_by_jar = {
            index: tuple(
                model.joint(f"{prefix}_{layer}_fold_{side}").id
                for layer in ("lotus", "paper")
                for side in self.SIDES
            )
            for index, prefix in (
                (1, "preloaded"),
                (2, "jar_02_preloaded"),
                (3, "jar_03_preloaded"),
            )
        }
        self.held_angles: dict[int, np.ndarray] = {}
        self.transitions: dict[int, CoverFoldTransition] = {}

    def reset(self):
        self.transitions.clear()
        self.held_angles = {index: np.zeros(8, dtype=np.float64) for index in self.joint_ids_by_jar}
        self.apply()

    def transition(self, jar_index: int, lotus_angle: float, duration_s: float, paper_angle: float | None = None):
        joint_ids = self.joint_ids_by_jar[jar_index]
        start_angles = np.asarray([self.data.qpos[self.model.jnt_qposadr[joint_id]] for joint_id in joint_ids])
        target_angles = np.asarray([lotus_angle] * 4 + [paper_angle if paper_angle is not None else lotus_angle] * 4)
        if duration_s <= 0.0:
            self.held_angles[jar_index] = target_angles
            self.transitions.pop(jar_index, None)
            self.apply()
            return
        self.transitions[jar_index] = CoverFoldTransition(
            joint_ids=joint_ids,
            start_angles=start_angles,
            target_angles=target_angles,
            duration_s=float(duration_s),
        )

    def advance(self, dt: float):
        completed: list[int] = []
        for jar_index, transition in self.transitions.items():
            transition.elapsed_s = min(transition.duration_s, transition.elapsed_s + dt)
            alpha = smoothstep(transition.elapsed_s / transition.duration_s)
            angles = transition.start_angles + (transition.target_angles - transition.start_angles) * alpha
            self._write_angles(transition.joint_ids, angles)
            if transition.elapsed_s >= transition.duration_s:
                self.held_angles[jar_index] = transition.target_angles.copy()
                completed.append(jar_index)
        for jar_index in completed:
            self.transitions.pop(jar_index, None)
        self.apply(exclude=set(self.transitions))

    def apply(self, exclude: set[int] | None = None):
        excluded = exclude or set()
        for jar_index, angles in self.held_angles.items():
            if jar_index not in excluded:
                self._write_angles(self.joint_ids_by_jar[jar_index], angles)

    def angles(self, jar_index: int) -> np.ndarray:
        return np.asarray(
            [self.data.qpos[self.model.jnt_qposadr[joint_id]] for joint_id in self.joint_ids_by_jar[jar_index]]
        )

    def _write_angles(self, joint_ids: tuple[int, ...], angles: np.ndarray):
        for joint_id, angle in zip(joint_ids, angles):
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = angle
            self.data.qvel[self.model.jnt_dofadr[joint_id]] = 0.0
