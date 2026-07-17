"""Visible cable-tie transfer and tightening around each jar neck."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_xarm6.production_demo.clock import smoothstep


@dataclass
class TieBandTransition:
    jar_index: int
    start_radius_m: float
    target_radius_m: float
    duration_s: float
    elapsed_s: float = 0.0


class TieBandController:
    """Shrink one white band from the tool ring onto a jar neck."""

    INITIAL_RADIUS_M = 0.105
    FINAL_RADIUS_M = 0.063
    BAND_RADIUS_M = 0.0022
    SEGMENT_COUNT = 16
    COLOR = np.array([0.96, 0.96, 0.93, 1.0], dtype=np.float64)

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.geom_ids_by_jar = {
            1: self._geom_ids("final_tie_band_visual_geom_", padded=True),
            2: self._geom_ids("jar_02_final_tie_band_visual_geom_", padded=True),
            3: self._geom_ids("jar_03_final_tie_band_visual_geom_", padded=True),
        }
        self.loaded_geom_ids = self._geom_ids("right_tie_gun_loaded_band_", padded=False)
        self.radii_m: dict[int, float] = {}
        self.visible_jars: set[int] = set()
        self.transition: TieBandTransition | None = None
        self.reset()

    def _geom_ids(self, prefix: str, padded: bool) -> tuple[int, ...]:
        ids = tuple(
            self.model.geom(f"{prefix}{index:02d}" if padded else f"{prefix}{index}").id
            for index in range(self.SEGMENT_COUNT)
        )
        return ids

    def reset(self):
        self.radii_m = {index: self.INITIAL_RADIUS_M for index in self.geom_ids_by_jar}
        self.visible_jars.clear()
        self.transition = None
        self.set_loaded_visible(True)
        self.apply()

    def tighten(self, jar_index: int, duration_s: float):
        self.visible_jars.add(jar_index)
        self.radii_m[jar_index] = self.INITIAL_RADIUS_M
        self.set_loaded_visible(False)
        if duration_s <= 0.0:
            self.radii_m[jar_index] = self.FINAL_RADIUS_M
            self.transition = None
            self.apply()
            return
        self.transition = TieBandTransition(
            jar_index=jar_index,
            start_radius_m=self.INITIAL_RADIUS_M,
            target_radius_m=self.FINAL_RADIUS_M,
            duration_s=float(duration_s),
        )
        self.apply()

    def advance(self, dt: float):
        if self.transition is None:
            return
        transition = self.transition
        transition.elapsed_s = min(transition.duration_s, transition.elapsed_s + dt)
        alpha = smoothstep(transition.elapsed_s / transition.duration_s)
        self.radii_m[transition.jar_index] = (
            transition.start_radius_m
            + (transition.target_radius_m - transition.start_radius_m) * alpha
        )
        self.apply()
        if transition.elapsed_s >= transition.duration_s:
            self.transition = None

    def set_loaded_visible(self, visible: bool):
        alpha = 1.0 if visible else 0.0
        for geom_id in self.loaded_geom_ids:
            self.model.geom_rgba[geom_id] = self.COLOR
            self.model.geom_rgba[geom_id, 3] = alpha

    def apply(self):
        for jar_index, geom_ids in self.geom_ids_by_jar.items():
            radius = self.radii_m[jar_index]
            alpha = 1.0 if jar_index in self.visible_jars else 0.0
            for segment, geom_id in enumerate(geom_ids):
                angle_a = 2.0 * np.pi * segment / self.SEGMENT_COUNT
                angle_b = 2.0 * np.pi * (segment + 1) / self.SEGMENT_COUNT
                start = np.array([radius * np.cos(angle_a), radius * np.sin(angle_a), 0.0])
                end = np.array([radius * np.cos(angle_b), radius * np.sin(angle_b), 0.0])
                self._set_capsule(geom_id, start, end)
                self.model.geom_size[geom_id, 0] = self.BAND_RADIUS_M
                self.model.geom_rgba[geom_id] = self.COLOR
                self.model.geom_rgba[geom_id, 3] = alpha

    def _set_capsule(self, geom_id: int, start: np.ndarray, end: np.ndarray):
        direction = end - start
        length = float(np.linalg.norm(direction))
        self.model.geom_pos[geom_id] = (start + end) * 0.5
        self.model.geom_size[geom_id, 1] = length * 0.5
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_quatZ2Vec(quat, direction / max(length, 1e-9))
        self.model.geom_quat[geom_id] = quat
