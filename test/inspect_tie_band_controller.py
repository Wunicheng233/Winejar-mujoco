#!/usr/bin/env python3
"""Regression checks for visible cable-tie tightening."""
from __future__ import annotations

import mujoco
import numpy as np

from mujoco_xarm6.production_demo.constants import SCENE_PATH
from mujoco_xarm6.production_demo.tie_bands import TieBandController


def capsule_endpoints(model: mujoco.MjModel, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
    axis = rotation.reshape(3, 3)[:, 2]
    half_length = float(model.geom_size[geom_id, 1])
    center = model.geom_pos[geom_id]
    return center - axis * half_length, center + axis * half_length


def assert_closed_band(model: mujoco.MjModel, geom_ids: tuple[int, ...]):
    endpoints = [capsule_endpoints(model, geom_id) for geom_id in geom_ids]
    for index, (_, end) in enumerate(endpoints):
        next_start = endpoints[(index + 1) % len(endpoints)][0]
        if np.linalg.norm(end - next_start) > 1e-8:
            raise AssertionError(f"Tie band has a visible gap at segment {index}")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    controller = TieBandController(model)
    first_band = controller.geom_ids_by_jar[1]
    if np.any(model.geom_rgba[list(first_band), 3] != 0.0):
        raise AssertionError("Unused final tie bands must begin hidden")

    controller.tighten(1, 0.30)
    if not np.all(model.geom_rgba[list(first_band), :3] > 0.9):
        raise AssertionError("The tightening cable tie must be visibly white")
    assert_closed_band(model, first_band)
    initial_radius = controller.radii_m[1]
    for _ in range(75):
        controller.advance(0.002)
    midpoint_radius = controller.radii_m[1]
    if not controller.FINAL_RADIUS_M < midpoint_radius < initial_radius:
        raise AssertionError(f"Tie radius did not shrink continuously: {midpoint_radius}")
    while controller.transition is not None:
        controller.advance(0.002)
    if abs(controller.radii_m[1] - controller.FINAL_RADIUS_M) > 1e-9:
        raise AssertionError("Tie band did not finish against the jar neck")
    if np.any(model.geom_rgba[list(first_band), 3] != 1.0):
        raise AssertionError("Finished tie band must remain on the jar")
    assert_closed_band(model, first_band)
    print("Tie band tightening controller checks OK")


if __name__ == "__main__":
    main()
