#!/usr/bin/env python3
"""Static integrity checks for the parallel production-line scene."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scene" / "scene_winejar_production_demo.xml"


def require(model, object_type, name: str):
    if mujoco.mj_name2id(model, object_type, name) < 0:
        raise AssertionError(f"Missing {object_type.name}: {name}")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for body in ("station_wine_jar", "station_wine_jar_02", "station_wine_jar_03", "left_line_base", "right_line_base"):
        require(model, mujoco.mjtObj.mjOBJ_BODY, body)
    for index in (1, 2, 3):
        prefix = "staged_bamboo_leaf" if index == 1 else f"jar_{index:02d}_bamboo_leaf"
        for side in ("top", "bottom"):
            require(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_{side}")
            require(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}_{side}_pick_site")
            for segment in range(11):
                require(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_{side}_seg_{segment:02d}")
                geom = model.geom(f"{prefix}_{side}_seg_{segment:02d}_geom")
                if int(geom.contype[0]) == 0 or int(geom.conaffinity[0]) == 0:
                    raise AssertionError(f"Leaf segment must retain a physical collision volume: {prefix}_{side}_seg_{segment:02d}")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "staged_label_paper") >= 0:
        raise AssertionError("Label paper should be absent from the revised production scene")
    for camera in ("global_overview_camera", "front_conveyor_camera", "side_overview_camera"):
        require(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)

    tie_collision_geoms = (
        "right_tie_gun_body",
        "right_tie_gun_support_rod_front",
        "right_tie_gun_closed_jaw_0",
        "right_tie_gun_loaded_band_0",
    )
    for geom_name in tie_collision_geoms:
        geom = model.geom(geom_name)
        if int(geom.contype[0]) == 0 or int(geom.conaffinity[0]) == 0:
            raise AssertionError(f"Tie-gun contact proxy is disabled: {geom_name}")

    for geom_name in ("preloaded_lotus_leaf_geom", "jar_02_preloaded_lotus_leaf_geom", "jar_03_preloaded_lotus_leaf_geom"):
        geom = model.geom(geom_name)
        if int(geom.dataid[0]) != model.mesh("cropped_lotus_leaf_tie_safe_mesh").id:
            raise AssertionError(f"Lotus leaf must use the reduced tie-safe mesh: {geom_name}")

    stack_centers = []
    for index in (1, 2, 3):
        prefix = "staged_bamboo_leaf" if index == 1 else f"jar_{index:02d}_bamboo_leaf"
        stack_centers.extend(
            data.xpos[model.body(f"{prefix}_{side}").id].copy()
            for side in ("top", "bottom")
        )
    stack_xy = np.asarray([center[:2] for center in stack_centers])
    if np.max(np.ptp(stack_xy, axis=0)) > 1e-4:
        raise AssertionError(f"All bamboo leaves must start in one physical material stack: {stack_xy}")
    stack_z = np.sort(np.asarray([center[2] for center in stack_centers]))
    gaps_mm = np.diff(stack_z) * 1000.0
    if np.any(gaps_mm < 3.1) or np.any(gaps_mm > 3.7):
        raise AssertionError(f"Leaf stack must show six distinct physical layers, got gaps {gaps_mm}")

    left_base = data.xpos[model.body("left_line_base").id]
    right_base = data.xpos[model.body("right_line_base").id]
    if right_base[0] - left_base[0] < 0.45:
        raise AssertionError(f"Tie-gun arm must be downstream of vacuum arm, got dx={right_base[0] - left_base[0]:.3f} m")
    camera_xy = data.xpos[model.body("global_camera_rig").id][:2]
    pedestals = (data.xpos[model.body("left_robot_pedestal").id][:2] + data.xpos[model.body("right_robot_pedestal").id][:2]) / 2.0
    if np.linalg.norm(camera_xy - pedestals) > 0.015:
        raise AssertionError(f"Global camera must be centered over both stations: {camera_xy} vs {pedestals}")
    print(f"Loaded parallel scene: bodies={model.nbody}, joints={model.njnt}, geoms={model.ngeom}, equalities={model.neq}")
    print("Three-jar production scene checks OK")


if __name__ == "__main__":
    main()
