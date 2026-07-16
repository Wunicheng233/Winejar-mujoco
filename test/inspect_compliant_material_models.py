#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "scene" / "scene_winejar_production_demo.xml"
SEGMENT_COUNT = 11
MIN_VISIBLE_DROOP_M = 0.010


def body_exists(model, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0


def site_exists(model, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0


def check_leaf_structure(model, leaf_name: str):
    if not body_exists(model, leaf_name):
        raise AssertionError(f"Missing leaf root body: {leaf_name}")
    missing = [
        f"{leaf_name}_seg_{index:02d}"
        for index in range(SEGMENT_COUNT)
        if not body_exists(model, f"{leaf_name}_seg_{index:02d}")
    ]
    if missing:
        raise AssertionError(f"Missing segment bodies for {leaf_name}: {missing}")
    for site_name in [f"{leaf_name}_pick_site", f"{leaf_name}_center_site"]:
        if not site_exists(model, site_name):
            raise AssertionError(f"Missing site: {site_name}")


def check_table_support(model, data, leaf_name: str, expected_bottom_z: float):
    for index in range(SEGMENT_COUNT):
        body_id = model.body(f"{leaf_name}_seg_{index:02d}").id
        bottom_z = float(data.xpos[body_id][2] - 0.003 / 2.0)
        gap = bottom_z - expected_bottom_z
        print(f"  {leaf_name}_seg_{index:02d} support gap: {gap * 1000.0:.2f} mm")
        if abs(gap) > 0.002:
            raise AssertionError(f"{leaf_name}_seg_{index:02d} is not supported correctly, gap={gap:.4f} m")


def check_nonrigid_response(model, leaf_name: str):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    middle_joint = model.joint(f"{leaf_name}_bend_{SEGMENT_COUNT // 2:02d}")
    middle_qpos = middle_joint.qposadr[0]
    middle_dof = middle_joint.dofadr[0]
    before = float(data.qpos[middle_qpos])
    for _ in range(160):
        data.qfrc_applied[middle_dof] = 0.002
        mujoco.mj_step(model, data)
    data.qfrc_applied[:] = 0
    mujoco.mj_forward(model, data)
    after = float(data.qpos[middle_qpos])
    delta = abs(after - before)
    print(f"  {leaf_name} middle bend response: {delta:.6f} rad")
    if delta < 0.0005:
        raise AssertionError(f"{leaf_name} did not show hinge bending response")

    root_id = model.body(leaf_name).id
    middle_id = model.body(f"{leaf_name}_seg_{SEGMENT_COUNT // 2:02d}").id
    root_pos = data.xpos[root_id].copy()
    middle_pos = data.xpos[middle_id].copy()
    distance = float(np.linalg.norm(middle_pos - root_pos))
    print(f"  {leaf_name} root-to-middle distance after response: {distance:.4f} m")


def set_leaf_pose(model, data, leaf_name: str, pos: np.ndarray, yaw_rad: float):
    qposadr = model.joint(f"{leaf_name}_freejoint").qposadr[0]
    data.qpos[qposadr : qposadr + 3] = pos
    data.qpos[qposadr + 3 : qposadr + 7] = [
        np.cos(yaw_rad / 2.0),
        0.0,
        0.0,
        np.sin(yaw_rad / 2.0),
    ]


def check_visible_jar_droop(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("production_home").id)
    mujoco.mj_forward(model, data)
    jar_mouth = data.site_xpos[model.site("jar_mouth_center").id].copy()
    set_leaf_pose(model, data, "staged_bamboo_leaf_top", jar_mouth + np.array([0.0, 0.0, 0.030]), 0.0)
    set_leaf_pose(model, data, "staged_bamboo_leaf_bottom", jar_mouth + np.array([0.0, 0.0, 0.036]), np.pi / 2.0)
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    for _ in range(5000):
        mujoco.mj_step(model, data)

    for leaf_name in ("staged_bamboo_leaf_top", "staged_bamboo_leaf_bottom"):
        center_z = float(data.site_xpos[model.site(f"{leaf_name}_center_site").id][2])
        end_z_values = [
            float(data.xpos[model.body(f"{leaf_name}_seg_00").id][2]),
            float(data.xpos[model.body(f"{leaf_name}_seg_{SEGMENT_COUNT - 1:02d}").id][2]),
        ]
        droops = [center_z - end_z for end_z in end_z_values]
        print(f"  {leaf_name} jar droop: {[round(value * 1000.0, 1) for value in droops]} mm")
        if max(droops) < MIN_VISIBLE_DROOP_M:
            raise AssertionError(
                f"{leaf_name} does not visibly droop on jar mouth, max droop={max(droops) * 1000.0:.1f} mm"
            )


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"Loaded scene: {SCENE_PATH}")
    print(f"nbody={model.nbody}, njnt={model.njnt}, ngeom={model.ngeom}, nsite={model.nsite}, nq={model.nq}, nv={model.nv}")

    table_top_z = data.xpos[model.body("left_material_table").id][2] + 0.24
    for leaf_name, expected_bottom_z in [
        ("staged_bamboo_leaf_bottom", table_top_z),
        ("staged_bamboo_leaf_top", table_top_z + 0.003),
    ]:
        print(f"\nChecking {leaf_name}")
        check_leaf_structure(model, leaf_name)
        check_table_support(model, data, leaf_name, expected_bottom_z)
        check_nonrigid_response(model, leaf_name)
    check_visible_jar_droop(model)
    print("\nCompliant bamboo leaf checks OK")


if __name__ == "__main__":
    main()
