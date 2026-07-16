#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "scene" / "scene_winejar_vacuum.xml"

BODIES = [
    "vacuum_end_effector",
    "robot_base_pedestal",
    "wine_jar",
    "ceramic_disc_on_jar",
    "material_table",
    "bamboo_leaf_bottom",
    "bamboo_leaf_top",
    "global_camera_rig",
]

SITES = [
    "vacuum_suction_site",
    "jar_mouth_center",
    "ceramic_disc_center_site",
    "jar_leaf_target_0",
    "jar_leaf_target_90",
    "bamboo_leaf_bottom_pick_site",
    "bamboo_leaf_top_pick_site",
    "leaf_stack_pick_site",
]

CAMERAS = [
    "global_overhead_camera",
]


def fmt(vec):
    return f"({vec[0]: .3f}, {vec[1]: .3f}, {vec[2]: .3f})"


def angle_between_site_x_axes_deg(model, data, first: str, second: str) -> float:
    first_x = data.site_xmat[model.site(first).id].reshape(3, 3)[:, 0]
    second_x = data.site_xmat[model.site(second).id].reshape(3, 3)[:, 0]
    dot = float(np.clip(np.dot(first_x, second_x), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def geom_vertices_xy(model, data, geom_id):
    geom_type = model.geom_type[geom_id]
    xpos = data.geom_xpos[geom_id]
    xmat = data.geom_xmat[geom_id].reshape(3, 3)

    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        sx, sy, sz = model.geom_size[geom_id]
        local = np.array(
            [[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)],
            dtype=float,
        )
    elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[geom_id]
        start = model.mesh_vertadr[mesh_id]
        count = model.mesh_vertnum[mesh_id]
        local = model.mesh_vert[start : start + count]
    else:
        return np.empty((0, 2), dtype=float)

    world = xpos + local @ xmat.T
    return world[:, :2]


def body_aabb_xy(model, data, body_name):
    body_id = model.body(body_name).id
    points = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == body_id:
            xy = geom_vertices_xy(model, data, geom_id)
            if len(xy):
                points.append(xy)
    if not points:
        raise AssertionError(f"{body_name}: no geometry found")
    all_points = np.vstack(points)
    return all_points.min(axis=0), all_points.max(axis=0)


def assert_aabb_inside(inner_name, inner, outer_name, outer, margin=0.005):
    inner_min, inner_max = inner
    outer_min, outer_max = outer
    if np.any(inner_min < outer_min + margin) or np.any(inner_max > outer_max - margin):
        raise AssertionError(
            f"{inner_name} is not fully inside {outer_name} with {margin:.3f} m margin: "
            f"inner=({inner_min.round(3).tolist()}, {inner_max.round(3).tolist()}), "
            f"outer=({outer_min.round(3).tolist()}, {outer_max.round(3).tolist()})"
        )


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"Loaded scene: {SCENE_PATH}")
    print(f"nbody={model.nbody}, ngeom={model.ngeom}, nsite={model.nsite}, ncam={model.ncam}")

    print("\nBodies:")
    for name in BODIES:
        body_id = model.body(name).id
        print(f"  {name}: {fmt(data.xpos[body_id])}")

    print("\nSites:")
    for name in SITES:
        site_id = model.site(name).id
        print(f"  {name}: {fmt(data.site_xpos[site_id])}")

    print("\nCameras:")
    for name in CAMERAS:
        cam_id = model.camera(name).id
        print(f"  {name}: {fmt(data.cam_xpos[cam_id])}")

    angle = angle_between_site_x_axes_deg(model, data, "jar_leaf_target_0", "jar_leaf_target_90")
    if abs(angle - 90.0) > 0.5:
        raise AssertionError(f"leaf target angle expected 90 deg, got {angle:.3f}")
    print(f"\nLeaf target crossing angle: {angle:.2f} deg")

    top_z = data.xpos[model.body("bamboo_leaf_top").id][2]
    bottom_z = data.xpos[model.body("bamboo_leaf_bottom").id][2]
    if top_z <= bottom_z:
        raise AssertionError("top leaf must be above bottom leaf")
    print(f"Leaf stack vertical gap: {(top_z - bottom_z) * 1000.0:.1f} mm")

    table_aabb = body_aabb_xy(model, data, "material_table")
    for leaf_name in ("bamboo_leaf_bottom", "bamboo_leaf_top"):
        leaf_aabb = body_aabb_xy(model, data, leaf_name)
        assert_aabb_inside(leaf_name, leaf_aabb, "material_table", table_aabb)
    print("Leaf stack is fully inside material table footprint.")


if __name__ == "__main__":
    main()
