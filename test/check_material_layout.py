#!/usr/bin/env python3
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "scene" / "scene_winejar.xml"
MATERIAL_BODIES = [
    "lotus_leaf",
    "white_paper",
    "ceramic_disc",
    "bamboo_leaf_1",
    "bamboo_leaf_2",
    "label_paper",
]
GRASP_SITES = [
    "lotus_leaf_grasp_site",
    "white_paper_grasp_site",
    "ceramic_disc_grasp_site",
    "bamboo_leaf_1_grasp_site",
    "bamboo_leaf_2_grasp_site",
    "label_paper_grasp_site",
]


def geom_vertices_xy(model, data, geom_id):
    geom_type = model.geom_type[geom_id]
    xpos = data.geom_xpos[geom_id]
    xmat = data.geom_xmat[geom_id].reshape(3, 3)

    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        sx, sy, sz = model.geom_size[geom_id]
        local = np.array(
            [
                [x, y, z]
                for x in (-sx, sx)
                for y in (-sy, sy)
                for z in (-sz, sz)
            ],
            dtype=float,
        )
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius = model.geom_size[geom_id][0]
        angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        local = np.array(
            [[radius * np.cos(a), radius * np.sin(a), 0.0] for a in angles],
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


def geom_top_z(model, data, geom_name):
    geom_id = model.geom(geom_name).id
    geom_type = model.geom_type[geom_id]
    if geom_type != mujoco.mjtGeom.mjGEOM_BOX:
        raise AssertionError(f"{geom_name}: expected box geom")
    return data.geom_xpos[geom_id][2] + model.geom_size[geom_id][2]


def assert_close(name: str, actual: float, expected: float, tolerance: float = 1e-6):
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{name}: expected {expected:.6f}, got {actual:.6f}")


def aabb_overlap(a, b, clearance):
    amin, amax = a
    bmin, bmax = b
    return not (
        amax[0] + clearance <= bmin[0]
        or bmax[0] + clearance <= amin[0]
        or amax[1] + clearance <= bmin[1]
        or bmax[1] + clearance <= amin[1]
    )


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert_close(
        "robot pedestal top height",
        geom_top_z(model, data, "robot_base_pedestal_geom"),
        0.20,
    )
    assert_close(
        "material table top height",
        geom_top_z(model, data, "material_table_geom"),
        0.50,
    )

    aabbs = {name: body_aabb_xy(model, data, name) for name in MATERIAL_BODIES}
    overlaps = []
    for left, right in combinations(MATERIAL_BODIES, 2):
        if aabb_overlap(aabbs[left], aabbs[right], clearance=0.01):
            overlaps.append(f"{left} overlaps {right}")
    if overlaps:
        raise AssertionError("\n".join(overlaps))

    table_geom = model.geom("material_table_geom").id
    table_edge_y = data.geom_xpos[table_geom][1] - model.geom_size[table_geom][1]
    required_overhang = 0.03
    for site_name in GRASP_SITES:
        site_y = data.site_xpos[model.site(site_name).id][1]
        if site_y > table_edge_y - required_overhang:
            raise AssertionError(
                f"{site_name}: grasp y {site_y:.3f} does not overhang robot-side table edge {table_edge_y:.3f} by {required_overhang:.3f}"
            )

    print("Material layout OK")


if __name__ == "__main__":
    main()
