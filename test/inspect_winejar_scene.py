#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "scene" / "scene_winejar.xml"

SITES = [
    "jar_mouth_center",
    "jar_label_target",
    "lotus_leaf_center_site",
    "lotus_leaf_grasp_site",
    "white_paper_center_site",
    "white_paper_grasp_site",
    "ceramic_disc_center_site",
    "ceramic_disc_grasp_site",
    "bamboo_leaf_1_center_site",
    "bamboo_leaf_1_grasp_site",
    "bamboo_leaf_2_center_site",
    "bamboo_leaf_2_grasp_site",
    "label_paper_center_site",
    "label_paper_grasp_site",
]

BODIES = [
    "robot_base_pedestal",
    "wine_jar",
    "material_table",
    "lotus_leaf",
    "white_paper",
    "ceramic_disc",
    "bamboo_leaf_1",
    "bamboo_leaf_2",
    "label_paper",
]


def fmt(vec):
    return f"({vec[0]: .3f}, {vec[1]: .3f}, {vec[2]: .3f})"


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


if __name__ == "__main__":
    main()
