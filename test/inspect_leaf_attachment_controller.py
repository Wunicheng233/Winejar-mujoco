#!/usr/bin/env python3
"""Regression checks for rigid leaf-root attachment during handling and indexing."""
from __future__ import annotations

import mujoco
import numpy as np

from mujoco_xarm6.production_demo.clock import AnimationClock
from mujoco_xarm6.production_demo.constants import SCENE_PATH
from mujoco_xarm6.production_demo.line import JARS, ProductionLine
from mujoco_xarm6.production_demo.scene_ops import freejoint_qpos_addr


def root_position(model: mujoco.MjModel, data: mujoco.MjData, leaf: str) -> np.ndarray:
    address = freejoint_qpos_addr(model, f"{leaf}_freejoint")
    return data.qpos[address : address + 3].copy()


def expected_root_position(model: mujoco.MjModel, data: mujoco.MjData, attachment) -> np.ndarray:
    parent_id = model.body(attachment.parent_body).id
    return data.xpos[parent_id] + data.xmat[parent_id].reshape(3, 3) @ attachment.relative_pos


def assert_root_follows_parent(model: mujoco.MjModel, data: mujoco.MjData, line: ProductionLine, leaf: str):
    attachment = line.leaves.attachments[leaf]
    expected = expected_root_position(model, data, attachment)
    error_mm = float(np.linalg.norm(root_position(model, data, leaf) - expected) * 1000.0)
    if error_mm > 0.01:
        raise AssertionError(f"{leaf} root drifted {error_mm:.4f} mm from {attachment.parent_body}")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    clock = AnimationClock(model, data, viewer=None, realtime=False, speed_scale=1.0)
    line = ProductionLine(model, data, clock)
    jar = JARS[0]
    line.reset()

    line._attach(jar, jar.top_leaf)
    data.qpos[line.left_addrs[0]] += 0.15
    mujoco.mj_forward(model, data)
    line.leaves.sync_roots()
    assert_root_follows_parent(model, data, line, jar.top_leaf)

    line._release(jar, jar.top_leaf)
    while line.leaves.pending_placements:
        line._advance_leaf_placements(model.opt.timestep)
        line.leaves.sync_roots()
    jar_body = model.body(jar.body).id
    model.body_pos[jar_body][0] += 0.62
    mujoco.mj_forward(model, data)
    line.leaves.sync_roots()
    assert_root_follows_parent(model, data, line, jar.top_leaf)
    print("Leaf attachment controller checks OK")


if __name__ == "__main__":
    main()
