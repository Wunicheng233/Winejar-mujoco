#!/usr/bin/env python3
"""Regression checks for rigid leaf-root attachment during handling and indexing."""
from __future__ import annotations

import mujoco
import numpy as np

from mujoco_xarm6.production_demo.clock import AnimationClock
from mujoco_xarm6.production_demo.constants import SCENE_PATH
from mujoco_xarm6.production_demo.line import GATHERED_LEAF_PROFILE, JARS, NATURAL_LEAF_PROFILE, ProductionLine
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


def end_drops_mm(model: mujoco.MjModel, data: mujoco.MjData, leaf: str) -> np.ndarray:
    root_z = data.xpos[model.body(f"{leaf}_seg_05").id][2]
    return np.asarray(
        [(data.xpos[model.body(f"{leaf}_seg_{segment:02d}").id][2] - root_z) * 1000.0 for segment in (0, 10)]
    )


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    clock = AnimationClock(model, data, viewer=None, realtime=False, speed_scale=1.0)
    line = ProductionLine(model, data, clock)
    jar = JARS[0]
    line.reset()

    line._attach(jar, jar.top_leaf)
    np.testing.assert_allclose(line.leaves.bend_values(jar.top_leaf), NATURAL_LEAF_PROFILE, atol=1e-9)
    natural_drops = end_drops_mm(model, data, jar.top_leaf)
    if not np.all((-15.0 < natural_drops) & (natural_drops < -7.0)):
        raise AssertionError(f"Natural profile should droop both ends slightly, got {natural_drops.round(2).tolist()} mm")
    data.qpos[line.left_addrs[0]] += 0.15
    mujoco.mj_forward(model, data)
    line.leaves.sync_roots()
    assert_root_follows_parent(model, data, line, jar.top_leaf)
    for _ in range(100):
        clock.step(1, after_step=line.leaves.sync_roots)
    np.testing.assert_allclose(line.leaves.bend_values(jar.top_leaf), NATURAL_LEAF_PROFILE, atol=1e-9)

    line._release(jar, jar.top_leaf)
    np.testing.assert_allclose(line.leaves.bend_values(jar.top_leaf), NATURAL_LEAF_PROFILE, atol=1e-9)
    jar_body = model.body(jar.body).id
    model.body_pos[jar_body][0] += 0.62
    mujoco.mj_forward(model, data)
    line.leaves.sync_roots()
    assert_root_follows_parent(model, data, line, jar.top_leaf)

    line.leaves.transition_profile(jar.top_leaf, GATHERED_LEAF_PROFILE, duration_s=0.25)
    while line.leaves.profile_transitions:
        line.leaves.advance_profiles(model.opt.timestep)
    np.testing.assert_allclose(line.leaves.bend_values(jar.top_leaf), GATHERED_LEAF_PROFILE, atol=1e-9)
    gathered_drops = end_drops_mm(model, data, jar.top_leaf)
    if not np.all(gathered_drops < natural_drops - 15.0):
        raise AssertionError(f"Gathered profile should lower both ends, got {gathered_drops.round(2).tolist()} mm")
    print("Leaf attachment controller checks OK")


if __name__ == "__main__":
    main()
