#!/usr/bin/env python3
"""Fast regression checks for the synchronized tie press and two-leaf stack."""
from __future__ import annotations

import mujoco
import numpy as np

from mujoco_xarm6.production_demo.clock import AnimationClock
from mujoco_xarm6.production_demo.constants import SCENE_PATH
from mujoco_xarm6.production_demo.line import JARS, TIE_X, ProductionLine
from mujoco_xarm6.production_demo.tie_press import TiePressController


def capsule_endpoints(model: mujoco.MjModel, geom_name: str) -> tuple[np.ndarray, np.ndarray]:
    geom = model.geom(geom_name)
    rotation = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, geom.quat)
    axis = rotation.reshape(3, 3)[:, 2]
    half_length = float(geom.size[1])
    return geom.pos - axis * half_length, geom.pos + axis * half_length


def assert_spring_is_connected(model: mujoco.MjModel, press: TiePressController):
    spring_start, _ = capsule_endpoints(model, "right_tie_gun_press_spring_0")
    _, spring_end = capsule_endpoints(model, "right_tie_gun_press_spring_3")
    start_distance = float(np.linalg.norm(spring_start - np.array([-0.010, 0.0, press.body_bottom_z])))
    ball_top = model.geom_pos[press.ball_id].copy()
    ball_top[2] -= press.ball_radius
    end_distance = float(np.linalg.norm(spring_end - ball_top))
    if start_distance > 1e-7 or end_distance > 1e-7:
        raise AssertionError(f"Spring must meet body and ball exactly: start={start_distance}, end={end_distance}")


def leaf_penetrations(model: mujoco.MjModel, data: mujoco.MjData, first_leaf: str, second_leaf: str) -> list[str]:
    first_ids = {model.geom(f"{first_leaf}_seg_{index:02d}_geom").id for index in range(11)}
    second_ids = {model.geom(f"{second_leaf}_seg_{index:02d}_geom").id for index in range(11)}
    pairs = []
    for contact in data.contact[: data.ncon]:
        if {contact.geom1, contact.geom2} & first_ids and {contact.geom1, contact.geom2} & second_ids and contact.dist < -0.0005:
            pairs.append(f"{contact.geom1}:{contact.geom2}:{contact.dist:.6f}")
    return pairs


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    press = TiePressController(model)
    assert_spring_is_connected(model, press)
    press.compress(0.0)
    assert_spring_is_connected(model, press)
    if abs(press.compression_m - press.MAX_COMPRESSION_M) > 1e-9:
        raise AssertionError("Press ball did not reach its specified compression")
    if model.geom_pos[press.ball_id][2] - press.ball_radius < press.body_bottom_z:
        raise AssertionError("Compressed press ball must not enter the tie-gun body")
    press.release(0.0)
    assert_spring_is_connected(model, press)

    clock = AnimationClock(model, data, viewer=None, realtime=False, speed_scale=1.0)
    line = ProductionLine(model, data, clock)
    line.reset()
    jar = JARS[0]
    model.body_pos[model.body(jar.body).id] = np.array([TIE_X, 0.05, 0.125])
    mujoco.mj_forward(model, data)
    for leaf, weld in ((jar.top_leaf, jar.top_mouth_weld), (jar.bottom_leaf, jar.bottom_mouth_weld)):
        line.leaves.attach_root(leaf, jar.body, weld)
    line._press_leaves(jar, 0.0)
    line.leaves.sync_roots()
    mujoco.mj_forward(model, data)
    pressed_gap_mm = float((line.leaves.leaf_center(jar.bottom_leaf)[2] - line.leaves.leaf_center(jar.top_leaf)[2]) * 1000.0)
    if not 4.5 <= pressed_gap_mm <= 5.5:
        raise AssertionError(f"Second leaf must remain above first while pressed, got {pressed_gap_mm:.3f} mm")
    line._gather_leaves(jar)
    while line.leaves.profile_transitions:
        line.leaves.advance_transitions(model.opt.timestep)
    line.leaves.sync_roots()
    mujoco.mj_forward(model, data)
    center_geom = model.geom(f"{jar.top_leaf}_seg_05_geom")
    leaf_width_mm = float(center_geom.size[1] * 2.0 * 1000.0)
    if abs(leaf_width_mm - 120.0) > 0.01:
        raise AssertionError(f"Leaf center must match the 120 mm jar mouth, got {leaf_width_mm:.3f} mm")
    root = data.xpos[model.body(f"{jar.top_leaf}_seg_05").id]
    inner = data.xpos[model.body(f"{jar.top_leaf}_seg_06").id]
    outer = data.xpos[model.body(f"{jar.top_leaf}_seg_07").id]
    fold_radius_mm = float(np.linalg.norm(((inner + outer) * 0.5 - root)[:2]) * 1000.0)
    outer_drop_mm = float((root[2] - outer[2]) * 1000.0)
    if not 55.0 <= fold_radius_mm <= 60.0:
        raise AssertionError(f"Tie fold must sit at the jar-mouth radius, got {fold_radius_mm:.3f} mm")
    if not 6.0 <= outer_drop_mm <= 10.0:
        raise AssertionError(f"Leaf must turn down directly outside the mouth, got {outer_drop_mm:.3f} mm")
    penetrations = leaf_penetrations(model, data, jar.top_leaf, jar.bottom_leaf)
    if penetrations:
        raise AssertionError(f"The two leaves intersect during tie gathering: {penetrations}")
    print("Tie press and two-leaf stack checks OK")


if __name__ == "__main__":
    main()
