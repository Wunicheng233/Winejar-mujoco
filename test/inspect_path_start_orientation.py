#!/usr/bin/env python3
"""Ensure planned paths begin from the measured TCP orientation."""
from __future__ import annotations

import mujoco
import numpy as np

from mujoco_xarm6.production_demo.clock import AnimationClock
from mujoco_xarm6.production_demo.constants import SCENE_PATH
from mujoco_xarm6.production_demo.line import JARS, SAFE_Z_M, TIE_X, ProductionLine


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    line = ProductionLine(model, data, AnimationClock(model, data, viewer=None, realtime=False, speed_scale=1.0))
    line.reset()
    model.body_pos[model.body(JARS[0].body).id] = np.array([TIE_X, 0.05, 0.125])
    mujoco.mj_forward(model, data)

    code, start_pose = line.right.get_position(is_radian=False)
    if code != 0:
        raise AssertionError(f"Could not read right-arm start pose: {code}")
    start_yaw = float(start_pose[5])
    path = line._tie_job(JARS[0])
    current_tcp = data.site_xpos[line.right.tcp_site_id].copy()
    first_target = current_tcp.copy()
    first_target[2] = SAFE_Z_M
    first_endpoint = len(line.paths.cartesian_samples(current_tcp, first_target, start_yaw, 90.0))

    measured_yaws = []
    for point in path.points[1 : first_endpoint + 1]:
        data.qpos[line.right_addrs] = point
        mujoco.mj_forward(model, data)
        measured_yaws.append(float(line.right.get_position(is_radian=False)[1][5]))
    if not np.allclose(measured_yaws, 90.0, atol=0.1):
        raise AssertionError(f"Right arm performs a redundant startup rotation: {measured_yaws}")

    wrist_rotation_deg = abs(float(np.degrees(path.points[first_endpoint][5] - path.points[0][5])))
    if wrist_rotation_deg > 0.5:
        raise AssertionError(f"Right wrist twists during the initial safe lift: {wrist_rotation_deg:.3f} degrees")
    print("Path start orientation checks OK")


if __name__ == "__main__":
    main()
