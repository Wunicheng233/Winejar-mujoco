from __future__ import annotations

import math

import mujoco
import numpy as np


def key_id(model, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if result < 0:
        raise KeyError(f"Missing keyframe: {name}")
    return result


def body_id(model, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if result < 0:
        raise KeyError(f"Missing body: {name}")
    return result


def site_pos(model, data, name: str) -> np.ndarray:
    return data.site_xpos[model.site(name).id].copy()


def body_world_pos(model, data, body_name: str) -> np.ndarray:
    return data.xpos[body_id(model, body_name)].copy()


def freejoint_qpos_addr(model, joint_name: str) -> int:
    return model.joint(joint_name).qposadr[0]


def joint_qpos_addrs(model, names: list[str]) -> np.ndarray:
    return np.array([model.joint(name).qposadr[0] for name in names], dtype=int)


def yaw_to_quat(yaw_rad: float) -> np.ndarray:
    return np.array([math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)], dtype=np.float64)


def quat_to_yaw(quat: np.ndarray) -> float:
    w, _x, _y, z = quat
    return 2.0 * math.atan2(z, w)


def set_freejoint_pose(model, data, joint_name: str, pos: np.ndarray, yaw_rad: float):
    adr = freejoint_qpos_addr(model, joint_name)
    data.qpos[adr : adr + 3] = pos
    data.qpos[adr + 3 : adr + 7] = yaw_to_quat(yaw_rad)


def get_freejoint_pose(model, data, joint_name: str) -> tuple[np.ndarray, np.ndarray]:
    adr = freejoint_qpos_addr(model, joint_name)
    return data.qpos[adr : adr + 3].copy(), data.qpos[adr + 3 : adr + 7].copy()


def optional_site_pos_mm(model, data, name: str) -> list[float] | None:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        return None
    pos = data.site_xpos[site_id] * 1000.0
    return [float(pos[0]), float(pos[1]), float(pos[2])]
