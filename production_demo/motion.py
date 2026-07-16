from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_xarm6.production_demo.clock import AnimationClock, smoothstep
from mujoco_xarm6.production_demo.constants import RIGHT_SIDE_STANDBY, ROBOT_SPEED_SCALE
from mujoco_xarm6.production_demo.scene_ops import optional_site_pos_mm, site_pos
from mujoco_xarm6.sim.xarm_sim_api import SimXArmAPI, rpy_to_mat


@dataclass(frozen=True)
class TcpTarget:
    point_m: np.ndarray
    yaw_deg: float | None = None
    roll_deg: float | None = None
    pitch_deg: float | None = None

    def to_sdk_pose(self) -> dict[str, float]:
        pose = {
            "x": float(self.point_m[0] * 1000.0),
            "y": float(self.point_m[1] * 1000.0),
            "z": float(self.point_m[2] * 1000.0),
        }
        if self.roll_deg is not None:
            pose["roll"] = float(self.roll_deg)
        if self.pitch_deg is not None:
            pose["pitch"] = float(self.pitch_deg)
        if self.yaw_deg is not None:
            pose["yaw"] = float(self.yaw_deg)
        return pose


def make_left_production_arm(clock: AnimationClock, hold_right_at_current: bool = False) -> SimXArmAPI:
    right_joint_names = [f"right_joint{i}" for i in range(1, 7)]
    right_hold_values = RIGHT_SIDE_STANDBY
    if hold_right_at_current:
        right_hold_addrs = np.array([clock.model.joint(name).qposadr[0] for name in right_joint_names], dtype=int)
        right_hold_values = clock.data.qpos[right_hold_addrs].copy()
    return SimXArmAPI(
        clock.model,
        clock.data,
        tcp_site="left_link_tcp",
        suction_site="left_vacuum_suction_site",
        joint_names=[f"left_joint{i}" for i in range(1, 7)],
        actuator_names=[f"left_vel{i}" for i in range(1, 7)],
        vacuum_body="left_vacuum_end_effector",
        suction_targets=[
            (
                "staged_bamboo_leaf_top",
                "staged_bamboo_leaf_top_pick_site",
                "left_suction_weld_leaf_top",
                "staged_bamboo_leaf_top_seg_05",
            ),
            (
                "staged_bamboo_leaf_bottom",
                "staged_bamboo_leaf_bottom_pick_site",
                "left_suction_weld_leaf_bottom",
                "staged_bamboo_leaf_bottom_seg_05",
            ),
            (
                "staged_label_paper",
                "label_paper_pick_site",
                "left_suction_weld_label_paper",
                "staged_label_paper_seg_03",
            ),
        ],
        suction_threshold_m=0.010,
        hold_joint_names=right_joint_names,
        hold_joint_values=right_hold_values,
        viewer=clock.viewer,
        realtime=clock.realtime,
    )


def make_right_tie_arm(clock: AnimationClock) -> SimXArmAPI:
    return SimXArmAPI(
        clock.model,
        clock.data,
        tcp_site="right_tie_gun_center_site",
        suction_site="right_tie_gun_center_site",
        joint_names=[f"right_joint{i}" for i in range(1, 7)],
        actuator_names=[f"right_vel{i}" for i in range(1, 7)],
        vacuum_body="right_tie_gun_tool",
        suction_targets=[],
        viewer=clock.viewer,
        realtime=clock.realtime,
    )


def tcp_pose_for_site(model, data, site_name: str, z_offset_m: float, yaw_deg: float | None) -> dict[str, float]:
    pos = site_pos(model, data, site_name).copy()
    pos[2] += z_offset_m
    if yaw_deg is not None:
        return TcpTarget(pos, yaw_deg=float(yaw_deg), roll_deg=180.0, pitch_deg=0.0).to_sdk_pose()
    return TcpTarget(pos).to_sdk_pose()


def tcp_pose_for_point(point_m: np.ndarray, yaw_deg: float | None) -> dict[str, float]:
    if yaw_deg is not None:
        return TcpTarget(point_m, yaw_deg=float(yaw_deg), roll_deg=180.0, pitch_deg=0.0).to_sdk_pose()
    return TcpTarget(point_m).to_sdk_pose()


def tcp_translation_with_yaw(point_m: np.ndarray, yaw_deg: float) -> dict[str, float]:
    return TcpTarget(point_m, yaw_deg=float(yaw_deg), roll_deg=180.0, pitch_deg=0.0).to_sdk_pose()


def tcp_pose_with_current_xy_and_z(
    arm: SimXArmAPI,
    z_mm: float,
    yaw_deg: float | None = None,
    roll_deg: float | None = None,
    pitch_deg: float | None = None,
) -> dict[str, float]:
    code, pose = arm.get_position(is_radian=False)
    if code != 0:
        raise RuntimeError(f"Failed to read current TCP pose: {code}")
    return {
        "x": float(pose[0]),
        "y": float(pose[1]),
        "z": float(z_mm),
        "roll": float(pose[3] if roll_deg is None else roll_deg),
        "pitch": float(pose[4] if pitch_deg is None else pitch_deg),
        "yaw": float(pose[5] if yaw_deg is None else yaw_deg),
    }


def set_position_precise(
    arm: SimXArmAPI,
    target_pose: dict[str, float],
    speed: float,
    timeout: float,
    joint_tolerance: float,
    position_tolerance_mm: float,
) -> int:
    code = arm.set_position(
        **target_pose,
        speed=speed,
        timeout=timeout,
        joint_tolerance=joint_tolerance,
    )
    pose_code, pose = arm.get_position(is_radian=False)
    if pose_code != 0:
        return pose_code
    target_xyz = np.array([target_pose["x"], target_pose["y"], target_pose["z"]], dtype=np.float64)
    actual_xyz = np.array(pose[:3], dtype=np.float64)
    position_error = float(np.linalg.norm(actual_xyz - target_xyz))
    arm.last_motion_diagnostics = {
        **arm.last_motion_diagnostics,
        "target_tcp_position_mm": target_xyz.tolist(),
        "actual_tcp_position_mm": actual_xyz.tolist(),
        "tcp_position_error_mm": position_error,
    }
    if code != 0:
        return code
    if position_error > position_tolerance_mm:
        arm.last_motion_diagnostics = {
            **arm.last_motion_diagnostics,
            "status": "tcp_position_error_too_large",
            "position_tolerance_mm": position_tolerance_mm,
        }
        return 100
    return 0


def refine_tcp_pose_after_blended_path(
    arm: SimXArmAPI,
    target_pose: dict[str, float],
    speed: float,
    timeout: float,
    joint_tolerance: float,
    position_tolerance_mm: float,
) -> int:
    blended_diagnostics = arm.last_motion_diagnostics.copy()
    code = set_position_precise(
        arm,
        target_pose,
        speed=speed,
        timeout=timeout,
        joint_tolerance=joint_tolerance,
        position_tolerance_mm=position_tolerance_mm,
    )
    refine_diagnostics = arm.last_motion_diagnostics.copy()
    arm.last_motion_diagnostics = {
        **blended_diagnostics,
        "final_refine_code": int(code),
        "final_refine_diagnostics": refine_diagnostics,
    }
    return code


def tcp_pose_for_visible_tie_ring(model, data, ring_point_m: np.ndarray) -> dict[str, float]:
    tcp_to_ring_offset = site_pos(model, data, "right_tie_gun_center_site") - site_pos(
        model,
        data,
        "right_tie_gun_ring_visual_site",
    )
    return tcp_pose_for_point(ring_point_m + tcp_to_ring_offset, None)


def require_ok(label: str, code: int, arm: SimXArmAPI, actions: list[dict]):
    pose_code, tcp_pose = arm.get_position(is_radian=False)
    actions.append(
        {
            "label": label,
            "code": int(code),
            "attached_body": arm.attached_body_name,
            "last_suction_distance_m": arm.last_suction_distance_m,
            "tcp_pose_code": int(pose_code),
            "tcp_pose_mm_deg": tcp_pose,
            "ring_visual_pos_mm": optional_site_pos_mm(arm.model, arm.data, "right_tie_gun_ring_visual_site"),
            "motion_diagnostics": arm.last_motion_diagnostics,
        }
    )
    print(f"{label}: {code}")
    if code != 0:
        raise RuntimeError(f"{label} failed with code {code}, diagnostics={arm.last_motion_diagnostics}")


def record_visual_action(label: str, arm: SimXArmAPI, actions: list[dict], diagnostics: dict | None = None):
    pose_code, tcp_pose = arm.get_position(is_radian=False)
    actions.append(
        {
            "label": label,
            "code": 0,
            "attached_body": arm.attached_body_name,
            "last_suction_distance_m": arm.last_suction_distance_m,
            "tcp_pose_code": int(pose_code),
            "tcp_pose_mm_deg": tcp_pose,
            "ring_visual_pos_mm": optional_site_pos_mm(arm.model, arm.data, "right_tie_gun_ring_visual_site"),
            "motion_diagnostics": diagnostics or {},
        }
    )
    print(f"{label}: visual")


def require_ok_or_pose_close(
    label: str,
    code: int,
    arm: SimXArmAPI,
    actions: list[dict],
    target_pose: dict[str, float],
    tolerance_mm: float,
):
    pose_code, pose = arm.get_position(is_radian=False)
    if code == 100 and pose_code == 0:
        error = math.sqrt(
            (pose[0] - target_pose["x"]) ** 2
            + (pose[1] - target_pose["y"]) ** 2
            + (pose[2] - target_pose["z"]) ** 2
        )
        if error <= tolerance_mm:
            arm.last_motion_diagnostics = {
                **arm.last_motion_diagnostics,
                "status": "tcp_pose_reached",
                "tcp_position_error_mm": error,
            }
            require_ok(label, 0, arm, actions)
            return
    require_ok(label, code, arm, actions)


def retreat_tcp_up(arm: SimXArmAPI, distance_mm: float, speed: float = 65.0, joint_tolerance: float = 0.04) -> int:
    code, pose = arm.get_position(is_radian=False)
    if code != 0:
        return code
    return arm.set_position(
        z=pose[2] + distance_mm,
        speed=speed,
        timeout=6,
        joint_tolerance=joint_tolerance,
    )


def _pose_xyz(pose: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    return np.array(pose[:3], dtype=np.float64)


def _target_xyz(target_pose: dict[str, float]) -> np.ndarray:
    return np.array([target_pose["x"], target_pose["y"], target_pose["z"]], dtype=np.float64)


def _annotate_motion_primitive(
    arm: SimXArmAPI,
    primitive: str,
    start_pose: list[float],
    target_pose: dict[str, float],
    extra: dict | None = None,
):
    pose_code, final_pose = arm.get_position(is_radian=False)
    start_xyz = _pose_xyz(start_pose)
    target_xyz = _target_xyz(target_pose)
    planned_xy_drift = float(np.linalg.norm(target_xyz[:2] - start_xyz[:2]))
    diagnostics = {
        **arm.last_motion_diagnostics,
        "motion_primitive": primitive,
        "start_tcp_pose_mm_deg": list(start_pose),
        "target_tcp_pose_mm_deg": dict(target_pose),
        "planned_xy_drift_mm": planned_xy_drift,
        "planned_delta_z_mm": float(target_xyz[2] - start_xyz[2]),
    }
    if pose_code == 0:
        final_xyz = _pose_xyz(final_pose)
        diagnostics.update(
            {
                "actual_tcp_pose_mm_deg": final_pose,
                "actual_xy_drift_mm": float(np.linalg.norm(final_xyz[:2] - start_xyz[:2])),
                "actual_delta_z_mm": float(final_xyz[2] - start_xyz[2]),
            }
        )
    if extra:
        diagnostics.update(extra)
    arm.last_motion_diagnostics = diagnostics


def linear_z_move(
    arm: SimXArmAPI,
    target_z_mm: float,
    yaw_deg: float | None = None,
    roll_deg: float | None = 180.0,
    pitch_deg: float | None = 0.0,
    waypoint_count: int = 4,
    speed: float = 520.0,
    timeout_per_waypoint: float = 2.0,
    joint_tolerance: float = 0.06,
    refine_speed: float | None = None,
    refine_timeout: float = 3.0,
    refine_joint_tolerance: float = 0.014,
    position_tolerance_mm: float | None = None,
) -> int:
    start_code, start_pose = arm.get_position(is_radian=False)
    if start_code != 0:
        return start_code
    target_pose = tcp_pose_with_current_xy_and_z(
        arm,
        target_z_mm,
        yaw_deg=yaw_deg,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
    )
    code = move_tcp_through_waypoints(
        arm,
        target_pose,
        waypoint_count=waypoint_count,
        speed=speed,
        timeout_per_waypoint=timeout_per_waypoint,
        joint_tolerance=joint_tolerance,
    )
    if code == 0 and position_tolerance_mm is not None:
        code = refine_tcp_pose_after_blended_path(
            arm,
            target_pose,
            speed=refine_speed if refine_speed is not None else speed,
            timeout=refine_timeout,
            joint_tolerance=refine_joint_tolerance,
            position_tolerance_mm=position_tolerance_mm,
        )
    _annotate_motion_primitive(arm, "linear_z", start_pose, target_pose)
    return code


def retreat_to_clearance(arm: SimXArmAPI, target_pose: dict[str, float]) -> int:
    start_code, start_pose = arm.get_position(is_radian=False)
    if start_code != 0:
        return start_code
    vertical_target_pose = tcp_pose_with_current_xy_and_z(
        arm,
        target_pose["z"],
        target_pose.get("yaw"),
        target_pose.get("roll"),
        target_pose.get("pitch"),
    )
    code = move_tcp_through_waypoints(
        arm,
        vertical_target_pose,
        waypoint_count=4,
        speed=620.0 * ROBOT_SPEED_SCALE,
        timeout_per_waypoint=2.0,
        joint_tolerance=0.024,
    )
    pose_code, pose = arm.get_position(is_radian=False)
    if code == 0 and pose_code == 0 and pose[2] - start_pose[2] >= 12.0:
        _annotate_motion_primitive(arm, "linear_z", start_pose, vertical_target_pose)
        return code
    if code == 0 and pose_code == 0:
        arm.last_motion_diagnostics = {
            **arm.last_motion_diagnostics,
            "status": "limited_vertical_retreat",
            "actual_vertical_retreat_mm": float(pose[2] - start_pose[2]),
        }
        _annotate_motion_primitive(arm, "linear_z", start_pose, vertical_target_pose)
        return 0
    pose_code, current_pose = arm.get_position(is_radian=False)
    if pose_code == 0:
        fallback_code = move_tcp_through_waypoints(
            arm,
            {
                "x": current_pose[0],
                "y": current_pose[1],
                "z": max(current_pose[2] + 24.0, start_pose[2] + 16.0),
                "roll": current_pose[3],
                "pitch": current_pose[4],
                "yaw": current_pose[5],
            },
            waypoint_count=4,
            speed=560.0 * ROBOT_SPEED_SCALE,
            timeout_per_waypoint=2.0,
            joint_tolerance=0.024,
        )
    else:
        fallback_code = pose_code
    if fallback_code != 0:
        fallback_code = retreat_tcp_up(arm, 18.0, speed=560.0 * ROBOT_SPEED_SCALE, joint_tolerance=0.06)
    arm.last_motion_diagnostics = {
        **arm.last_motion_diagnostics,
        "fallback_reason": "clearance_pose_unreachable",
        "first_attempt_code": int(code),
    }
    _annotate_motion_primitive(arm, "linear_z", start_pose, vertical_target_pose)
    return fallback_code


def interpolate_yaw_deg(start: float, target: float, alpha: float) -> float:
    delta = ((target - start + 180.0) % 360.0) - 180.0
    return start + delta * alpha


def move_tcp_through_waypoints(
    arm: SimXArmAPI,
    target_pose: dict[str, float],
    waypoint_count: int,
    speed: float,
    timeout_per_waypoint: float,
    joint_tolerance: float,
) -> int:
    code, current_pose = arm.get_position(is_radian=False)
    if code != 0:
        return code
    start_xyz = np.array(current_pose[:3], dtype=np.float64)
    target_xyz = np.array([target_pose["x"], target_pose["y"], target_pose["z"]], dtype=np.float64)
    start_roll = float(current_pose[3])
    start_pitch = float(current_pose[4])
    start_yaw = float(current_pose[5])
    target_roll = float(target_pose["roll"]) if "roll" in target_pose else start_roll
    target_pitch = float(target_pose["pitch"]) if "pitch" in target_pose else start_pitch
    target_yaw = float(target_pose["yaw"]) if "yaw" in target_pose else start_yaw

    q_start = arm.data.qpos.copy()
    qvel_start = arm.data.qvel.copy()
    joint_waypoints: list[np.ndarray] = []
    seed_q = arm.data.qpos[arm._qposadr].copy()
    for index in range(1, waypoint_count + 1):
        alpha = smoothstep(index / waypoint_count)
        xyz = start_xyz + (target_xyz - start_xyz) * alpha
        pose = [
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            interpolate_yaw_deg(start_roll, target_roll, alpha),
            start_pitch + (target_pitch - start_pitch) * alpha,
            interpolate_yaw_deg(start_yaw, target_yaw, alpha),
        ]
        arm.data.qpos[arm._qposadr] = seed_q
        arm.data.qvel[:] = 0
        mujoco.mj_forward(arm.model, arm.data)
        target_q = arm._solve_ik(np.array(pose[:3], dtype=np.float64) / 1000.0, rpy_to_mat(*pose[3:]))
        if target_q is None:
            relaxed_pose = pose.copy()
            relaxed_pose[3] = -180.0
            relaxed_pose[4] = 0.0
            target_q = arm._solve_ik(
                np.array(relaxed_pose[:3], dtype=np.float64) / 1000.0,
                rpy_to_mat(*relaxed_pose[3:]),
            )
        if target_q is None:
            arm.data.qpos[:] = q_start
            arm.data.qvel[:] = qvel_start
            mujoco.mj_forward(arm.model, arm.data)
            arm.last_motion_diagnostics = {
                "status": "ik_failed",
                "target_pose_mm_deg": pose,
                "waypoint_index": index,
                "waypoint_count": waypoint_count,
            }
            return -8
        joint_waypoints.append(target_q)
        seed_q = target_q

    arm.data.qpos[:] = q_start
    arm.data.qvel[:] = qvel_start
    mujoco.mj_forward(arm.model, arm.data)
    blended_code = arm.drive_joint_waypoints_blended(
        joint_waypoints,
        speed=float(np.clip(speed / 100.0, 0.15, max(7.0, 7.0 * ROBOT_SPEED_SCALE))),
        timeout_per_waypoint=timeout_per_waypoint,
        joint_tolerance=max(joint_tolerance, 0.075),
        blend_tolerance=max(joint_tolerance * 2.0, 0.075),
    )
    return blended_code


def loaded_transfer_move(
    arm: SimXArmAPI,
    target_pose: dict[str, float],
    waypoint_count: int,
    speed: float,
    timeout_per_waypoint: float,
    joint_tolerance: float,
    refine_speed: float | None = None,
    refine_timeout: float = 3.0,
    refine_joint_tolerance: float = 0.014,
    position_tolerance_mm: float | None = None,
) -> int:
    start_code, start_pose = arm.get_position(is_radian=False)
    if start_code != 0:
        return start_code
    code = move_tcp_through_waypoints(
        arm,
        target_pose,
        waypoint_count=waypoint_count,
        speed=speed,
        timeout_per_waypoint=timeout_per_waypoint,
        joint_tolerance=joint_tolerance,
    )
    if code == 0 and position_tolerance_mm is not None:
        code = refine_tcp_pose_after_blended_path(
            arm,
            target_pose,
            speed=refine_speed if refine_speed is not None else speed,
            timeout=refine_timeout,
            joint_tolerance=refine_joint_tolerance,
            position_tolerance_mm=position_tolerance_mm,
        )
    start_z = float(start_pose[2])
    target_z = float(target_pose["z"])
    _annotate_motion_primitive(
        arm,
        "loaded_transfer",
        start_pose,
        target_pose,
        {
            "planned_min_transfer_z_mm": min(start_z, target_z),
            "loaded_body": arm.attached_body_name,
        },
    )
    return code
