#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from mujoco_xarm6.production_demo.clock import AnimationClock
from mujoco_xarm6.production_demo.constants import (
    BOTTOM_LEAF_PLACE_CLEARANCE_M,
    CLOSED_JAW_PREFIX,
    CONVEYOR_TRANSFER_SECONDS,
    FINAL_TIE_PREFIX,
    JAR,
    LEAF_PLACE_CLEARANCE_M,
    LEAF_PLACE_WAYPOINTS,
    LEAF_SETTLE_STEPS,
    LEFT_HOME,
    LOADED_TIE_PREFIX,
    LABEL_PAPER,
    LABEL_PAPER_PLACE_CLEARANCE_M,
    LABEL_PAPER_SETTLE_STEPS,
    OPEN_JAW_PREFIXES,
    OUTPUT_ROOT,
    REPO_ROOT,
    RIGHT_SIDE_STANDBY,
    ROBOT_SPEED_SCALE,
    SCENE_PATH,
    TIE_GUN_EXTENSION_GEOM_NAMES,
    TIE_GUN_EXTENSION_GEOM_PREFIXES,
    TIE_GUN_EXTENSION_SITE_NAMES,
    TIE_GUN_HOLD_SECONDS,
)
from mujoco_xarm6.production_demo.motion import (
    linear_z_move,
    loaded_transfer_move,
    make_left_production_arm,
    make_right_tie_arm,
    record_visual_action,
    require_ok,
    require_ok_or_pose_close,
    retreat_to_clearance,
    tcp_pose_for_site,
    tcp_pose_for_visible_tie_ring,
    tcp_translation_with_yaw,
)
from mujoco_xarm6.production_demo.scene_ops import (
    body_id,
    body_world_pos,
    freejoint_qpos_addr,
    get_freejoint_pose,
    joint_qpos_addrs,
    key_id,
    quat_to_yaw,
    site_pos,
    yaw_to_quat,
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TIE_GUN_EXTENSION_BASELINES: dict[int, dict[str, dict[int, np.ndarray]]] = {}


def arm_speed(speed: float) -> float:
    return float(speed * ROBOT_SPEED_SCALE)


def belt_marker_ids(model) -> list[int]:
    ids: list[int] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name and name.startswith("belt_marker_"):
            ids.append(geom_id)
    return ids


BELT_MARKER_MIN_X = -0.88
BELT_MARKER_MAX_X = 0.88
BELT_MARKER_TRACK_LENGTH = BELT_MARKER_MAX_X - BELT_MARKER_MIN_X


def belt_marker_start_positions(model, marker_ids: list[int]) -> dict[int, np.ndarray]:
    return {geom_id: model.geom_pos[geom_id].copy() for geom_id in marker_ids}


def wrapped_belt_x(x_value: float) -> float:
    while x_value > BELT_MARKER_MAX_X:
        x_value -= BELT_MARKER_TRACK_LENGTH
    while x_value < BELT_MARKER_MIN_X:
        x_value += BELT_MARKER_TRACK_LENGTH
    return x_value


def set_belt_markers_from_displacement(
    clock: AnimationClock,
    marker_start_positions: dict[int, np.ndarray],
    displacement_m: float,
):
    for geom_id, start_pos in marker_start_positions.items():
        pos = start_pos.copy()
        pos[0] = wrapped_belt_x(float(start_pos[0] + displacement_m))
        clock.model.geom_pos[geom_id] = pos


def record_belt_speed_check(clock: AnimationClock, phase: str, jar_displacement_m: float, marker_displacement_m: float):
    checks = clock.extra_diagnostics.setdefault("conveyor_belt_speed_checks", [])
    ratio = marker_displacement_m / jar_displacement_m if abs(jar_displacement_m) > 1e-9 else 1.0
    checks.append(
        {
            "phase": phase,
            "jar_displacement_m": float(jar_displacement_m),
            "marker_displacement_m": float(marker_displacement_m),
            "marker_to_jar_ratio": float(ratio),
        }
    )


def final_tie_geom_ids(model) -> list[int]:
    ids: list[int] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name and name.startswith(FINAL_TIE_PREFIX):
            ids.append(geom_id)
    if not ids:
        raise KeyError(f"Missing final tie geoms with prefix: {FINAL_TIE_PREFIX}")
    return ids


def loaded_tie_geom_ids(model) -> list[int]:
    ids: list[int] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name and name.startswith(LOADED_TIE_PREFIX):
            ids.append(geom_id)
    if not ids:
        raise KeyError(f"Missing loaded tie geoms with prefix: {LOADED_TIE_PREFIX}")
    return ids


def tie_gun_extension_baseline(model) -> dict[str, dict[int, np.ndarray]]:
    model_key = id(model)
    if model_key in TIE_GUN_EXTENSION_BASELINES:
        return TIE_GUN_EXTENSION_BASELINES[model_key]

    geom_baselines: dict[int, np.ndarray] = {}
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if not name:
            continue
        if name in TIE_GUN_EXTENSION_GEOM_NAMES or name.startswith(TIE_GUN_EXTENSION_GEOM_PREFIXES):
            geom_baselines[geom_id] = model.geom_pos[geom_id].copy()

    site_baselines: dict[int, np.ndarray] = {}
    for site_name in TIE_GUN_EXTENSION_SITE_NAMES:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise KeyError(f"Missing tie-gun extension site: {site_name}")
        site_baselines[site_id] = model.site_pos[site_id].copy()

    baseline = {"geom": geom_baselines, "site": site_baselines}
    TIE_GUN_EXTENSION_BASELINES[model_key] = baseline
    return baseline


def set_tie_gun_ring_extension(clock: AnimationClock, extension_m: float):
    baseline = tie_gun_extension_baseline(clock.model)
    local_offset = np.array([0.0, 0.0, extension_m], dtype=np.float64)
    for geom_id, base_pos in baseline["geom"].items():
        clock.model.geom_pos[geom_id] = base_pos + local_offset
    for site_id, base_pos in baseline["site"].items():
        clock.model.site_pos[site_id] = base_pos + local_offset
    mujoco.mj_forward(clock.model, clock.data)


def animate_tie_gun_ring_extension(clock: AnimationClock, target_extension_m: float, seconds: float):
    baseline = tie_gun_extension_baseline(clock.model)
    first_site_id = next(iter(baseline["site"]))
    start_extension = clock.model.site_pos[first_site_id][2] - baseline["site"][first_site_id][2]

    def update(alpha: float):
        extension = start_extension + (target_extension_m - start_extension) * alpha
        set_tie_gun_ring_extension(clock, extension)

    clock.run_seconds(seconds, update)


def geom_ids_with_prefix(model, prefixes: str | tuple[str, ...]) -> list[int]:
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    ids: list[int] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name and any(name.startswith(prefix) for prefix in prefixes):
            ids.append(geom_id)
    if not ids:
        raise KeyError(f"Missing geoms with prefix: {prefixes}")
    return ids


def set_geom_alpha_by_prefix(model, prefixes: str | tuple[str, ...], alpha: float):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    for geom_id in geom_ids_with_prefix(model, prefixes):
        model.geom_rgba[geom_id, 3] = alpha


def prefix_alpha(model, prefixes: str | tuple[str, ...]) -> float:
    return min(float(model.geom_rgba[geom_id, 3]) for geom_id in geom_ids_with_prefix(model, prefixes))


def set_final_tie_alpha(model, alpha: float):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    for geom_id in final_tie_geom_ids(model):
        model.geom_rgba[geom_id, 3] = alpha


def set_loaded_tie_alpha(model, alpha: float):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    for geom_id in loaded_tie_geom_ids(model):
        model.geom_rgba[geom_id, 3] = alpha


def geom_base_positions(model, prefix: str) -> dict[int, np.ndarray]:
    return {geom_id: model.geom_pos[geom_id].copy() for geom_id in geom_ids_with_prefix(model, prefix)}


def set_geom_xy_scale(model, base_positions: dict[int, np.ndarray], scale: float):
    for geom_id, base_pos in base_positions.items():
        pos = base_pos.copy()
        pos[:2] = base_pos[:2] * scale
        model.geom_pos[geom_id] = pos


def final_tie_alpha(model) -> float:
    return min(float(model.geom_rgba[geom_id, 3]) for geom_id in final_tie_geom_ids(model))


def loaded_tie_alpha(model) -> float:
    return min(float(model.geom_rgba[geom_id, 3]) for geom_id in loaded_tie_geom_ids(model))


def set_tie_gun_jaw_state(model, closed_alpha: float):
    closed_alpha = float(np.clip(closed_alpha, 0.0, 1.0))
    set_geom_alpha_by_prefix(model, CLOSED_JAW_PREFIX, closed_alpha)
    set_geom_alpha_by_prefix(model, OPEN_JAW_PREFIXES, 1.0 - 0.65 * closed_alpha)


def phase_conveyor_entry(clock: AnimationClock):
    print("\n=== Conveyor entry ===")
    marker_ids = belt_marker_ids(clock.model)
    marker_starts = belt_marker_start_positions(clock.model, marker_ids)
    jar_bid = body_id(clock.model, JAR)
    stop_pos = clock.model.body_pos[jar_bid].copy()
    entry_pos = stop_pos.copy()
    entry_pos[0] = -0.86
    clock.model.body_pos[jar_bid] = entry_pos
    mujoco.mj_forward(clock.model, clock.data)

    def update(alpha: float):
        pos = entry_pos + (stop_pos - entry_pos) * alpha
        clock.model.body_pos[jar_bid] = pos
        set_belt_markers_from_displacement(clock, marker_starts, float(pos[0] - entry_pos[0]))
        mujoco.mj_forward(clock.model, clock.data)

    clock.run_seconds(CONVEYOR_TRANSFER_SECONDS, update)
    record_belt_speed_check(clock, "conveyor_entry", float(stop_pos[0] - entry_pos[0]), float(stop_pos[0] - entry_pos[0]))


def phase_conveyor_exit(clock: AnimationClock, carry_payload: bool = False):
    print("\n=== Conveyor exit ===")
    marker_ids = belt_marker_ids(clock.model)
    marker_starts = belt_marker_start_positions(clock.model, marker_ids)
    jar_bid = body_id(clock.model, JAR)
    start_pos = clock.model.body_pos[jar_bid].copy()
    exit_pos = start_pos.copy()
    exit_pos[0] = 0.86
    payload_joints = [
        "staged_bamboo_leaf_top_freejoint",
        "staged_bamboo_leaf_bottom_freejoint",
        "staged_label_paper_freejoint",
    ]
    payload_starts = {}
    if carry_payload:
        for joint_name in payload_joints:
            pos, quat = get_freejoint_pose(clock.model, clock.data, joint_name)
            payload_starts[joint_name] = (pos, quat)
    delta = exit_pos - start_pos

    def update(alpha: float):
        jar_pos = start_pos + (exit_pos - start_pos) * alpha
        clock.model.body_pos[jar_bid] = jar_pos
        if carry_payload:
            for joint_name, (pos, quat) in payload_starts.items():
                adr = freejoint_qpos_addr(clock.model, joint_name)
                clock.data.qpos[adr : adr + 3] = pos + delta * alpha
                clock.data.qpos[adr + 3 : adr + 7] = quat
        set_belt_markers_from_displacement(clock, marker_starts, float(jar_pos[0] - start_pos[0]))
        clock.data.qvel[:] = 0
        mujoco.mj_forward(clock.model, clock.data)

    clock.run_seconds(CONVEYOR_TRANSFER_SECONDS, update)
    record_belt_speed_check(clock, "conveyor_exit", float(exit_pos[0] - start_pos[0]), float(exit_pos[0] - start_pos[0]))


def set_weld_to_current_relative_pose(model, data, equality_name: str, parent_body: str, child_body: str):
    eq_id = model.equality(equality_name).id
    parent_id = model.body(parent_body).id
    child_id = model.body(child_body).id
    parent_pos = data.xpos[parent_id].copy()
    child_pos = data.xpos[child_id].copy()
    parent_rot = data.xmat[parent_id].reshape(3, 3).copy()
    child_rot = data.xmat[child_id].reshape(3, 3).copy()
    rel_pos = parent_rot.T @ (child_pos - parent_pos)
    rel_rot = parent_rot.T @ child_rot
    rel_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(rel_quat, rel_rot.reshape(-1))
    model.eq_data[eq_id, 3:6] = rel_pos
    model.eq_data[eq_id, 6:10] = rel_quat
    model.eq_data[eq_id, 10] = 1.0
    data.eq_active[eq_id] = 1
    mujoco.mj_forward(model, data)


def align_leaf_root_yaw(model, data, leaf_name: str, desired_axis_yaw_deg: float):
    current_yaw = leaf_axis_yaw_deg(model, data, leaf_name)
    delta = math.radians(desired_axis_yaw_deg - current_yaw)
    adr = freejoint_qpos_addr(model, f"{leaf_name}_freejoint")
    current_root_yaw = quat_to_yaw(data.qpos[adr + 3 : adr + 7])
    data.qpos[adr + 3 : adr + 7] = yaw_to_quat(current_root_yaw + delta)
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def engage_mouth_static_friction(clock: AnimationClock, body_name: str, desired_axis_yaw_deg: float | None):
    del desired_axis_yaw_deg
    if body_name == "staged_bamboo_leaf_top":
        set_weld_to_current_relative_pose(
            clock.model,
            clock.data,
            "mouth_static_friction_weld_leaf_top",
            JAR,
            "staged_bamboo_leaf_top_seg_05",
        )
    elif body_name == "staged_bamboo_leaf_bottom":
        set_weld_to_current_relative_pose(
            clock.model,
            clock.data,
            "mouth_static_friction_weld_leaf_bottom",
            JAR,
            "staged_bamboo_leaf_bottom_seg_05",
        )
    elif body_name == LABEL_PAPER:
        set_weld_to_current_relative_pose(
            clock.model,
            clock.data,
            "mouth_static_friction_weld_label_paper",
            JAR,
            "staged_label_paper_seg_03",
        )
        animate_label_paper_side_droop(clock)


def engage_table_static_friction(clock: AnimationClock, body_name: str):
    del clock, body_name


LEAF_DROOP_BEND_PROFILE = np.array(
    [-0.0309, -0.3253, 0.0026, 0.3692, 0.0581, 0.0039, -0.0807, 0.3311, -0.1450, -0.0060],
    dtype=np.float64,
)

LABEL_SIDE_DROOP_BEND_PROFILE = {
    4: 0.12,
    5: 0.24,
    6: 0.38,
    7: 0.52,
    8: 0.64,
    9: 0.74,
    10: 0.82,
}


def set_label_paper_side_droop(clock: AnimationClock, scale: float):
    scale = float(np.clip(scale, 0.0, 1.0))
    anchor_before = leaf_center_pos(clock.model, clock.data, LABEL_PAPER)
    for index in range(1, 11):
        joint = clock.model.joint(f"staged_label_paper_bend_{index:02d}")
        clock.data.qpos[joint.qposadr[0]] = LABEL_SIDE_DROOP_BEND_PROFILE.get(index, 0.0) * scale
    clock.data.qvel[:] = 0
    mujoco.mj_forward(clock.model, clock.data)
    anchor_after = leaf_center_pos(clock.model, clock.data, LABEL_PAPER)
    freejoint_adr = freejoint_qpos_addr(clock.model, "staged_label_paper_freejoint")
    clock.data.qpos[freejoint_adr : freejoint_adr + 3] += anchor_before - anchor_after
    clock.data.qvel[:] = 0
    mujoco.mj_forward(clock.model, clock.data)


def animate_label_paper_side_droop(clock: AnimationClock, seconds: float = 0.45):
    steps = max(8, int(round(seconds / clock.model.opt.timestep / max(clock.speed_scale, 1e-6))))
    for index in range(steps):
        alpha = (index + 1) / steps
        smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        set_label_paper_side_droop(clock, smooth_alpha)
        clock.step(1)
    clock.extra_diagnostics["label_paper_droop_animation"] = {
        "seconds": float(seconds),
        "steps": int(steps),
        "target_long_end_droop_mm": float(LABEL_SIDE_DROOP_BEND_PROFILE[10] * 1000.0),
    }


def shape_released_leaf_droop(clock: AnimationClock, leaf_name: str):
    center_before = leaf_center_pos(clock.model, clock.data, leaf_name)
    for index, bend_value in enumerate(LEAF_DROOP_BEND_PROFILE, start=1):
        joint = clock.model.joint(f"{leaf_name}_bend_{index:02d}")
        clock.data.qpos[joint.qposadr[0]] = bend_value
    clock.data.qvel[:] = 0
    mujoco.mj_forward(clock.model, clock.data)
    center_after = leaf_center_pos(clock.model, clock.data, leaf_name)
    freejoint_adr = freejoint_qpos_addr(clock.model, f"{leaf_name}_freejoint")
    clock.data.qpos[freejoint_adr : freejoint_adr + 3] += center_before - center_after
    clock.data.qvel[:] = 0
    mujoco.mj_forward(clock.model, clock.data)
    set_weld_to_current_relative_pose(
        clock.model,
        clock.data,
        f"mouth_static_friction_weld_leaf_{'top' if leaf_name.endswith('top') else 'bottom'}",
        JAR,
        f"{leaf_name}_seg_05",
    )


def mouth_material_collection_state(model, data) -> dict[str, float]:
    mujoco.mj_forward(model, data)
    return {
        "top_leaf_center_z_m": float(leaf_center_pos(model, data, "staged_bamboo_leaf_top")[2]),
        "bottom_leaf_center_z_m": float(leaf_center_pos(model, data, "staged_bamboo_leaf_bottom")[2]),
        "label_paper_z_m": float(leaf_center_pos(model, data, LABEL_PAPER)[2]),
        "lotus_leaf_z_m": float(body_world_pos(model, data, "preloaded_lotus_leaf")[2]),
        "white_paper_z_m": float(body_world_pos(model, data, "preloaded_white_paper")[2]),
    }


def mouth_stack_order_state(model, data) -> dict[str, float]:
    mujoco.mj_forward(model, data)
    label_half_thickness_m = 0.0015
    ceramic_half_thickness_m = float(model.geom("preloaded_ceramic_disc_geom").size[1])
    label_center_z = float(leaf_center_pos(model, data, LABEL_PAPER)[2])
    ceramic_center_z = float(body_world_pos(model, data, "preloaded_ceramic_disc")[2])
    return {
        "label_center_z_m": label_center_z,
        "ceramic_center_z_m": ceramic_center_z,
        "label_bottom_above_ceramic_top_mm": float(
            (label_center_z - label_half_thickness_m - (ceramic_center_z + ceramic_half_thickness_m)) * 1000.0
        ),
    }


def collection_effect_mm(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        "top_leaf_center_lowered_mm": float((before["top_leaf_center_z_m"] - after["top_leaf_center_z_m"]) * 1000.0),
        "bottom_leaf_center_lowered_mm": float((before["bottom_leaf_center_z_m"] - after["bottom_leaf_center_z_m"]) * 1000.0),
        "label_paper_lowered_mm": float((before["label_paper_z_m"] - after["label_paper_z_m"]) * 1000.0),
        "lotus_leaf_lowered_mm": float((before["lotus_leaf_z_m"] - after["lotus_leaf_z_m"]) * 1000.0),
        "white_paper_lowered_mm": float((before["white_paper_z_m"] - after["white_paper_z_m"]) * 1000.0),
    }


def run_sdk_pick_place(
    clock: AnimationClock,
    arm: SimXArmAPI,
    actions: list[dict],
    label: str,
    expected_body: str,
    pick_site: str,
    place_point_m: np.ndarray,
    pick_yaw_deg: float,
    place_yaw_deg: float,
    settle_steps: int,
    release_equality_after_vacuum: str | None = None,
    release_mode: str = "mouth",
    pick_approach_clearance_m: float = 0.150,
    lift_after_pick_clearance_m: float = 0.190,
    place_clearance_m: float | None = None,
    skip_lift_after_pick: bool = False,
    transfer_speed: float | None = None,
    transfer_waypoint_count: int | None = None,
):
    print(f"\n--- SDK pick/place: {label} ---")
    pick_above = tcp_pose_for_site(clock.model, clock.data, pick_site, pick_approach_clearance_m, pick_yaw_deg)
    pick_contact = tcp_pose_for_site(clock.model, clock.data, pick_site, -0.006, pick_yaw_deg)
    lift_after_pick = tcp_pose_for_site(clock.model, clock.data, pick_site, lift_after_pick_clearance_m, pick_yaw_deg)
    if expected_body == "staged_bamboo_leaf_bottom":
        place_clearance = BOTTOM_LEAF_PLACE_CLEARANCE_M
    elif expected_body.startswith("staged_bamboo_leaf"):
        place_clearance = LEAF_PLACE_CLEARANCE_M
    else:
        place_clearance = LABEL_PAPER_PLACE_CLEARANCE_M
    if place_clearance_m is not None:
        place_clearance = place_clearance_m
    place_above = tcp_translation_with_yaw(place_point_m + np.array([0.0, 0.0, place_clearance]), place_yaw_deg)
    place_contact = tcp_translation_with_yaw(place_point_m, place_yaw_deg)

    require_ok("vacuum off before approach", arm.set_cgpio_digital(9, 0), arm, actions)
    safe_lift_before_pick_approach(arm, actions)
    require_ok(
        "move above pick",
        loaded_transfer_move(
            arm,
            pick_above,
            waypoint_count=3,
            speed=arm_speed(680.0),
            timeout_per_waypoint=2.5,
            joint_tolerance=0.06,
            refine_speed=arm_speed(420.0),
            refine_timeout=3.0,
            refine_joint_tolerance=0.018,
            position_tolerance_mm=12.0,
        ),
        arm,
        actions,
    )
    require_ok(
        "descend to pick",
        linear_z_move(
            arm,
            pick_contact["z"],
            yaw_deg=pick_yaw_deg,
            waypoint_count=4,
            speed=arm_speed(360.0),
            timeout_per_waypoint=2.0,
            joint_tolerance=0.018,
            refine_speed=arm_speed(260.0),
            refine_timeout=3.0,
            refine_joint_tolerance=0.014,
            position_tolerance_mm=5.0 if expected_body == LABEL_PAPER else 4.0,
        ),
        arm,
        actions,
    )
    all_suction_targets = arm._suction_targets
    arm._suction_targets = [target for target in all_suction_targets if target[0] == expected_body]
    try:
        require_ok("vacuum on", arm.set_cgpio_digital(9, 1), arm, actions)
    finally:
        arm._suction_targets = all_suction_targets
    if release_equality_after_vacuum is not None:
        clock.data.eq_active[clock.model.equality(release_equality_after_vacuum).id] = 0
        mujoco.mj_forward(clock.model, clock.data)
    if arm.attached_body_name != expected_body:
        raise RuntimeError(
            f"{label}: expected suction to attach {expected_body}, got {arm.attached_body_name}; "
            f"distance={arm.last_suction_distance_m}"
        )
    if not skip_lift_after_pick:
        require_ok(
            "lift after pick",
            linear_z_move(
                arm,
                lift_after_pick["z"],
                yaw_deg=pick_yaw_deg,
                waypoint_count=4,
                speed=arm_speed(620.0),
                timeout_per_waypoint=2.0,
                joint_tolerance=0.09,
            ),
            arm,
            actions,
        )
    if expected_body.startswith("staged_bamboo_leaf"):
        move_above_code = loaded_transfer_move(
            arm,
            place_above,
            waypoint_count=LEAF_PLACE_WAYPOINTS,
            speed=arm_speed(700.0),
            timeout_per_waypoint=3.5,
            joint_tolerance=0.055,
            refine_speed=arm_speed(320.0),
            refine_timeout=3.0,
            refine_joint_tolerance=0.014,
            position_tolerance_mm=8.0,
        )
    else:
        move_above_code = loaded_transfer_move(
            arm,
            place_above,
            speed=transfer_speed if transfer_speed is not None else arm_speed(520.0),
            waypoint_count=transfer_waypoint_count if transfer_waypoint_count is not None else 4,
            timeout_per_waypoint=2.0,
            joint_tolerance=0.018,
            refine_speed=transfer_speed if transfer_speed is not None else arm_speed(520.0),
            refine_timeout=8,
            refine_joint_tolerance=0.018,
            position_tolerance_mm=12.0,
        )
    require_ok("move above place", move_above_code, arm, actions)
    descend_code = linear_z_move(
        arm,
        place_contact["z"],
        yaw_deg=place_yaw_deg,
        waypoint_count=4,
        speed=arm_speed(520.0),
        timeout_per_waypoint=2.0,
        joint_tolerance=0.06,
        refine_speed=arm_speed(260.0) if expected_body.startswith("staged_bamboo_leaf") else None,
        refine_timeout=3.0,
        refine_joint_tolerance=0.014,
        position_tolerance_mm=8.0 if expected_body.startswith("staged_bamboo_leaf") else None,
    )
    require_ok(
        "descend to place",
        descend_code,
        arm,
        actions,
    )
    require_ok("vacuum off release", arm.set_cgpio_digital(9, 0), arm, actions)
    clock.data.qvel[:] = 0
    mujoco.mj_forward(clock.model, clock.data)
    if release_mode == "mouth":
        engage_mouth_static_friction(clock, expected_body, place_yaw_deg if expected_body.startswith("staged_bamboo_leaf") else None)
    elif release_mode == "table":
        engage_table_static_friction(clock, expected_body)
    elif release_mode != "none":
        raise ValueError(f"Unknown release mode: {release_mode}")
    clock.step(settle_steps)
    if expected_body.startswith("staged_bamboo_leaf"):
        shape_released_leaf_droop(clock, expected_body)
    retreat_code = retreat_to_clearance(arm, place_above)
    require_ok(
        "retreat after release",
        retreat_code,
        arm,
        actions,
    )


def safe_lift_before_pick_approach(arm: SimXArmAPI, actions: list[dict], safe_z_mm: float = 660.0):
    code, pose = arm.get_position(is_radian=False)
    if code != 0:
        require_ok("safe lift before pick approach", code, arm, actions)
        return
    if float(pose[2]) >= 645.0:
        record_visual_action(
            "safe lift before pick approach",
            arm,
            actions,
            {
                "status": "already_at_safe_clearance",
                "motion_primitive": "linear_z",
                "start_tcp_pose_mm_deg": pose,
                "target_tcp_pose_mm_deg": {
                    "x": float(pose[0]),
                    "y": float(pose[1]),
                    "z": float(pose[2]),
                    "roll": float(pose[3]),
                    "pitch": float(pose[4]),
                    "yaw": float(pose[5]),
                },
                "planned_xy_drift_mm": 0.0,
                "planned_delta_z_mm": 0.0,
            },
        )
        return
    target_z = max(float(safe_z_mm), float(pose[2]) + 45.0)
    require_ok(
        "safe lift before pick approach",
        linear_z_move(
            arm,
            target_z,
            yaw_deg=float(pose[5]),
            waypoint_count=4,
            speed=arm_speed(620.0),
            timeout_per_waypoint=2.0,
            joint_tolerance=0.055,
            position_tolerance_mm=18.0,
        ),
        arm,
        actions,
    )


def safe_lift_before_left_home(arm: SimXArmAPI, actions: list[dict], min_z_mm: float = 680.0):
    code, pose = arm.get_position(is_radian=False)
    if code != 0:
        require_ok("safe lift before left home", code, arm, actions)
        return
    if float(pose[2]) >= 645.0:
        record_visual_action(
            "safe lift before left home",
            arm,
            actions,
            {
                "status": "already_at_safe_clearance",
                "motion_primitive": "linear_z",
                "start_tcp_pose_mm_deg": pose,
                "target_tcp_pose_mm_deg": {
                    "x": float(pose[0]),
                    "y": float(pose[1]),
                    "z": float(pose[2]),
                    "roll": float(pose[3]),
                    "pitch": float(pose[4]),
                    "yaw": float(pose[5]),
                },
                "planned_xy_drift_mm": 0.0,
                "planned_delta_z_mm": 0.0,
            },
        )
        return
    target_z = max(660.0, min(float(min_z_mm), float(pose[2]) + 80.0))
    require_ok(
        "safe lift before left home",
        linear_z_move(
            arm,
            target_z,
            yaw_deg=float(pose[5]),
            waypoint_count=4,
            speed=arm_speed(620.0),
            timeout_per_waypoint=2.0,
            joint_tolerance=0.055,
            position_tolerance_mm=18.0,
        ),
        arm,
        actions,
    )


def phase_left_arm_loading(clock: AnimationClock) -> list[dict]:
    print("\n=== Left arm loading ===")
    arm = make_left_production_arm(clock)
    actions: list[dict] = []
    jar_mouth = site_pos(clock.model, clock.data, "jar_mouth_center")
    set_weld_to_current_relative_pose(
        clock.model,
        clock.data,
        "table_static_friction_weld_leaf_bottom",
        "left_material_table",
        "staged_bamboo_leaf_bottom_seg_05",
    )
    run_sdk_pick_place(
        clock,
        arm,
        actions,
        label="top bamboo leaf",
        expected_body="staged_bamboo_leaf_top",
        pick_site="staged_bamboo_leaf_top_pick_site",
        place_point_m=jar_mouth + np.array([0.0, 0.0, 0.030]),
        pick_yaw_deg=0.0,
        place_yaw_deg=0.0,
        settle_steps=LEAF_SETTLE_STEPS,
    )
    safe_lift_before_left_home(arm, actions)
    require_ok("return left arm home after top leaf", arm.set_servo_angle(LEFT_HOME, is_radian=True, speed=arm_speed(4.0), timeout=5), arm, actions)
    run_sdk_pick_place(
        clock,
        arm,
        actions,
        label="bottom bamboo leaf",
        expected_body="staged_bamboo_leaf_bottom",
        pick_site="staged_bamboo_leaf_bottom_pick_site",
        place_point_m=jar_mouth + np.array([0.0, 0.0, 0.041]),
        pick_yaw_deg=0.0,
        place_yaw_deg=90.0,
        settle_steps=LEAF_SETTLE_STEPS,
        release_equality_after_vacuum="table_static_friction_weld_leaf_bottom",
    )
    safe_lift_before_left_home(arm, actions)
    require_ok("return left arm home after bottom leaf", arm.set_servo_angle(LEFT_HOME, is_radian=True, speed=arm_speed(4.0), timeout=5), arm, actions)
    run_sdk_pick_place(
        clock,
        arm,
        actions,
        label="label paper",
        expected_body=LABEL_PAPER,
        pick_site="label_paper_pick_site",
        place_point_m=jar_mouth + np.array([0.0, 0.0, 0.0408]),
        pick_yaw_deg=0.0,
        place_yaw_deg=45.0,
        settle_steps=LABEL_PAPER_SETTLE_STEPS,
        lift_after_pick_clearance_m=0.100,
        place_clearance_m=0.025,
    )
    safe_lift_before_left_home(arm, actions)
    require_ok("return left arm home", arm.set_servo_angle(LEFT_HOME, is_radian=True, speed=arm_speed(4.0), timeout=5), arm, actions)
    return actions


def phase_tie_gun(clock: AnimationClock, actions: list[dict]):
    print("\n=== Tie-gun tying ===")
    clock.clear_joint_hold()
    arm = make_right_tie_arm(clock)
    neck = site_pos(clock.model, clock.data, "neck_tie_target_site")
    set_tie_gun_ring_extension(clock, 0.0)
    above_neck = tcp_translation_with_yaw(neck + np.array([0.0, -0.020, 0.120]), 90.0)
    neck_ring_point = neck + np.array([0.0, 0.0, 0.030])

    require_ok(
        "prepare right arm side standby",
        arm.set_servo_angle(RIGHT_SIDE_STANDBY, is_radian=True, speed=arm_speed(4.0), timeout=5),
        arm,
        actions,
    )
    move_above_result = arm.set_position(
        **above_neck,
        speed=arm_speed(560.0),
        timeout=7,
        joint_tolerance=0.065,
    )
    arm.last_motion_diagnostics = {
        **arm.last_motion_diagnostics,
        "waypoint_count": 1,
    }
    require_ok_or_pose_close(
        "move tie ring above jar neck",
        move_above_result,
        arm,
        actions,
        above_neck,
        24.0,
    )
    neck_pose = tcp_pose_for_visible_tie_ring(clock.model, clock.data, neck_ring_point)
    collection_before = mouth_material_collection_state(clock.model, clock.data)
    neck_result = arm.set_position(**neck_pose, speed=arm_speed(460.0), timeout=5, joint_tolerance=0.075)
    require_ok_or_pose_close(
        "move tie ring to bottle neck",
        neck_result,
        arm,
        actions,
        neck_pose,
        36.0,
    )

    print("Tie-gun holding at bottle neck for demo")
    right_joint_addrs = joint_qpos_addrs(clock.model, [f"right_joint{i}" for i in range(1, 7)])
    clock.hold_joints(right_joint_addrs, clock.data.qpos[right_joint_addrs].copy())
    collection_after = mouth_material_collection_state(clock.model, clock.data)
    collection_effect = collection_effect_mm(collection_before, collection_after)
    clock.extra_diagnostics["tie_gun_collection_effect"] = collection_effect
    clock.extra_diagnostics["mouth_stack_after_tie"] = mouth_stack_order_state(clock.model, clock.data)
    hold_diagnostics = clock.demo_pause_seconds(TIE_GUN_HOLD_SECONDS, fps=30.0)
    record_visual_action(
        "hold tie gun at neck for demo",
        arm,
        actions,
        {
            "status": "demo_hold",
            **hold_diagnostics,
            "collection_effect": collection_effect,
            "final_tie_alpha": final_tie_alpha(clock.model),
            "loaded_tie_alpha": loaded_tie_alpha(clock.model),
        },
    )

    print("Tie-gun retracting after demo hold")
    clock.clear_joint_hold()
    require_ok(
        "raise tie gun after demo hold",
        linear_z_move(
            arm,
            above_neck["z"],
            yaw_deg=None,
            roll_deg=None,
            pitch_deg=None,
            waypoint_count=4,
            speed=arm_speed(460.0),
            timeout_per_waypoint=2.0,
            joint_tolerance=0.06,
            position_tolerance_mm=24.0,
        ),
        arm,
        actions,
    )
    side_high_ring_point = np.array([-0.120, -0.380, 0.640], dtype=np.float64)
    side_high_pose = tcp_pose_for_visible_tie_ring(clock.model, clock.data, side_high_ring_point)
    require_ok_or_pose_close(
        "move tie gun to side high standby",
        loaded_transfer_move(
            arm,
            side_high_pose,
            waypoint_count=3,
            speed=arm_speed(560.0),
            timeout_per_waypoint=2.5,
            joint_tolerance=0.065,
            refine_speed=arm_speed(360.0),
            refine_timeout=3.0,
            refine_joint_tolerance=0.024,
            position_tolerance_mm=18.0,
        ),
        arm,
        actions,
        side_high_pose,
        24.0,
    )
    require_ok(
        "return tie gun to side standby",
        arm.set_servo_angle(RIGHT_SIDE_STANDBY, is_radian=True, speed=arm_speed(4.0), timeout=5),
        arm,
        actions,
    )


def angle_difference_deg(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def leaf_end_droop_mm(model, data, leaf_name: str) -> list[float]:
    center_z = float(data.site_xpos[model.site(f"{leaf_name}_center_site").id][2])
    return [
        (center_z - float(data.xpos[model.body(f"{leaf_name}_seg_00").id][2])) * 1000.0,
        (center_z - float(data.xpos[model.body(f"{leaf_name}_seg_10").id][2])) * 1000.0,
    ]


def leaf_center_pos(model, data, leaf_name: str) -> np.ndarray:
    return data.site_xpos[model.site(f"{leaf_name}_center_site").id].copy()


def leaf_axis_yaw_deg(model, data, leaf_name: str) -> float:
    p0 = data.xpos[model.body(f"{leaf_name}_seg_00").id].copy()
    p1 = data.xpos[model.body(f"{leaf_name}_seg_10").id].copy()
    delta = p1 - p0
    return float(math.degrees(math.atan2(delta[1], delta[0])))


def bamboo_leaf_geom_ids(model) -> list[int]:
    ids: list[int] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith("staged_bamboo_leaf_") and name.endswith("_geom"):
            ids.append(geom_id)
    if not ids:
        raise KeyError("Missing staged bamboo leaf geoms")
    return ids


def xy_point_inside_geom_projection(model, data, geom_id: int, point_xy: np.ndarray, margin_m: float = 0.002) -> bool:
    xpos_xy = data.geom_xpos[geom_id][:2]
    xmat = data.geom_xmat[geom_id].reshape(3, 3)
    basis_xy = xmat[:2, :2]
    if abs(np.linalg.det(basis_xy)) < 1e-6:
        return False
    local_xy = np.linalg.solve(basis_xy, point_xy - xpos_xy)
    size = model.geom_size[geom_id]
    return bool(abs(local_xy[0]) <= size[0] + margin_m and abs(local_xy[1]) <= size[1] + margin_m)


def ceramic_leaf_coverage(model, data) -> dict:
    mujoco.mj_forward(model, data)
    ceramic_center = site_pos(model, data, "ceramic_disc_center_site")[:2]
    ceramic_geom = model.geom("preloaded_ceramic_disc_geom")
    radius = float(model.geom_size[ceramic_geom.id][0])
    leaf_geoms = bamboo_leaf_geom_ids(model)
    total = 0
    covered = 0
    worst_uncovered_radius = 0.0
    for ix in range(-8, 9):
        for iy in range(-8, 9):
            point_xy = ceramic_center + radius * np.array([ix / 8.0, iy / 8.0], dtype=np.float64)
            dist = float(np.linalg.norm(point_xy - ceramic_center))
            if dist > radius:
                continue
            total += 1
            if any(xy_point_inside_geom_projection(model, data, geom_id, point_xy) for geom_id in leaf_geoms):
                covered += 1
            else:
                worst_uncovered_radius = max(worst_uncovered_radius, dist)
    return {
        "covered_ratio": covered / total if total else 0.0,
        "covered_points": covered,
        "sample_points": total,
        "ceramic_radius_m": radius,
        "worst_uncovered_radius_m": worst_uncovered_radius,
    }


def industrial_route_checks(model, data, sdk_actions: list[dict]) -> dict:
    jar_mouth_z_mm = float(site_pos(model, data, "jar_mouth_center")[2] * 1000.0)
    jar_xy_m = body_world_pos(model, data, JAR)[:2]
    label_xy_m = leaf_center_pos(model, data, LABEL_PAPER)[:2]
    loaded_transfers = [
        action
        for action in sdk_actions
        if action["label"] == "move above place" and action.get("attached_body") is not None
    ]
    loaded_transfers_to_jar = [
        action
        for action in loaded_transfers
        if float(action["tcp_pose_mm_deg"][0]) > -100.0
    ]
    transfer_z_values = [float(action["tcp_pose_mm_deg"][2]) for action in loaded_transfers_to_jar]

    leaf_release_z_values: list[float] = []
    leaf_retreat_z_values: list[float] = []
    vertical_lift_xy_drifts: list[float] = []
    vertical_place_xy_drifts: list[float] = []
    vertical_retreat_xy_drifts: list[float] = []
    leaf_place_tilts_deg: list[float] = []
    top_leaf_release_z_mm: float | None = None
    bottom_leaf_above_z_mm: float | None = None
    bottom_leaf_release_z_mm: float | None = None

    def action_xy_distance_mm(first: dict, second: dict) -> float:
        first_xy = np.array(first["tcp_pose_mm_deg"][:2], dtype=np.float64)
        second_xy = np.array(second["tcp_pose_mm_deg"][:2], dtype=np.float64)
        return float(np.linalg.norm(second_xy - first_xy))

    def tcp_vertical_tilt_deg(action: dict) -> float:
        pose = action.get("tcp_pose_mm_deg") or [0.0, 0.0, 0.0, 180.0, 0.0, 0.0]
        roll_error = abs(abs(float(pose[3])) - 180.0)
        pitch_error = abs(float(pose[4]))
        return float(math.hypot(roll_error, pitch_error))

    for index, action in enumerate(sdk_actions):
        if action["label"] == "lift after pick":
            previous_attach = next(
                (
                    candidate
                    for candidate in reversed(sdk_actions[:index])
                    if candidate["label"] == "vacuum on"
                    and candidate.get("attached_body") == action.get("attached_body")
                ),
                None,
            )
            if previous_attach is not None:
                vertical_lift_xy_drifts.append(action_xy_distance_mm(previous_attach, action))
        if action["label"] == "descend to place":
            previous_above = next(
                (
                    candidate
                    for candidate in reversed(sdk_actions[:index])
                    if candidate["label"] == "move above place"
                    and candidate.get("attached_body") == action.get("attached_body")
                ),
                None,
            )
            if previous_above is not None:
                vertical_place_xy_drifts.append(action_xy_distance_mm(previous_above, action))
            if action.get("attached_body") in ("staged_bamboo_leaf_top", "staged_bamboo_leaf_bottom"):
                leaf_place_tilts_deg.append(tcp_vertical_tilt_deg(action))
        if action["label"] == "vacuum off release" and action.get("attached_body") is None:
            previous_attached = next(
                (
                    candidate
                    for candidate in reversed(sdk_actions[:index])
                    if candidate.get("attached_body") is not None
                ),
                None,
            )
            if previous_attached is not None and previous_attached.get("attached_body") == "staged_bamboo_leaf_top":
                top_leaf_release_z_mm = float(action["tcp_pose_mm_deg"][2])
                leaf_place_tilts_deg.append(tcp_vertical_tilt_deg(action))
            if previous_attached is not None and previous_attached.get("attached_body") == "staged_bamboo_leaf_bottom":
                bottom_leaf_release_z_mm = float(action["tcp_pose_mm_deg"][2])
                leaf_place_tilts_deg.append(tcp_vertical_tilt_deg(action))
        if action["label"] == "move above place" and action.get("attached_body") == "staged_bamboo_leaf_bottom":
            bottom_leaf_above_z_mm = float(action["tcp_pose_mm_deg"][2])
        if action["label"] == "move above place" and action.get("attached_body") in (
            "staged_bamboo_leaf_top",
            "staged_bamboo_leaf_bottom",
        ):
            leaf_place_tilts_deg.append(tcp_vertical_tilt_deg(action))
        if action["label"] != "retreat after release":
            continue
        previous_release = next(
            (
                candidate
                for candidate in reversed(sdk_actions[:index])
                if candidate["label"] == "vacuum off release"
            ),
            None,
        )
        if previous_release is None:
            continue
        previous_attached = next(
            (
                candidate
                for candidate in reversed(sdk_actions[:index])
                if candidate.get("attached_body") is not None
            ),
            None,
        )
        if previous_attached is not None and str(previous_attached.get("attached_body", "")).startswith("staged_bamboo_leaf"):
            leaf_release_z_values.append(float(previous_release["tcp_pose_mm_deg"][2]))
            leaf_retreat_z_values.append(float(action["tcp_pose_mm_deg"][2]))
            vertical_retreat_xy_drifts.append(action_xy_distance_mm(previous_release, action))

    side_standby = next(
        (action for action in sdk_actions if action["label"] == "prepare right arm side standby"),
        None,
    )
    tie_above = next((action for action in sdk_actions if action["label"] == "move tie ring above jar neck"), None)
    tie_neck = next((action for action in sdk_actions if action["label"] == "move tie ring to bottle neck"), None)
    tie_raise = next((action for action in sdk_actions if action["label"] == "raise tie gun after demo hold"), None)

    right_neck_xy_drift_mm = 0.0
    if tie_above and tie_neck and tie_above.get("ring_visual_pos_mm") and tie_neck.get("ring_visual_pos_mm"):
        above_xy = np.array(tie_above["ring_visual_pos_mm"][:2], dtype=np.float64)
        neck_xy = np.array(tie_neck["ring_visual_pos_mm"][:2], dtype=np.float64)
        right_neck_xy_drift_mm = float(np.linalg.norm(above_xy - neck_xy))

    right_raise_min_vertical_mm = 0.0
    if tie_raise and tie_neck and tie_raise.get("ring_visual_pos_mm") and tie_neck.get("ring_visual_pos_mm"):
        right_raise_min_vertical_mm = float(tie_raise["ring_visual_pos_mm"][2] - tie_neck["ring_visual_pos_mm"][2])

    return {
        "left_loaded_transfer_count": len(loaded_transfers),
        "left_loaded_transfer_min_z_mm": min(transfer_z_values) if transfer_z_values else 0.0,
        "left_loaded_transfer_clearance_mm": min(transfer_z_values) - jar_mouth_z_mm if transfer_z_values else 0.0,
        "left_retreat_min_vertical_mm": min(
            [retreat - release for release, retreat in zip(leaf_release_z_values, leaf_retreat_z_values)]
        )
        if leaf_retreat_z_values
        else 0.0,
        "left_vertical_lift_max_xy_drift_mm": max(vertical_lift_xy_drifts) if vertical_lift_xy_drifts else 0.0,
        "left_vertical_place_max_xy_drift_mm": max(vertical_place_xy_drifts) if vertical_place_xy_drifts else 0.0,
        "left_vertical_retreat_max_xy_drift_mm": max(vertical_retreat_xy_drifts) if vertical_retreat_xy_drifts else 0.0,
        "left_leaf_place_max_tilt_deg": max(leaf_place_tilts_deg) if leaf_place_tilts_deg else 0.0,
        "bottom_leaf_place_clearance_mm": (
            bottom_leaf_above_z_mm - bottom_leaf_release_z_mm
            if bottom_leaf_above_z_mm is not None and bottom_leaf_release_z_mm is not None
            else 0.0
        ),
        "label_paper_final_jar_offset_mm": float(np.linalg.norm(label_xy_m - jar_xy_m) * 1000.0),
        "bottom_leaf_above_top_release_clearance_mm": (
            bottom_leaf_above_z_mm - top_leaf_release_z_mm
            if bottom_leaf_above_z_mm is not None and top_leaf_release_z_mm is not None
            else 0.0
        ),
        "bottom_leaf_release_above_top_release_mm": (
            bottom_leaf_release_z_mm - top_leaf_release_z_mm
            if bottom_leaf_release_z_mm is not None and top_leaf_release_z_mm is not None
            else 0.0
        ),
        "right_standby_side_y_mm": float(side_standby["tcp_pose_mm_deg"][1]) if side_standby else 0.0,
        "right_approach_min_z_mm": float(tie_above["tcp_pose_mm_deg"][2]) if tie_above else 0.0,
        "right_neck_xy_drift_mm": right_neck_xy_drift_mm,
        "right_raise_min_vertical_mm": right_raise_min_vertical_mm,
    }


def final_diagnostics(model, data, phase: str, sdk_actions: list[dict], extra_diagnostics: dict | None = None) -> dict:
    mujoco.mj_forward(model, data)
    jar_pos = body_world_pos(model, data, JAR)
    top_pos = leaf_center_pos(model, data, "staged_bamboo_leaf_top")
    bottom_pos = leaf_center_pos(model, data, "staged_bamboo_leaf_bottom")
    label_pos = leaf_center_pos(model, data, LABEL_PAPER)
    top_yaw = leaf_axis_yaw_deg(model, data, "staged_bamboo_leaf_top")
    bottom_yaw = leaf_axis_yaw_deg(model, data, "staged_bamboo_leaf_bottom")
    label_yaw = leaf_axis_yaw_deg(model, data, LABEL_PAPER)
    diagnostics = {
        "phase": phase,
        "jar_pos": jar_pos.tolist(),
        "top_leaf_pos": top_pos.tolist(),
        "bottom_leaf_pos": bottom_pos.tolist(),
        "label_paper_pos": label_pos.tolist(),
        "top_leaf_yaw_deg": top_yaw,
        "bottom_leaf_yaw_deg": bottom_yaw,
        "label_paper_yaw_deg": label_yaw,
        "leaf_crossing_angle_deg": angle_difference_deg(top_yaw, bottom_yaw),
        "top_leaf_end_droop_mm": leaf_end_droop_mm(model, data, "staged_bamboo_leaf_top"),
        "bottom_leaf_end_droop_mm": leaf_end_droop_mm(model, data, "staged_bamboo_leaf_bottom"),
        "label_paper_end_droop_mm": leaf_end_droop_mm(model, data, LABEL_PAPER),
        "ceramic_leaf_coverage": ceramic_leaf_coverage(model, data),
        "right_tie_gun_center_pos": site_pos(model, data, "right_tie_gun_center_site").tolist(),
        "right_tie_gun_ring_visual_pos": site_pos(model, data, "right_tie_gun_ring_visual_site").tolist(),
        "final_tie_alpha": final_tie_alpha(model),
        "loaded_tie_alpha": loaded_tie_alpha(model),
        "closed_jaw_alpha": prefix_alpha(model, CLOSED_JAW_PREFIX),
        "demo_timing": {
            "conveyor_entry_seconds": CONVEYOR_TRANSFER_SECONDS,
            "conveyor_exit_seconds": CONVEYOR_TRANSFER_SECONDS,
            "leaf_settle_steps": LEAF_SETTLE_STEPS,
            "label_paper_settle_steps": LABEL_PAPER_SETTLE_STEPS,
            "leaf_place_waypoints": LEAF_PLACE_WAYPOINTS,
            "leaf_place_clearance_mm": LEAF_PLACE_CLEARANCE_M * 1000.0,
            "label_paper_place_clearance_mm": LABEL_PAPER_PLACE_CLEARANCE_M * 1000.0,
            "tie_gun_hold_seconds": TIE_GUN_HOLD_SECONDS,
        },
        "industrial_route_checks": industrial_route_checks(model, data, sdk_actions),
        "sdk_actions": sdk_actions,
    }
    if extra_diagnostics:
        diagnostics.update(extra_diagnostics)
    return diagnostics


def save_diagnostics(diagnostics: dict) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / "latest_result.json"
    output_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run the wine jar production demo animation in MuJoCo.")
    parser.add_argument("--scene", type=Path, default=SCENE_PATH)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--hold-open", action="store_true")
    parser.add_argument("--speed-scale", type=float, default=3.0)
    parser.add_argument("--phase", choices=["all", "conveyor", "left-arm", "tie-gun"], default="all")
    parser.add_argument(
        "--quiet-diagnostics",
        action="store_true",
        help="Save diagnostics JSON without printing the full payload to stdout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, key_id(model, "production_home"))
    set_final_tie_alpha(model, 0.0)
    set_loaded_tie_alpha(model, 1.0)
    set_tie_gun_jaw_state(model, 0.0)
    right_joint_addrs = joint_qpos_addrs(model, [f"right_joint{i}" for i in range(1, 7)])
    data.qpos[right_joint_addrs] = RIGHT_SIDE_STANDBY
    mujoco.mj_forward(model, data)
    viewer = mujoco.viewer.launch_passive(model, data) if args.viewer else None
    clock = AnimationClock(model, data, viewer, args.realtime or args.viewer, args.speed_scale)
    clock.hold_joints(right_joint_addrs, RIGHT_SIDE_STANDBY)
    sdk_actions: list[dict] = []

    try:
        print(f"Loaded scene: {args.scene}")
        print(f"Animation phase: {args.phase}")
        if args.viewer:
            print("Viewer mode: live MuJoCo window is open.")

        if args.phase == "all":
            phase_conveyor_entry(clock)
            sdk_actions.extend(phase_left_arm_loading(clock))
            phase_tie_gun(clock, sdk_actions)
            phase_conveyor_exit(clock, carry_payload=True)
        elif args.phase == "conveyor":
            phase_conveyor_entry(clock)
            phase_conveyor_exit(clock)
        elif args.phase == "left-arm":
            sdk_actions.extend(phase_left_arm_loading(clock))
        elif args.phase == "tie-gun":
            phase_tie_gun(clock, sdk_actions)

        diagnostics = final_diagnostics(model, data, args.phase, sdk_actions, clock.extra_diagnostics)
        output_path = save_diagnostics(diagnostics)
        if args.quiet_diagnostics:
            print(
                "\nFinal diagnostics summary: "
                f"phase={diagnostics['phase']}, "
                f"actions={len(diagnostics['sdk_actions'])}, "
                f"jar_x={diagnostics['jar_pos'][0]:.3f}, "
                f"label={diagnostics['label_paper_pos']}"
            )
        else:
            print("\nFinal diagnostics:")
            print(json.dumps(diagnostics, indent=2))
        print(f"Saved diagnostics: {output_path}")

        if viewer is not None and args.hold_open:
            print("Animation complete. Viewer will stay open until closed.")
            while viewer.is_running():
                viewer.sync()
        return 0
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
