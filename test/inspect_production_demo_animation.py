#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
RESULT = ROOT / "data" / "production_demo_animation" / "latest_result.json"
SCRIPT = ROOT / "scripts" / "02_production_demo_animation.py"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mujoco_xarm6.production_demo.constants import CONVEYOR_TRANSFER_SECONDS, SCENE_PATH  # noqa: E402


def assert_close(label: str, value: float, expected: float, tolerance: float):
    if abs(value - expected) > tolerance:
        raise AssertionError(f"{label}: expected {expected:.3f} +/- {tolerance:.3f}, got {value:.3f}")


def action_by_label(actions: list[dict], label: str) -> dict:
    matches = [action for action in actions if action["label"] == label]
    if not matches:
        raise AssertionError(f"Missing action: {label}")
    return matches[-1]


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect the wine-jar production demo animation.")
    parser.add_argument(
        "--scope",
        choices=["smoke", "quick", "full", "left-arm", "all"],
        default="full",
        help=(
            "smoke checks static scene/recording configuration only; "
            "quick runs one full animation with core regressions only; full keeps the historical checks; "
            "left-arm checks the loading phase only; all skips the separate left-arm precheck."
        ),
    )
    return parser.parse_args()


def run_animation(*args: str):
    subprocess.run([sys.executable, str(SCRIPT), "--quiet-diagnostics", *args], cwd=REPO, check=True)


def load_recording_module():
    script_path = ROOT / "scripts" / "03_record_production_demo_videos.py"
    spec = importlib.util.spec_from_file_location("production_demo_video_recorder", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load recorder script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_smoke_checks():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    required_cameras = [
        "global_overview_camera",
        "front_conveyor_camera",
        "side_overview_camera",
    ]
    missing = [
        camera
        for camera in required_cameras
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0
    ]
    if missing:
        raise AssertionError(f"Missing production recording cameras: {missing}")

    recorder = load_recording_module()
    if recorder.DEFAULT_CAMERAS != required_cameras:
        raise AssertionError(
            "Default recording cameras should match the approved three-camera plan, "
            f"got {recorder.DEFAULT_CAMERAS}"
        )
    assert_close("Conveyor transfer seconds", CONVEYOR_TRANSFER_SECONDS, 4.0, 1e-6)
    print("Production demo smoke checks OK")


def main():
    args = parse_args()
    if args.scope == "smoke":
        run_smoke_checks()
        return

    if args.scope in ("full", "left-arm"):
        run_animation("--phase", "left-arm")
        left_diagnostics = json.loads(RESULT.read_text(encoding="utf-8"))
        if left_diagnostics["bottom_leaf_pos"][2] < left_diagnostics["top_leaf_pos"][2] + 0.004:
            raise AssertionError(
                "Second bamboo leaf should still sit above the first after suction placement, "
                f"got top={left_diagnostics['top_leaf_pos']}, bottom={left_diagnostics['bottom_leaf_pos']}"
            )
        left_leaf_stack_gap_mm = (
            left_diagnostics["bottom_leaf_pos"][2] - left_diagnostics["top_leaf_pos"][2]
        ) * 1000.0
        left_label_bottom_to_leaf_gap_mm = (
            left_diagnostics["label_paper_pos"][2] - 0.0015 - left_diagnostics["bottom_leaf_pos"][2]
        ) * 1000.0
        if left_leaf_stack_gap_mm > 12.0:
            raise AssertionError(
                "Second bamboo leaf should not visibly float above the first, "
                f"gap={left_leaf_stack_gap_mm:.1f} mm, diagnostics={left_diagnostics}"
            )
        if left_label_bottom_to_leaf_gap_mm > 2.0:
            raise AssertionError(
                "Flexible label paper should visually sit on the upper bamboo leaf with almost no visible air gap, "
                f"gap={left_label_bottom_to_leaf_gap_mm:.1f} mm, diagnostics={left_diagnostics}"
            )
        if left_label_bottom_to_leaf_gap_mm < 0.3:
            raise AssertionError(
                "Flexible label paper should keep a tiny clearance above the bamboo leaf to avoid z-fighting/intersection, "
                f"gap={left_label_bottom_to_leaf_gap_mm:.1f} mm, diagnostics={left_diagnostics}"
            )
        assert_close("label paper yaw", left_diagnostics["label_paper_yaw_deg"], 45.0, 6.0)
        label_droop = left_diagnostics["label_paper_end_droop_mm"]
        if abs(label_droop[0]) > 12.0 or label_droop[1] < 30.0:
            raise AssertionError(
                "Long label paper should keep its short side on the jar mouth and droop only on the long side, "
                f"got droop={label_droop}, diagnostics={left_diagnostics}"
            )
        if args.scope == "left-arm":
            print("Production demo left-arm checks OK")
            return

    run_animation()
    diagnostics = json.loads(RESULT.read_text(encoding="utf-8"))
    crossing = diagnostics["leaf_crossing_angle_deg"]
    assert_close("leaf crossing angle", crossing, 90.0, 4.0)
    bad_actions = [action for action in diagnostics["sdk_actions"] if action["code"] != 0]
    if bad_actions:
        raise AssertionError(f"SDK-style robot actions failed: {bad_actions}")
    labels = [action["label"] for action in diagnostics["sdk_actions"]]
    doubled_speed_minima = {
        "move above pick": 12.9,
        "descend to pick": 6.8,
        "lift after pick": 11.7,
        "move above place": 9.8,
        "descend to place": 9.8,
        "return left arm home": 7.6,
        "return left arm home after top leaf": 7.6,
        "return left arm home after bottom leaf": 7.6,
        "prepare right arm side standby": 7.6,
        "move tie ring above jar neck": 10.6,
        "move tie ring to bottle neck": 8.7,
        "raise tie gun after demo hold": 8.7,
    }
    for label, minimum_speed in doubled_speed_minima.items():
        matching_actions = [action for action in diagnostics["sdk_actions"] if action["label"] == label]
        if not matching_actions:
            raise AssertionError(f"Missing speed-check action: {label}")
        action_speeds = [
            float(action.get("motion_diagnostics", {}).get("commanded_joint_speed_rad_s", 0.0))
            for action in matching_actions
        ]
        if min(action_speeds) < minimum_speed:
            raise AssertionError(
                f"{label} should run at about 2x the previous demo speed, "
                f"expected min>={minimum_speed:.1f} rad/s, got {action_speeds}"
            )
    if "rotate held material" in labels:
        raise AssertionError("Leaf yaw should change through the arm path, not through a separate instant rotation step")
    attached = [action["attached_body"] for action in diagnostics["sdk_actions"] if action["label"] == "vacuum on"]
    for body_name in ("staged_bamboo_leaf_top", "staged_bamboo_leaf_bottom", "staged_label_paper"):
        if body_name not in attached:
            raise AssertionError(f"Expected vacuum attachment for {body_name}, got {attached}")
    if "staged_metal_weight" in attached:
        raise AssertionError(f"Production flow should not pick the old metal weight, got vacuum attachments={attached}")
    for action in [action for action in diagnostics["sdk_actions"] if action["label"] == "vacuum on"]:
        if action["last_suction_distance_m"] is None or action["last_suction_distance_m"] > 0.010:
            raise AssertionError(f"Vacuum attachment should happen near contact, got {action}")
    for index, action in enumerate(diagnostics["sdk_actions"]):
        if action["label"] != "move above pick":
            continue
        previous_action = diagnostics["sdk_actions"][index - 1] if index > 0 else None
        if previous_action is None or previous_action["label"] != "safe lift before pick approach":
            raise AssertionError(
                "Vacuum arm should enter every pickup by first moving to a safe overhead layer, "
                f"got move action={action}, previous={previous_action}"
            )
        motion = previous_action.get("motion_diagnostics", {})
        if motion.get("motion_primitive") != "linear_z" or motion.get("planned_xy_drift_mm", 999.0) > 1.5:
            raise AssertionError(f"Safe lift before pickup should be vertical, got {previous_action}")
        if previous_action["tcp_pose_mm_deg"][2] < 645.0:
            raise AssertionError(f"Safe lift before pickup should clear the mouth/material zone, got {previous_action}")

    side_standby = action_by_label(diagnostics["sdk_actions"], "prepare right arm side standby")["tcp_pose_mm_deg"]
    if side_standby[1] > diagnostics["jar_pos"][1] * 1000.0 - 360.0:
        raise AssertionError(
            "Right tie-gun arm should wait on the conveyor right side relative to the current station, "
            f"got standby_y={side_standby[1]:.1f} mm, jar_y={diagnostics['jar_pos'][1] * 1000.0:.1f} mm"
        )

    if "lower tie gun for demo" in labels:
        raise AssertionError("Tie-gun demo should not include a downward pressing/lowering action")
    for required_label in ("move tie ring to bottle neck", "hold tie gun at neck for demo"):
        if required_label not in labels:
            raise AssertionError(f"Missing visible tie-gun process action: {required_label}")
    for removed_label in ("transfer tie band to neck", "tighten tie band around neck"):
        if removed_label in labels:
            raise AssertionError(f"Tie-gun demo should not include detailed tie-band action: {removed_label}")

    tie_above_action = action_by_label(diagnostics["sdk_actions"], "move tie ring above jar neck")
    tie_above_ring = tie_above_action.get("ring_visual_pos_mm")
    tie_neck_action = action_by_label(diagnostics["sdk_actions"], "move tie ring to bottle neck")
    tie_neck_ring = tie_neck_action.get("ring_visual_pos_mm")
    if tie_above_ring is None:
        raise AssertionError(f"Tie above action should report the visible ring center, got {tie_above_action}")
    if tie_neck_ring is None:
        raise AssertionError(f"Tie neck action should report the visible ring center, got {tie_neck_action}")
    above_motion = tie_above_action.get("motion_diagnostics", {})
    if above_motion.get("waypoint_count", 1) > 2:
        raise AssertionError(f"Tie-gun approach should not look segmented, got diagnostics={above_motion}")
    jar_stop_xy_mm = [0.0, diagnostics["jar_pos"][1] * 1000.0]
    tie_neck_xy_error = (
        (tie_neck_ring[0] - jar_stop_xy_mm[0]) ** 2
        + (tie_neck_ring[1] - jar_stop_xy_mm[1]) ** 2
    ) ** 0.5
    if tie_neck_xy_error > 12.0 or not (596.0 <= tie_neck_ring[2] <= 612.0):
        raise AssertionError(f"Visible tie ring should stay near the bottle neck for the demo, got pose={tie_neck_ring}")
    tie_raise_action = action_by_label(diagnostics["sdk_actions"], "raise tie gun after demo hold")
    tie_raise_ring = tie_raise_action.get("ring_visual_pos_mm")
    if tie_raise_ring is None:
        raise AssertionError(f"Tie raise action should report the visible ring center, got {tie_raise_action}")
    raise_motion = tie_raise_action.get("motion_diagnostics", {})
    if raise_motion.get("motion_primitive") != "linear_z" or raise_motion.get("planned_xy_drift_mm", 999.0) > 1.5:
        raise AssertionError(f"Tie-gun should retract vertically away from the bottle neck before returning sideways, got {tie_raise_action}")
    tie_raise_index = diagnostics["sdk_actions"].index(tie_raise_action)
    tie_side_high_action = action_by_label(diagnostics["sdk_actions"], "move tie gun to side high standby")
    tie_side_high_index = diagnostics["sdk_actions"].index(tie_side_high_action)
    if tie_side_high_index <= tie_raise_index:
        raise AssertionError("Tie gun should move sideways at a high clearance after vertically retracting")
    tie_side_high_ring = tie_side_high_action.get("ring_visual_pos_mm")
    if tie_side_high_ring is None:
        raise AssertionError(f"Tie side-high action should report the visible ring center, got {tie_side_high_action}")
    if tie_side_high_ring[1] > diagnostics["jar_pos"][1] * 1000.0 - 360.0 or tie_side_high_ring[2] < 630.0:
        raise AssertionError(
            "Tie gun should reach the conveyor side while still high above the jar before lowering to standby, "
            f"got ring={tie_side_high_ring}"
        )
    tie_return_action = action_by_label(diagnostics["sdk_actions"], "return tie gun to side standby")
    tie_return_index = diagnostics["sdk_actions"].index(tie_return_action)
    if tie_return_index <= tie_side_high_index:
        raise AssertionError("Tie gun should lower to side standby only after moving sideways at high clearance")
    tie_return_ring = tie_return_action.get("ring_visual_pos_mm")
    if tie_return_ring is None:
        raise AssertionError(f"Tie return action should report the visible ring center, got {tie_return_action}")
    if tie_return_ring[1] > diagnostics["jar_pos"][1] * 1000.0 - 360.0:
        raise AssertionError(
            "Tie gun should visibly retract to the conveyor side before the jar exits, "
            f"got ring={tie_return_ring}, jar_y={diagnostics['jar_pos'][1] * 1000.0:.1f} mm"
        )
    first_post_tie_left_motion = next(
        (
            action
            for action in diagnostics["sdk_actions"][tie_return_index + 1 :]
            if action["label"] in {"move above pick", "descend to pick", "vacuum on"}
        ),
        None,
    )
    if first_post_tie_left_motion is not None:
        raise AssertionError(
            "There should be no left-arm material return after the tie-gun hold in the label-paper flow, "
            f"got first post-tie left action={first_post_tie_left_motion}"
        )
    for label, action, ring in [
        ("above", tie_above_action, tie_above_ring),
        ("neck", tie_neck_action, tie_neck_ring),
        ("raise", tie_raise_action, tie_raise_ring),
    ]:
        tcp_z = float(action["tcp_pose_mm_deg"][2])
        tool_to_ring_offset = tcp_z - float(ring[2])
        if abs(tool_to_ring_offset - 60.0) > 5.0:
            raise AssertionError(
                "Tie-gun ring should stay fixed relative to the tool; "
                f"the arm should provide the downward motion, got {label} offset={tool_to_ring_offset:.1f} mm"
            )
    hold_action = action_by_label(diagnostics["sdk_actions"], "hold tie gun at neck for demo")
    hold_seconds = hold_action["motion_diagnostics"].get("hold_seconds", 0.0)
    assert_close("Tie-gun demo hold seconds", hold_seconds, 1.0, 0.05)
    if hold_action["motion_diagnostics"].get("viewer_sync_frames", 9999) > 60:
        raise AssertionError(f"Tie-gun demo hold should not look stalled in the viewer, got {hold_action}")

    coverage = diagnostics.get("ceramic_leaf_coverage")
    if coverage is None:
        raise AssertionError("Animation diagnostics should report whether bamboo leaves cover the ceramic disc")
    if coverage["covered_ratio"] < 0.985:
        raise AssertionError(f"Bamboo leaves should cover the ceramic disc, got coverage={coverage}")

    if diagnostics["final_tie_alpha"] > 0.05:
        raise AssertionError("Simplified tie-gun demo should not leave a final tie band on the bottle")
    if diagnostics["jar_pos"][0] < 0.75:
        raise AssertionError(f"Expected jar to exit conveyor, got x={diagnostics['jar_pos'][0]:.3f}")
    belt_checks = diagnostics.get("conveyor_belt_speed_checks")
    if not belt_checks:
        raise AssertionError("Animation diagnostics should report conveyor belt marker speed checks")
    for check in belt_checks:
        assert_close(f"{check['phase']} belt marker speed ratio", check["marker_to_jar_ratio"], 1.0, 0.03)
    timing = diagnostics.get("demo_timing")
    if not timing:
        raise AssertionError("Animation diagnostics should report high-speed demo timing")
    assert_close("Conveyor entry seconds", timing["conveyor_entry_seconds"], 4.0, 0.05)
    assert_close("Conveyor exit seconds", timing["conveyor_exit_seconds"], 4.0, 0.05)
    if timing["leaf_settle_steps"] > 25 or timing["label_paper_settle_steps"] > 80:
        raise AssertionError(f"Material settle pauses should be short in demo mode, got timing={timing}")
    if timing["leaf_place_waypoints"] > 6:
        raise AssertionError(f"Leaf transfer path should not be over-segmented in demo mode, got timing={timing}")
    if timing.get("leaf_place_clearance_mm", 0.0) < 50.0:
        raise AssertionError(f"Bamboo leaves should approach from a safer height before descent, got timing={timing}")
    if timing.get("label_paper_place_clearance_mm", 0.0) < 20.0:
        raise AssertionError(f"Label paper should report an explicit place clearance, got timing={timing}")
    label_droop_animation = diagnostics.get("label_paper_droop_animation")
    if not label_droop_animation:
        raise AssertionError("Animation diagnostics should report the gradual label-paper droop animation")
    if label_droop_animation.get("seconds", 0.0) < 0.30:
        raise AssertionError(f"Label paper droop should be animated over time, got {label_droop_animation}")
    if label_droop_animation.get("steps", 0) < 8:
        raise AssertionError(f"Label paper droop animation needs multiple frames, got {label_droop_animation}")
    left_home_labels = {
        "return left arm home",
        "return left arm home after top leaf",
        "return left arm home after bottom leaf",
    }
    for index, action in enumerate(diagnostics["sdk_actions"]):
        if action["label"] not in left_home_labels:
            continue
        previous_action = diagnostics["sdk_actions"][index - 1] if index > 0 else None
        if previous_action is None or previous_action["label"] != "safe lift before left home":
            raise AssertionError(
                "Left arm should lift vertically to a safe clearance before returning home, "
                f"got return action={action}, previous={previous_action}"
            )
        motion = previous_action.get("motion_diagnostics", {})
        if motion.get("motion_primitive") != "linear_z":
            raise AssertionError(f"Safe lift before home should be a vertical TCP move, got {motion}")
        if previous_action["tcp_pose_mm_deg"][2] < 645.0:
            raise AssertionError(f"Safe lift before home should clear mouth materials, got {previous_action}")
        if motion.get("planned_xy_drift_mm", 999.0) > 1.5:
            raise AssertionError(f"Safe lift before home should not sweep over placed materials, got {motion}")
    if "metal_weight_pos" in diagnostics:
        raise AssertionError(f"Production demo diagnostics should not report the old metal weight: {diagnostics['metal_weight_pos']}")
    label_xy_offset = (
        (diagnostics["label_paper_pos"][0] - diagnostics["jar_pos"][0]) ** 2
        + (diagnostics["label_paper_pos"][1] - diagnostics["jar_pos"][1]) ** 2
    ) ** 0.5 * 1000.0
    if label_xy_offset > 80.0:
        raise AssertionError(f"Label paper should remain on the jar mouth stack, got offset={label_xy_offset:.1f} mm")
    assert_close("label paper final yaw", diagnostics["label_paper_yaw_deg"], 45.0, 6.0)
    label_droop = diagnostics["label_paper_end_droop_mm"]
    if abs(label_droop[0]) > 12.0 or label_droop[1] < 30.0:
        raise AssertionError(
            "Long label paper should still show only one side hanging after the full cycle, "
            f"got droop={label_droop}"
        )
    mouth_stack = diagnostics.get("mouth_stack_after_tie")
    if not mouth_stack:
        raise AssertionError("Animation diagnostics should report the mouth material stack order after tie-gun hold")
    if mouth_stack["label_bottom_above_ceramic_top_mm"] < 0.0:
        raise AssertionError(
            "Tie-gun hold should not drive the label paper below the ceramic disc, "
            f"got mouth_stack_after_tie={mouth_stack}"
        )
    if args.scope == "quick":
        print("Production demo quick animation checks OK")
        return

    leaf_transfer_actions = [
        action
        for action in diagnostics["sdk_actions"]
        if action["label"] == "move above place" and action.get("attached_body", "").startswith("staged_bamboo_leaf")
    ]
    if len(leaf_transfer_actions) != 2:
        raise AssertionError(f"Expected two bamboo leaf transfer actions, got {leaf_transfer_actions}")
    for action in leaf_transfer_actions:
        motion = action.get("motion_diagnostics", {})
        if motion.get("motion_primitive") != "loaded_transfer":
            raise AssertionError(f"Leaf transfer should be expressed as an SDK-style loaded-transfer primitive, got {motion}")
        if motion.get("path_mode") != "blended_joint_waypoints":
            raise AssertionError(f"Leaf transfer should use a blended path instead of stop-start waypoints, got {motion}")
        if motion.get("intermediate_stop_count") != 0:
            raise AssertionError(f"Leaf transfer should not stop at intermediate waypoints, got {motion}")

    for vertical_label in ("lift after pick", "descend to place", "retreat after release"):
        vertical_actions = [
            action
            for action in diagnostics["sdk_actions"]
            if action["label"] == vertical_label
            and (
                str(action.get("attached_body", "")).startswith("staged_bamboo_leaf")
                or vertical_label == "retreat after release"
            )
        ]
        if not vertical_actions:
            raise AssertionError(f"Missing vertical primitive actions for {vertical_label}")
        for action in vertical_actions:
            motion = action.get("motion_diagnostics", {})
            if motion.get("motion_primitive") != "linear_z":
                raise AssertionError(f"{vertical_label} should use an SDK-style linear-z primitive, got {motion}")
            if motion.get("planned_xy_drift_mm", 999.0) > 1.5:
                raise AssertionError(f"{vertical_label} should keep x/y fixed while moving in z, got {motion}")

    route = diagnostics.get("industrial_route_checks")
    if not route:
        raise AssertionError("Animation diagnostics should report industrial route quality checks")
    missing_route = [
        key
        for key in (
            "left_loaded_transfer_count",
            "left_loaded_transfer_min_z_mm",
            "left_loaded_transfer_clearance_mm",
            "left_retreat_min_vertical_mm",
            "left_vertical_lift_max_xy_drift_mm",
            "left_vertical_place_max_xy_drift_mm",
            "left_vertical_retreat_max_xy_drift_mm",
            "left_leaf_place_max_tilt_deg",
            "bottom_leaf_place_clearance_mm",
            "right_standby_side_y_mm",
            "right_approach_min_z_mm",
            "right_neck_xy_drift_mm",
            "right_raise_min_vertical_mm",
        )
        if key not in route
    ]
    if missing_route:
        raise AssertionError(f"Industrial route checks missing keys: {missing_route}, got {route}")
    if route["left_loaded_transfer_count"] < 3:
        raise AssertionError(f"Left arm should report all loaded transfers, got {route}")
    if route["left_loaded_transfer_min_z_mm"] < 645.0:
        raise AssertionError(f"Loaded left-arm transfer should use a high clearance layer, got {route}")
    if route["left_loaded_transfer_clearance_mm"] < 45.0:
        raise AssertionError(f"Loaded left-arm transfer should clear the jar/material stack generously, got {route}")
    if route["left_retreat_min_vertical_mm"] < 1.0:
        raise AssertionError(f"Left arm should visibly retreat upward after releases, got {route}")
    if route["left_vertical_lift_max_xy_drift_mm"] > 5.0:
        raise AssertionError(f"Suction lift should be nearly vertical, got {route}")
    if route["left_vertical_place_max_xy_drift_mm"] > 5.0:
        raise AssertionError(f"Suction placement descent should be nearly vertical, got {route}")
    if route["left_vertical_retreat_max_xy_drift_mm"] > 5.0:
        raise AssertionError(f"Suction retreat should be nearly vertical, got {route}")
    if route["left_leaf_place_max_tilt_deg"] > 2.0:
        raise AssertionError(f"Suction cup should stay visually vertical while placing leaves, got {route}")
    if route["bottom_leaf_place_clearance_mm"] < 40.0:
        raise AssertionError(f"Second leaf should approach from a safe vertical clearance, got {route}")
    if route.get("label_paper_final_jar_offset_mm", 999.0) > 80.0:
        raise AssertionError(f"Label paper should remain on the jar mouth stack, got {route}")
    if not (4.0 <= route.get("bottom_leaf_release_above_top_release_mm", 0.0) <= 14.0):
        raise AssertionError(
            "Second bamboo leaf should be released slightly above the first without leaving a visible air layer, "
            f"got {route}"
        )
    if route["right_standby_side_y_mm"] > diagnostics["jar_pos"][1] * 1000.0 - 360.0:
        raise AssertionError(f"Right arm should start from the conveyor side before approach, got {route}")
    if route["right_approach_min_z_mm"] < 640.0:
        raise AssertionError(f"Right tie-gun approach should stay on a high approach layer, got {route}")
    if route["right_neck_xy_drift_mm"] > 50.0:
        raise AssertionError(f"Right tie-gun neck approach should stay centered over the bottle neck, got {route}")
    if route["right_raise_min_vertical_mm"] < 24.0:
        raise AssertionError(f"Right tie-gun should raise clear of the work after tying, got {route}")

    motion_actions = [
        action
        for action in diagnostics["sdk_actions"]
        if action["label"]
        in {
            "move above pick",
            "descend to pick",
            "lift after pick",
            "move above place",
            "descend to place",
            "retreat after release",
            "return left arm home after top leaf",
            "return left arm home after bottom leaf",
            "return left arm home",
            "prepare right arm side standby",
            "move tie ring above jar neck",
            "move tie ring to bottle neck",
            "raise tie gun after demo hold",
        }
    ]
    slow_actions = []
    for action in motion_actions:
        motion = action.get("motion_diagnostics", {})
        limit = 1.50 if motion.get("path_mode") == "blended_joint_waypoints" else 1.25
        target_pose = motion.get("target_tcp_pose_mm_deg", {})
        if motion.get("elapsed_sim_seconds", 99.0) > limit:
            slow_actions.append(action)
    if slow_actions:
        raise AssertionError(f"Mechanical arm motions should be brisk with minimal pauses, got slow actions={slow_actions[:3]}")

    jar_xy = diagnostics["jar_pos"][:2]
    bottom_leaf_xy_offset = (
        (diagnostics["bottom_leaf_pos"][0] - jar_xy[0]) ** 2
        + (diagnostics["bottom_leaf_pos"][1] - jar_xy[1]) ** 2
    ) ** 0.5
    if bottom_leaf_xy_offset > 0.085:
        raise AssertionError(f"Bottom bamboo leaf should be centered better on jar mouth, got {bottom_leaf_xy_offset:.3f} m")
    top_leaf_xy_offset = (
        (diagnostics["top_leaf_pos"][0] - jar_xy[0]) ** 2
        + (diagnostics["top_leaf_pos"][1] - jar_xy[1]) ** 2
    ) ** 0.5
    if top_leaf_xy_offset > 0.035:
        raise AssertionError(f"Top bamboo leaf should be centered on jar mouth, got {top_leaf_xy_offset:.3f} m")
    for key in ("top_leaf_pos", "bottom_leaf_pos"):
        offset_xy = ((diagnostics[key][0] - jar_xy[0]) ** 2 + (diagnostics[key][1] - jar_xy[1]) ** 2) ** 0.5
        if offset_xy > 0.16:
            raise AssertionError(f"{key} should travel near exiting jar mouth, got xy offset {offset_xy:.3f} m")

    for key in ("top_leaf_end_droop_mm", "bottom_leaf_end_droop_mm"):
        if max(diagnostics[key]) < 22.0:
            raise AssertionError(f"{key} should show visible flexible droop, got {diagnostics[key]}")
    tie_collection = diagnostics.get("tie_gun_collection_effect", {})
    for key in ("lotus_leaf_lowered_mm", "white_paper_lowered_mm"):
        if abs(float(tie_collection.get(key, 0.0))) > 3.0:
            raise AssertionError(f"Tie-gun hold should not press the preloaded rigid stack, got {tie_collection}")
    for key in ("top_leaf_center_lowered_mm", "bottom_leaf_center_lowered_mm"):
        if abs(float(tie_collection.get(key, 0.0))) > 6.0:
            raise AssertionError(f"Flexible bamboo leaves should only settle slightly during tie-gun hold, got {tie_collection}")
    if abs(float(tie_collection.get("label_paper_lowered_mm", 0.0))) > 8.0:
        raise AssertionError(f"Flexible label paper should only settle slightly during tie-gun hold, got {tie_collection}")

    print("Production demo animation checks OK")


if __name__ == "__main__":
    main()
