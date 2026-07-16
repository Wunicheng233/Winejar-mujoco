#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mujoco_xarm6.sim.xarm_sim_api import SimXArmAPI

SCENE_PATH = ROOT / "scene" / "scene_winejar_vacuum.xml"
OUTPUT_ROOT = ROOT / "data" / "vacuum_leaf_place_sequence"
HOME_JOINTS_DEG = [0.0, -35.0, -70.0, 0.0, 105.0, 0.0]


def site_pose_for_top_down(model, data, site_name: str, z_offset_m: float, yaw_deg: float):
    site_id = model.site(site_name).id
    pos = data.site_xpos[site_id].copy()
    return {
        "x": float((pos[0]) * 1000.0),
        "y": float((pos[1]) * 1000.0),
        "z": float((pos[2] + z_offset_m) * 1000.0),
        "roll": 180.0,
        "pitch": 0.0,
        "yaw": float(yaw_deg),
    }


def call(label: str, func, *args, **kwargs):
    code = func(*args, **kwargs)
    print(f"{label}: {code}")
    if code != 0:
        owner = getattr(func, "__self__", None)
        diagnostics = getattr(owner, "last_motion_diagnostics", None)
        if diagnostics:
            print(f"{label} diagnostics: {json.dumps(diagnostics, indent=2)}")
        raise RuntimeError(f"{label} failed with code {code}")
    return code


def run_leaf_transfer(
    arm: SimXArmAPI,
    model,
    data,
    leaf_label: str,
    pick_site: str,
    target_site: str,
    pick_yaw_deg: float,
    place_yaw_deg: float,
    speed: float,
):
    print(f"\n=== {leaf_label} ===")
    pick_above = site_pose_for_top_down(model, data, pick_site, 0.13, pick_yaw_deg)
    pick_contact = site_pose_for_top_down(model, data, pick_site, 0.004, pick_yaw_deg)
    lift_after_pick = site_pose_for_top_down(model, data, pick_site, 0.16, pick_yaw_deg)
    place_above = site_pose_for_top_down(model, data, target_site, 0.10, place_yaw_deg)
    place_contact = site_pose_for_top_down(model, data, target_site, 0.024, place_yaw_deg)

    call("open vacuum before approach", arm.set_cgpio_digital, 9, 0)
    call("move above pick", arm.set_position, **pick_above, speed=speed, timeout=20)
    call("descend to pick", arm.set_position, **pick_contact, speed=min(speed, 35), timeout=18, joint_tolerance=0.008)
    call("vacuum on", arm.set_cgpio_digital, 9, 1)
    if arm.attached_body_name is None:
        distance_text = "unknown"
        if arm.last_suction_distance_m is not None:
            distance_text = f"{arm.last_suction_distance_m * 1000.0:.1f} mm"
        raise RuntimeError(f"{leaf_label}: vacuum did not attach near {pick_site}, suction distance={distance_text}")
    print(f"attached: {arm.attached_body_name}, suction distance={arm.last_suction_distance_m * 1000.0:.1f} mm")
    call("lift leaf", arm.set_position, **lift_after_pick, speed=speed, timeout=12)
    call("move above place", arm.set_position, **place_above, speed=speed, timeout=20)
    call("descend to place", arm.set_position, **place_contact, speed=min(speed, 35), timeout=18, joint_tolerance=0.008)
    call("vacuum off", arm.set_cgpio_digital, 9, 0)
    arm.step(200)
    call("retreat after release", arm.set_position, **place_above, speed=speed, timeout=12)


def initialize_robot_home(arm: SimXArmAPI):
    arm.data.qpos[arm._qposadr] = np.radians(HOME_JOINTS_DEG)
    arm.data.qvel[:] = 0
    arm.data.ctrl[:6] = 0
    mujoco.mj_forward(arm.model, arm.data)
    print(f"Initialized simulated robot home joints [deg]: {HOME_JOINTS_DEG}")
    print(f"Initial TCP pose [mm/deg]: {[round(v, 3) for v in arm.get_position()[1]]}")


def body_yaw_deg(model, data, body_name: str) -> float:
    body_id = model.body(body_name).id
    rot = data.xmat[body_id].reshape(3, 3)
    return float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))


def angle_difference_deg(a: float, b: float) -> float:
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)


def main():
    parser = argparse.ArgumentParser(
        description="Run a realistic SDK-style vacuum sequence for placing two bamboo leaves in MuJoCo."
    )
    parser.add_argument("--scene", type=Path, default=SCENE_PATH)
    parser.add_argument("--speed", type=float, default=85.0, help="Approximate TCP speed in mm/s for simulated commands.")
    parser.add_argument("--settle-steps", type=int, default=500)
    parser.add_argument("--viewer", action="store_true", help="Open a MuJoCo viewer and show the sequence live.")
    parser.add_argument("--realtime", action="store_true", help="Sleep between simulation steps so motion is easier to watch.")
    parser.add_argument("--hold-open", action="store_true", help="Keep the viewer open after the sequence finishes.")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    viewer = mujoco.viewer.launch_passive(model, data) if args.viewer else None
    arm = SimXArmAPI(model, data, viewer=viewer, realtime=args.realtime or args.viewer)

    try:
        print(f"Loaded scene: {args.scene}")
        print("Control style: SimXArmAPI.set_position + SimXArmAPI.set_cgpio_digital, no object teleportation.")
        if args.viewer:
            print("Viewer mode: live MuJoCo window is open; close it to stop watching after completion.")
        initialize_robot_home(arm)
        arm.step(args.settle_steps)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_ROOT / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        run_leaf_transfer(
            arm,
            model,
            data,
            leaf_label="top leaf to 0 deg target",
            pick_site="bamboo_leaf_top_pick_site",
            target_site="jar_leaf_target_0",
            pick_yaw_deg=90.0,
            place_yaw_deg=0.0,
            speed=args.speed,
        )
        run_leaf_transfer(
            arm,
            model,
            data,
            leaf_label="bottom leaf to 90 deg target",
            pick_site="bamboo_leaf_bottom_pick_site",
            target_site="jar_leaf_target_90",
            pick_yaw_deg=90.0,
            place_yaw_deg=90.0,
            speed=args.speed,
        )

        mujoco.mj_forward(model, data)
        top_yaw_deg = body_yaw_deg(model, data, "bamboo_leaf_top")
        bottom_yaw_deg = body_yaw_deg(model, data, "bamboo_leaf_bottom")
        diagnostics = {
            "timestamp": timestamp,
            "scene": str(args.scene),
            "top_leaf_center_m": data.site_xpos[model.site("bamboo_leaf_top_center_site").id].tolist(),
            "bottom_leaf_center_m": data.site_xpos[model.site("bamboo_leaf_bottom_center_site").id].tolist(),
            "target_0_m": data.site_xpos[model.site("jar_leaf_target_0").id].tolist(),
            "target_90_m": data.site_xpos[model.site("jar_leaf_target_90").id].tolist(),
            "attached_body_name": arm.attached_body_name,
            "top_leaf_yaw_deg": top_yaw_deg,
            "bottom_leaf_yaw_deg": bottom_yaw_deg,
            "leaf_crossing_angle_deg": angle_difference_deg(top_yaw_deg, bottom_yaw_deg),
        }
        diagnostics["top_leaf_to_target_0_mm"] = float(
            np.linalg.norm(
                np.array(diagnostics["top_leaf_center_m"]) - np.array(diagnostics["target_0_m"])
            )
            * 1000.0
        )
        diagnostics["bottom_leaf_to_target_90_mm"] = float(
            np.linalg.norm(
                np.array(diagnostics["bottom_leaf_center_m"]) - np.array(diagnostics["target_90_m"])
            )
            * 1000.0
        )
        output_path = output_dir / "result.json"
        output_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        print("\nFinal diagnostics:")
        print(json.dumps(diagnostics, indent=2))
        print(f"Saved result: {output_path}")
        if viewer is not None and args.hold_open:
            print("Sequence complete. Viewer will stay open until you close it or press Ctrl+C.")
            while viewer.is_running():
                viewer.sync()
        return 0
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
