#!/usr/bin/env python3
"""Select a detected target and plan a top-down simulated xArm TCP pose."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mujoco_xarm6.sim.xarm_sim_api import SimXArmAPI


DEFAULT_SCENE = ROOT / "scene" / "scene_winejar_vacuum.xml"
HOME_JOINTS_DEG = [0.0, -35.0, -70.0, 0.0, 105.0, 0.0]


def load_vision_result(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "vision_result.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing vision result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def select_detection(
    detections: list[dict[str, Any]],
    label_contains: str,
    min_score: float,
) -> dict[str, Any]:
    label_filter = label_contains.lower().strip()
    candidates = []
    for detection in detections:
        label = str(detection.get("label", "")).lower()
        if label_filter and label_filter not in label:
            continue
        if float(detection.get("score", 0.0)) < min_score:
            continue
        if detection.get("point_world_m") is None:
            continue
        candidates.append(detection)
    if not candidates:
        summary = [
            {
                "label": det.get("label"),
                "score": det.get("score"),
                "point_world_m": det.get("point_world_m"),
            }
            for det in detections
        ]
        raise RuntimeError(
            "No usable detection matched "
            f"label_contains={label_contains!r}, min_score={min_score}. "
            f"Available detections: {json.dumps(summary, ensure_ascii=False)}"
        )
    return max(candidates, key=lambda det: float(det["score"]))


def initialize_robot_home(arm: SimXArmAPI) -> None:
    arm.data.qpos[arm._qposadr] = np.radians(HOME_JOINTS_DEG)
    arm.data.qvel[:] = 0
    arm.data.ctrl[:6] = 0
    mujoco.mj_forward(arm.model, arm.data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-result", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--label-contains", default="bamboo", help="Case-insensitive label substring.")
    parser.add_argument("--min-score", type=float, default=0.20)
    parser.add_argument("--above-mm", type=float, default=120.0)
    parser.add_argument("--yaw-deg", type=float, default=90.0)
    parser.add_argument("--verify-ik", action="store_true")
    args = parser.parse_args()

    result = load_vision_result(args.vision_result.expanduser().resolve())
    detection = select_detection(
        result.get("detections", []),
        label_contains=args.label_contains,
        min_score=args.min_score,
    )
    point_world_m = np.array(detection["point_world_m"], dtype=float)
    above_pose = {
        "x": float(point_world_m[0] * 1000.0),
        "y": float(point_world_m[1] * 1000.0),
        "z": float(point_world_m[2] * 1000.0 + args.above_mm),
        "roll": 180.0,
        "pitch": 0.0,
        "yaw": float(args.yaw_deg),
    }

    output = {
        "schema_version": 1,
        "vision_result": str(args.vision_result),
        "selected_detection": detection,
        "planned_tcp_pose_mm_deg": above_pose,
        "assumption": "scene_winejar_vacuum uses MuJoCo world coordinates as the simulated xArm base frame.",
    }

    if args.verify_ik:
        model = mujoco.MjModel.from_xml_path(str(args.scene.expanduser().resolve()))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        arm = SimXArmAPI(model, data)
        initialize_robot_home(arm)
        code = arm.set_position(**above_pose, speed=120.0, timeout=20)
        output["ik_verification"] = {
            "code": int(code),
            "diagnostics": arm.last_motion_diagnostics,
            "final_tcp_pose_mm_deg": arm.get_position()[1],
        }

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
