#!/usr/bin/env python3
"""Use simulated RGB-D vision to cross-place two bamboo leaves in MuJoCo.

Each pick is localized from a fresh RGB-D frame. The ceramic center is detected
before it is occluded, then reused as the visual placement reference.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any

# OpenCV's bundled Qt looks for a removed package-local fonts directory unless
# an explicit system font directory is supplied.
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

import cv2
import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mujoco_xarm6.sim.xarm_sim_api import SimXArmAPI


SCENE_PATH = ROOT / "scene" / "scene_winejar_vacuum.xml"
OUTPUT_ROOT = ROOT / "data" / "visual_pick_first_leaf"
HOME_JOINTS_DEG = [0.0, -35.0, -70.0, 0.0, 105.0, 0.0]
sim_capture = SourceFileLoader(
    "sim_capture",
    str(ROOT / "scripts" / "04_capture_sim_camera_frame.py"),
).load_module()


class CameraPreview:
    def __init__(
        self,
        model: mujoco.MjModel,
        camera: str,
        width: int,
        height: int,
        every_steps: int,
        window_name: str = "sim global camera",
    ) -> None:
        self.camera = camera
        self.every_steps = max(1, every_steps)
        self.window_name = window_name
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.step_count = 0
        self.enabled = True
        self.vision_overlay_bgr: np.ndarray | None = None
        self.selected_detections: list[tuple[str, dict[str, Any], tuple[int, int, int]]] = []

    def set_vision_result(
        self,
        annotated_bgr: np.ndarray,
        selected_detections: list[tuple[str, dict[str, Any], tuple[int, int, int]]],
    ) -> None:
        self.vision_overlay_bgr = annotated_bgr.copy()
        self.selected_detections = selected_detections

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        del model
        if not self.enabled:
            return
        self.step_count += 1
        if self.step_count % self.every_steps != 0:
            return
        self.renderer.update_scene(data, camera=self.camera)
        color_rgb = self.renderer.render()
        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        if self.vision_overlay_bgr is not None:
            overlay = self.vision_overlay_bgr
            if overlay.shape[:2] != color_bgr.shape[:2]:
                overlay = cv2.resize(overlay, (color_bgr.shape[1], color_bgr.shape[0]))
            color_bgr = cv2.addWeighted(color_bgr, 0.45, overlay, 0.55, 0)
        for row, (role, detection, color) in enumerate(self.selected_detections):
            centroid = detection.get("centroid_px")
            point_world = detection.get("point_world_m")
            label = detection.get("label", "selected")
            score = float(detection.get("score", 0.0))
            if centroid is not None:
                u = int(round(float(centroid[0]) * color_bgr.shape[1] / 640.0))
                v = int(round(float(centroid[1]) * color_bgr.shape[0] / 480.0))
                cv2.drawMarker(
                    color_bgr,
                    (u, v),
                    color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=28,
                    thickness=3,
                )
            if point_world is not None:
                text = (
                    f"{role} BY VISION: {label} {score:.2f} | "
                    f"world m=({point_world[0]:.3f},{point_world[1]:.3f},{point_world[2]:.3f})"
                )
                cv2.putText(
                    color_bgr,
                    text,
                    (10, 26 + row * 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    color,
                    2,
                )
        cv2.putText(
            color_bgr,
            "MuJoCo global camera + GroundingDINO/SAM2 overlay | q close preview",
            (10, color_bgr.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.imshow(self.window_name, color_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self.enabled = False
            cv2.destroyWindow(self.window_name)

    def close(self) -> None:
        self.renderer.close()
        cv2.destroyWindow(self.window_name)


def initialize_robot_home(arm: SimXArmAPI) -> None:
    arm.data.qpos[arm._qposadr] = np.radians(HOME_JOINTS_DEG)
    arm.data.qvel[:] = 0
    arm.data.ctrl[:6] = 0
    mujoco.mj_forward(arm.model, arm.data)


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


def hide_sites(model: mujoco.MjModel) -> None:
    model.site_rgba[:, 3] = 0.0


def capture_current_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: str,
    output_root: Path,
    width: int,
    height: int,
) -> Path:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        raise RuntimeError(f"Camera {camera!r} not found")

    color_bgr, depth_raw = sim_capture.render_color_and_depth(
        model=model,
        data=data,
        camera=camera,
        width=width,
        height=height,
    )

    timestamp = datetime.now().astimezone()
    output_dir = output_root.expanduser().resolve() / timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_dir.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(output_dir / "color.png"), color_bgr)
    cv2.imwrite(str(output_dir / "depth_raw.png"), depth_raw)
    cv2.imwrite(
        str(output_dir / "depth_preview.png"),
        sim_capture.make_depth_preview(depth_raw, sim_capture.DEPTH_SCALE_M_PER_UNIT),
    )

    intrinsics = sim_capture.intrinsics_from_fovy(width, height, model.cam_fovy[cam_id])
    cam_to_world = sim_capture.opencv_camera_to_world(data, cam_id)
    valid_depth = depth_raw > 0
    metadata = {
        "schema_version": 1,
        "frame_id": output_dir.name,
        "captured_at": timestamp.isoformat(timespec="milliseconds"),
        "source": "mujoco",
        "scene": str(SCENE_PATH),
        "camera": {
            "name": camera,
            "id": int(cam_id),
            "fovy_deg": float(model.cam_fovy[cam_id]),
            "position_world_m": data.cam_xpos[cam_id].tolist(),
            "opencv_camera_to_world": cam_to_world.tolist(),
            "world_to_opencv_camera": np.linalg.inv(cam_to_world).tolist(),
            "mujoco_camera_xmat": data.cam_xmat[cam_id].reshape(3, 3).tolist(),
        },
        "stream": {
            "width": width,
            "height": height,
            "fps": None,
            "color_format": "BGR8_PNG",
            "depth_format": "Z16_PNG",
            "depth_aligned_to": "color",
            "depth_scale_m_per_unit": sim_capture.DEPTH_SCALE_M_PER_UNIT,
            "coordinate_convention": "+x right, +y down, +z forward",
        },
        "intrinsics": intrinsics,
        "quality": {
            "valid_depth_fraction": float(np.mean(valid_depth)),
            "minimum_valid_depth_m": (
                float(np.min(depth_raw[valid_depth]) * sim_capture.DEPTH_SCALE_M_PER_UNIT)
                if np.any(valid_depth)
                else None
            ),
            "maximum_valid_depth_m": (
                float(np.max(depth_raw[valid_depth]) * sim_capture.DEPTH_SCALE_M_PER_UNIT)
                if np.any(valid_depth)
                else None
            ),
        },
        "files": {
            "color_bgr": "color.png",
            "depth_raw": "depth_raw.png",
            "depth_preview": "depth_preview.png",
            "metadata": "frame.json",
        },
    }
    (output_dir / "frame.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_dir


def run_vision(
    frame_dir: Path,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    vision_env: str,
) -> dict[str, Any]:
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        vision_env,
        "python",
        str(REPO_ROOT / "vision" / "scripts" / "01_offline_grounded_sam2.py"),
        "--frame-dir",
        str(frame_dir),
        "--text-prompt",
        text_prompt,
        "--box-threshold",
        str(box_threshold),
        "--text-threshold",
        str(text_threshold),
    ]
    print("Running vision:", " ".join(cmd))
    vision_env_vars = os.environ.copy()
    vision_env_vars["HF_HUB_OFFLINE"] = "1"
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=vision_env_vars, check=True)
    return json.loads((frame_dir / "vision_result.json").read_text(encoding="utf-8"))


def select_detection(
    detections: list[dict[str, Any]],
    label_contains: str,
    min_score: float,
    exclude_xy_m: np.ndarray | None = None,
    min_xy_distance_m: float = 0.0,
) -> dict[str, Any]:
    candidates = []
    label_filter = label_contains.lower().strip()
    for detection in detections:
        label = str(detection.get("label", "")).lower()
        if label_filter and label_filter not in label:
            continue
        if float(detection.get("score", 0.0)) < min_score:
            continue
        if detection.get("point_world_m") is None:
            continue
        point_world = np.asarray(detection["point_world_m"], dtype=float)
        if exclude_xy_m is not None:
            distance_m = float(np.linalg.norm(point_world[:2] - exclude_xy_m[:2]))
            if distance_m < min_xy_distance_m:
                continue
        candidates.append(detection)
    if not candidates:
        raise RuntimeError(f"No detection matched label={label_contains!r}, min_score={min_score}")
    return max(candidates, key=lambda item: float(item["score"]))


def load_annotated_image(frame_dir: Path) -> np.ndarray | None:
    path = frame_dir / "annotated.png"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image


def site_position(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> list[float]:
    mujoco.mj_forward(model, data)
    return data.site_xpos[model.site(site_name).id].astype(float).tolist()


def body_xy_yaw(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> tuple[list[float], float]:
    mujoco.mj_forward(model, data)
    body_id = model.body(body_name).id
    position = data.xpos[body_id].astype(float)
    rotation = data.xmat[body_id].reshape(3, 3)
    yaw_deg = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    return position.tolist(), yaw_deg


def target_pose(point_world_m: np.ndarray, z_offset_m: float, yaw_deg: float) -> dict[str, float]:
    return {
        "x": float(point_world_m[0] * 1000.0),
        "y": float(point_world_m[1] * 1000.0),
        "z": float((point_world_m[2] + z_offset_m) * 1000.0),
        "roll": 180.0,
        "pitch": 0.0,
        "yaw": float(yaw_deg),
    }


def execute_pick_place(
    arm: SimXArmAPI,
    pick_point_world_m: np.ndarray,
    place_point_world_m: np.ndarray,
    pick_yaw_deg: float,
    place_yaw_deg: float,
    approach_offset_m: float,
    contact_offset_m: float,
    lift_offset_m: float,
    place_above_offset_m: float,
    place_contact_offset_m: float,
    speed: float,
    cycle_name: str,
) -> dict[str, Any]:
    approach = target_pose(pick_point_world_m, approach_offset_m, pick_yaw_deg)
    contact = target_pose(pick_point_world_m, contact_offset_m, pick_yaw_deg)
    lift = target_pose(pick_point_world_m, lift_offset_m, pick_yaw_deg)
    place_above = target_pose(place_point_world_m, place_above_offset_m, place_yaw_deg)
    place_contact = target_pose(place_point_world_m, place_contact_offset_m, place_yaw_deg)

    call(f"{cycle_name}: vacuum off", arm.set_cgpio_digital, 9, 0)
    call(f"{cycle_name}: move above visual leaf", arm.set_position, **approach, speed=speed, timeout=20)
    call(
        f"{cycle_name}: descend to visual suction point",
        arm.set_position,
        **contact,
        speed=min(speed, 35.0),
        timeout=18,
        joint_tolerance=0.008,
    )
    call(f"{cycle_name}: vacuum on", arm.set_cgpio_digital, 9, 1)
    if arm.attached_body_name is None:
        distance_text = (
            f"{arm.last_suction_distance_m * 1000.0:.1f} mm"
            if arm.last_suction_distance_m is not None
            else "unknown"
        )
        raise RuntimeError(f"{cycle_name}: vacuum did not attach; distance={distance_text}")
    attached_body_name = arm.attached_body_name
    suction_distance_m = arm.last_suction_distance_m
    print(f"{cycle_name}: attached {attached_body_name} at {suction_distance_m * 1000.0:.1f} mm")

    call(f"{cycle_name}: lift leaf", arm.set_position, **lift, speed=speed, timeout=15)
    call(f"{cycle_name}: move above visual ceramic center", arm.set_position, **place_above, speed=speed, timeout=20)
    call(
        f"{cycle_name}: descend to placement height",
        arm.set_position,
        **place_contact,
        speed=min(speed, 35.0),
        timeout=18,
        joint_tolerance=0.008,
    )
    call(f"{cycle_name}: release leaf", arm.set_cgpio_digital, 9, 0)
    arm.step(180)
    call(f"{cycle_name}: retreat vertically", arm.set_position, **place_above, speed=speed, timeout=15)
    return {
        "attached_body_name": attached_body_name,
        "suction_distance_m": suction_distance_m,
        "poses": {
            "approach": approach,
            "contact": contact,
            "lift": lift,
            "place_above": place_above,
            "place_contact": place_contact,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=SCENE_PATH)
    parser.add_argument("--camera", default="global_overhead_camera")
    parser.add_argument("--text-prompt", default="bamboo leaf. ceramic disk.")
    parser.add_argument("--vision-env", default="winejar-vision")
    parser.add_argument("--label-contains", default="bamboo")
    parser.add_argument("--place-label-contains", default="ceramic")
    parser.add_argument("--box-threshold", type=float, default=0.45)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument(
        "--placed-leaf-exclusion-radius-m",
        type=float,
        default=0.15,
        help="Ignore bamboo detections this close to the visually detected ceramic center.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--approach-offset-m", type=float, default=0.14)
    parser.add_argument("--contact-offset-m", type=float, default=0.004)
    parser.add_argument("--lift-offset-m", type=float, default=0.18)
    parser.add_argument("--yaw-deg", type=float, default=90.0)
    parser.add_argument("--place-audit-site", default="ceramic_disc_center_site")
    parser.add_argument("--place-above-offset-m", type=float, default=0.12)
    parser.add_argument("--place-contact-offset-m", type=float, default=0.024)
    parser.add_argument("--place-yaw-deg", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=95.0)
    parser.add_argument("--settle-steps", type=int, default=300)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--camera-window", action="store_true")
    parser.add_argument("--camera-window-width", type=int, default=640)
    parser.add_argument("--camera-window-height", type=int, default=480)
    parser.add_argument("--camera-window-every-steps", type=int, default=20)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--hold-open", action="store_true")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene.expanduser().resolve()))
    data = mujoco.MjData(model)
    hide_sites(model)
    mujoco.mj_forward(model, data)
    viewer = mujoco.viewer.launch_passive(model, data) if args.viewer else None
    camera_preview = (
        CameraPreview(
            model=model,
            camera=args.camera,
            width=args.camera_window_width,
            height=args.camera_window_height,
            every_steps=args.camera_window_every_steps,
        )
        if args.camera_window
        else None
    )
    arm = SimXArmAPI(
        model,
        data,
        viewer=viewer,
        realtime=args.realtime or args.viewer or args.camera_window,
        sync_callback=camera_preview,
    )

    try:
        print(f"Loaded scene: {args.scene}")
        initialize_robot_home(arm)
        arm.step(args.settle_steps)

        first_frame_dir = capture_current_state(
            model,
            data,
            camera=args.camera,
            output_root=OUTPUT_ROOT,
            width=args.width,
            height=args.height,
        )
        print(f"Saved first simulation camera frame: {first_frame_dir}")

        first_vision_result = run_vision(
            frame_dir=first_frame_dir,
            text_prompt=args.text_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            vision_env=args.vision_env,
        )
        first_pick_detection = select_detection(
            first_vision_result.get("detections", []),
            label_contains=args.label_contains,
            min_score=args.min_score,
        )
        place_detection = select_detection(
            first_vision_result.get("detections", []),
            label_contains=args.place_label_contains,
            min_score=args.min_score,
        )
        place_point_world_m = np.array(place_detection["point_world_m"], dtype=float)
        first_pick_point_world_m = np.array(first_pick_detection["point_world_m"], dtype=float)
        annotated = load_annotated_image(first_frame_dir)
        if camera_preview is not None and annotated is not None:
            camera_preview.set_vision_result(
                annotated,
                [
                    ("PICK 1", first_pick_detection, (0, 0, 255)),
                    ("PLACE", place_detection, (255, 80, 0)),
                ],
            )
            for _ in range(30):
                camera_preview(model, data)

        ceramic_site_m = site_position(model, data, args.place_audit_site)
        ceramic_center_error_mm = float(
            np.linalg.norm(place_point_world_m[:2] - np.array(ceramic_site_m)[:2]) * 1000.0
        )
        print(f"First visual leaf: {first_pick_detection['label']} score={first_pick_detection['score']:.3f}")
        print(f"First pick point [m]: {first_pick_point_world_m.round(4).tolist()}")
        print(f"Selected place detection: {place_detection['label']} score={place_detection['score']:.3f}")
        print(f"Visual place point [m]: {place_point_world_m.round(4).tolist()}")
        print(
            "Place audit: target XY is from ceramic SAM2 mask in vision_result.json; "
            f"distance to ceramic audit site={ceramic_center_error_mm:.1f} mm"
        )

        first_motion = execute_pick_place(
            arm=arm,
            pick_point_world_m=first_pick_point_world_m,
            place_point_world_m=place_point_world_m,
            pick_yaw_deg=args.yaw_deg,
            place_yaw_deg=args.place_yaw_deg,
            approach_offset_m=args.approach_offset_m,
            contact_offset_m=args.contact_offset_m,
            lift_offset_m=args.lift_offset_m,
            place_above_offset_m=args.place_above_offset_m,
            place_contact_offset_m=args.place_contact_offset_m,
            speed=args.speed,
            cycle_name="leaf 1",
        )

        second_frame_dir = capture_current_state(
            model,
            data,
            camera=args.camera,
            output_root=OUTPUT_ROOT,
            width=args.width,
            height=args.height,
        )
        print(f"Saved second simulation camera frame: {second_frame_dir}")
        second_vision_result = run_vision(
            frame_dir=second_frame_dir,
            text_prompt="bamboo leaf.",
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            vision_env=args.vision_env,
        )
        second_pick_detection = select_detection(
            second_vision_result.get("detections", []),
            label_contains=args.label_contains,
            min_score=args.min_score,
            exclude_xy_m=place_point_world_m,
            min_xy_distance_m=args.placed_leaf_exclusion_radius_m,
        )
        second_pick_point_world_m = np.array(second_pick_detection["point_world_m"], dtype=float)
        print(f"Second visual leaf: {second_pick_detection['label']} score={second_pick_detection['score']:.3f}")
        print(f"Second pick point [m]: {second_pick_point_world_m.round(4).tolist()}")

        annotated = load_annotated_image(second_frame_dir)
        if camera_preview is not None and annotated is not None:
            camera_preview.set_vision_result(
                annotated,
                [
                    ("PICK 2", second_pick_detection, (0, 0, 255)),
                    ("PLACE", place_detection, (255, 80, 0)),
                ],
            )
            for _ in range(30):
                camera_preview(model, data)

        second_motion = execute_pick_place(
            arm=arm,
            pick_point_world_m=second_pick_point_world_m,
            place_point_world_m=place_point_world_m,
            pick_yaw_deg=args.yaw_deg,
            place_yaw_deg=args.place_yaw_deg + 90.0,
            approach_offset_m=args.approach_offset_m,
            contact_offset_m=args.contact_offset_m,
            lift_offset_m=args.lift_offset_m,
            place_above_offset_m=args.place_above_offset_m + 0.006,
            place_contact_offset_m=args.place_contact_offset_m + 0.006,
            speed=args.speed,
            cycle_name="leaf 2",
        )

        top_position_m, top_yaw_deg = body_xy_yaw(model, data, "bamboo_leaf_top")
        bottom_position_m, bottom_yaw_deg = body_xy_yaw(model, data, "bamboo_leaf_bottom")
        yaw_difference_deg = abs((top_yaw_deg - bottom_yaw_deg + 90.0) % 180.0 - 90.0)
        center_separation_mm = float(
            np.linalg.norm(np.asarray(top_position_m)[:2] - np.asarray(bottom_position_m)[:2]) * 1000.0
        )
        final_color_bgr, _ = sim_capture.render_color_and_depth(
            model=model,
            data=data,
            camera=args.camera,
            width=args.width,
            height=args.height,
        )
        final_scene_path = first_frame_dir / "final_scene.png"
        cv2.imwrite(str(final_scene_path), final_color_bgr)
        print(
            f"Final placement audit: center separation={center_separation_mm:.1f} mm, "
            f"leaf angle={yaw_difference_deg:.1f} deg"
        )

        diagnostics = {
            "frame_dirs": [str(first_frame_dir), str(second_frame_dir)],
            "selected_pick_detections": [first_pick_detection, second_pick_detection],
            "selected_place_detection": place_detection,
            "vision_audit": {
                "pick_coordinate_sources": [
                    str(first_frame_dir / "vision_result.json"),
                    str(second_frame_dir / "vision_result.json"),
                ],
                "first_pick_point_world_m": first_pick_point_world_m.tolist(),
                "second_pick_point_world_m": second_pick_point_world_m.tolist(),
                "place_point_world_m": place_point_world_m.tolist(),
                "ceramic_audit_site_m": ceramic_site_m,
                "place_xy_distance_to_ceramic_audit_site_mm": ceramic_center_error_mm,
                "minimum_detection_score": args.min_score,
                "used_site_for_pick": False,
                "used_site_for_place": False,
                "note": "Both picks and the common ceramic placement center come from vision. The site is audit-only.",
            },
            "motions": [first_motion, second_motion],
            "final_placement_audit": {
                "top_leaf_position_m": top_position_m,
                "bottom_leaf_position_m": bottom_position_m,
                "top_leaf_yaw_deg": top_yaw_deg,
                "bottom_leaf_yaw_deg": bottom_yaw_deg,
                "center_separation_mm": center_separation_mm,
                "cross_angle_deg": yaw_difference_deg,
                "final_scene": str(final_scene_path),
            },
            "final_tcp_pose_mm_deg": arm.get_position()[1],
        }
        output_path = first_frame_dir / "visual_cross_place_result.json"
        output_path.write_text(json.dumps(diagnostics, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        print(f"Saved visual cross-place result: {output_path}")

        if viewer is not None and args.hold_open:
            print("Two-leaf cross placement complete. Viewer stays open until closed or Ctrl+C.")
            while viewer.is_running():
                viewer.sync()
                if camera_preview is not None:
                    camera_preview(model, data)
        return 0
    finally:
        if camera_preview is not None:
            camera_preview.close()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    raise SystemExit(main())
