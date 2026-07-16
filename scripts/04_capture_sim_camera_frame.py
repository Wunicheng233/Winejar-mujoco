#!/usr/bin/env python3
"""Capture one RGB-D frame from an existing MuJoCo camera.

The output schema intentionally matches the RealSense capture script closely so
the same GroundingDINO + SAM2 offline pipeline can consume real and simulated
frames.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "scene" / "scene_winejar_vacuum.xml"
DEFAULT_CAMERA = "global_overhead_camera"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "sim_camera_frames"
DEPTH_SCALE_M_PER_UNIT = 0.001


def make_depth_preview(depth_raw: np.ndarray, depth_scale_m: float) -> np.ndarray:
    depth_m = depth_raw.astype(np.float32) * depth_scale_m
    valid = depth_m > 0
    normalized = np.zeros(depth_raw.shape, dtype=np.uint8)
    if np.any(valid):
        near_m, far_m = np.percentile(depth_m[valid], [2, 98])
        if far_m > near_m:
            clipped = np.clip(depth_m, near_m, far_m)
            normalized[valid] = np.round(
                255.0 * (far_m - clipped[valid]) / (far_m - near_m)
            ).astype(np.uint8)
    preview = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview


def intrinsics_from_fovy(width: int, height: int, fovy_deg: float) -> dict[str, Any]:
    fovy_rad = math.radians(float(fovy_deg))
    fy = 0.5 * float(height) / math.tan(0.5 * fovy_rad)
    fx = fy
    ppx = 0.5 * float(width)
    ppy = 0.5 * float(height)
    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "ppx": ppx,
        "ppy": ppy,
        "fovy_deg": float(fovy_deg),
        "distortion_model": "none",
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        "camera_matrix": [
            [fx, 0.0, ppx],
            [0.0, fy, ppy],
            [0.0, 0.0, 1.0],
        ],
    }


def opencv_camera_to_world(data: mujoco.MjData, cam_id: int) -> np.ndarray:
    mujoco_cam_to_world = data.cam_xmat[cam_id].reshape(3, 3)
    x_world = mujoco_cam_to_world[:, 0]
    y_world = mujoco_cam_to_world[:, 1]
    z_world = mujoco_cam_to_world[:, 2]

    # MuJoCo cameras look along local -Z with +Y up. The vision pipeline uses
    # OpenCV convention: +X right, +Y down, +Z forward.
    cv_to_world = np.column_stack([x_world, -y_world, -z_world])
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = cv_to_world
    transform[:3, 3] = data.cam_xpos[cam_id]
    return transform


def render_color_and_depth(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: str,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera)
        color_rgb = renderer.render()
        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=camera)
        depth_m = renderer.render()
        renderer.disable_depth_rendering()
    finally:
        renderer.close()

    depth_m = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    depth_raw = np.zeros(depth_m.shape, dtype=np.uint16)
    depth_mm = np.clip(depth_m[valid] / DEPTH_SCALE_M_PER_UNIT, 1, np.iinfo(np.uint16).max)
    depth_raw[valid] = np.round(depth_mm).astype(np.uint16)
    return color_bgr, depth_raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--show-sites",
        action="store_true",
        help="Render MuJoCo sites. Hidden by default to avoid confusing vision models.",
    )
    args = parser.parse_args()

    scene_path = args.scene.expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    if cam_id < 0:
        camera_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(model.ncam)
        ]
        raise RuntimeError(f"Camera {args.camera!r} not found; available: {camera_names}")

    if not args.show_sites:
        model.site_rgba[:, 3] = 0.0

    mujoco.mj_forward(model, data)
    for _ in range(max(0, args.settle_steps)):
        mujoco.mj_step(model, data)

    color_bgr, depth_raw = render_color_and_depth(
        model=model,
        data=data,
        camera=args.camera,
        width=args.width,
        height=args.height,
    )

    timestamp = datetime.now().astimezone()
    timestamp_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_dir = args.output_root.expanduser().resolve() / timestamp_id
    output_dir.mkdir(parents=True, exist_ok=False)

    paths = {
        "color_bgr": output_dir / "color.png",
        "depth_raw": output_dir / "depth_raw.png",
        "depth_preview": output_dir / "depth_preview.png",
        "metadata": output_dir / "frame.json",
    }
    if not cv2.imwrite(str(paths["color_bgr"]), color_bgr):
        raise RuntimeError("Failed to save color image")
    if not cv2.imwrite(str(paths["depth_raw"]), depth_raw):
        raise RuntimeError("Failed to save raw depth image")
    cv2.imwrite(str(paths["depth_preview"]), make_depth_preview(depth_raw, DEPTH_SCALE_M_PER_UNIT))

    valid_depth = depth_raw > 0
    cam_to_world = opencv_camera_to_world(data, cam_id)
    world_to_cam = np.linalg.inv(cam_to_world)
    metadata = {
        "schema_version": 1,
        "frame_id": timestamp_id,
        "captured_at": timestamp.isoformat(timespec="milliseconds"),
        "source": "mujoco",
        "scene": str(scene_path),
        "camera": {
            "name": args.camera,
            "id": int(cam_id),
            "fovy_deg": float(model.cam_fovy[cam_id]),
            "position_world_m": data.cam_xpos[cam_id].tolist(),
            "opencv_camera_to_world": cam_to_world.tolist(),
            "world_to_opencv_camera": world_to_cam.tolist(),
            "mujoco_camera_xmat": data.cam_xmat[cam_id].reshape(3, 3).tolist(),
        },
        "stream": {
            "width": args.width,
            "height": args.height,
            "fps": None,
            "color_format": "BGR8_PNG",
            "depth_format": "Z16_PNG",
            "depth_aligned_to": "color",
            "depth_scale_m_per_unit": DEPTH_SCALE_M_PER_UNIT,
            "coordinate_convention": "+x right, +y down, +z forward",
        },
        "intrinsics": intrinsics_from_fovy(args.width, args.height, model.cam_fovy[cam_id]),
        "quality": {
            "valid_depth_fraction": float(np.mean(valid_depth)),
            "minimum_valid_depth_m": (
                float(np.min(depth_raw[valid_depth]) * DEPTH_SCALE_M_PER_UNIT)
                if np.any(valid_depth)
                else None
            ),
            "maximum_valid_depth_m": (
                float(np.max(depth_raw[valid_depth]) * DEPTH_SCALE_M_PER_UNIT)
                if np.any(valid_depth)
                else None
            ),
        },
        "render": {
            "settle_steps": args.settle_steps,
            "sites_visible": bool(args.show_sites),
        },
        "files": {key: path.name for key, path in paths.items()},
    }
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Scene: {scene_path}")
    print(f"Camera: {args.camera}")
    print(f"Saved simulated RGB-D frame: {output_dir}")
    print(f"Color: {color_bgr.shape}, depth: {depth_raw.shape} ({depth_raw.dtype})")
    print(f"Valid depth: {metadata['quality']['valid_depth_fraction']:.1%}")
    print(f"Sites visible: {args.show_sites}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
