#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mujoco_xarm6.production_demo.line import ProductionLine, hide_debug_markers  # noqa: E402
from mujoco_xarm6.production_demo.clock import AnimationClock  # noqa: E402
from mujoco_xarm6.production_demo.constants import OUTPUT_ROOT, SCENE_PATH  # noqa: E402


DEFAULT_CAMERAS = [
    "global_overview_camera",
    "front_conveyor_camera",
    "side_overview_camera",
]
ALL_CAMERAS = DEFAULT_CAMERAS + [
    "global_top_camera",
    "production_hero_camera",
    "jar_mouth_closeup_camera",
    "left_vacuum_closeup_camera",
    "right_tie_gun_closeup_camera",
]


class FfmpegVideoSink:
    def __init__(self, path: Path, width: int, height: int, output_fps: int, crf: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.frame_count = 0
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(output_fps),
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                str(crf),
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray):
        if self.process.stdin is None:
            raise RuntimeError(f"ffmpeg stdin is closed for {self.path}")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.frame_count += 1

    def close(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path} with code {return_code}")


class MultiCameraRecorder:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        output_dir: Path,
        cameras: list[str],
        width: int,
        height: int,
        fps: int,
        playback_speed: float,
        crf: int,
    ):
        self.model = model
        self.data = data
        self.cameras = cameras
        self.output_fps = fps
        self.capture_fps = fps / max(playback_speed, 1e-6)
        self.playback_speed = playback_speed
        self.next_capture_time = 0.0
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.sinks = {
            camera: FfmpegVideoSink(output_dir / f"{camera}.mp4", width, height, fps, crf)
            for camera in cameras
        }

    def sync(self):
        if self.data.time + 1e-12 < self.next_capture_time:
            return
        for camera, sink in self.sinks.items():
            self.renderer.update_scene(self.data, camera=camera)
            sink.write(self.renderer.render())
        self.next_capture_time += 1.0 / self.capture_fps

    def close(self):
        try:
            self.renderer.close()
        finally:
            for sink in self.sinks.values():
                sink.close()

    def summary(self) -> dict[str, dict[str, object]]:
        return {
            camera: {
                "path": str(sink.path),
                "frames": sink.frame_count,
                "output_fps": self.output_fps,
                "capture_fps": self.capture_fps,
                "playback_speed": self.playback_speed,
            }
            for camera, sink in self.sinks.items()
        }


def write_png(path: Path, frame: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{frame.shape[1]}x{frame.shape[0]}",
            "-i",
            "-",
            "-frames:v",
            "1",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin is closed while writing preview still")
    process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed while writing preview still: {path}")


def save_preview_stills(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output_dir: Path,
    cameras: list[str],
    width: int,
    height: int,
) -> dict[str, Path]:
    renderer = mujoco.Renderer(model, height=height, width=width)
    preview_paths: dict[str, Path] = {}
    try:
        for camera in cameras:
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            path = output_dir / f"{camera}.png"
            write_png(path, frame)
            preview_paths[camera] = path
    finally:
        renderer.close()
    return preview_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Record multi-camera videos for the wine jar production demo.")
    parser.add_argument("--scene", type=Path, default=SCENE_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Output playback speed relative to simulated time; 0.4 records 2.5x more frames at the same output fps.",
    )
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--speed-scale", type=float, default=3.0)
    parser.add_argument("--camera", action="append", choices=ALL_CAMERAS, help="Record only selected camera(s).")
    parser.add_argument("--preview-stills", action="store_true", help="Render one still image per camera and exit.")
    parser.add_argument("--show-debug-markers", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cameras = args.camera or DEFAULT_CAMERAS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (OUTPUT_ROOT / "videos" / timestamp)

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    if not args.show_debug_markers:
        hide_debug_markers(model)
    mujoco.mj_resetData(model, data)

    if args.preview_stills:
        preview_paths = save_preview_stills(model, data, output_dir / "preview_stills", cameras, args.width, args.height)
        print(f"Saved preview stills to: {output_dir / 'preview_stills'}")
        for camera, path in preview_paths.items():
            print(f"  {camera}: {path}")
        return 0

    recorder = MultiCameraRecorder(
        model=model,
        data=data,
        output_dir=output_dir,
        cameras=cameras,
        width=args.width,
        height=args.height,
        fps=args.fps,
        playback_speed=args.playback_speed,
        crf=args.crf,
    )
    clock = AnimationClock(model, data, recorder, realtime=False, speed_scale=args.speed_scale)
    production_line = ProductionLine(model, data, clock)

    try:
        recorder.sync()
        production_line.run()
        recorder.sync()
    finally:
        recorder.close()

    print(f"Recorded videos to: {output_dir}")
    for camera, info in recorder.summary().items():
        print(
            f"  {camera}: {info['frames']} frames, "
            f"capture_fps={info['capture_fps']:.2f}, output_fps={info['output_fps']} -> {info['path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
