#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer


ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "scene" / "scene_winejar.xml"


def main():
    parser = argparse.ArgumentParser(description="Open a MuJoCo winejar scene in the interactive viewer.")
    parser.add_argument(
        "--scene",
        type=Path,
        default=SCENE_PATH,
        help="Scene XML path. Defaults to the original winejar scene.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Optional keyframe name to apply before opening the viewer.",
    )
    args = parser.parse_args()
    scene_path = args.scene
    if not scene_path.is_absolute():
        scene_path = Path.cwd() / scene_path

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    if args.key:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.key)
        if key_id < 0:
            raise ValueError(f"Keyframe not found: {args.key}")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    print(f"Opening MuJoCo viewer for: {scene_path}")
    if args.key:
        print(f"Applied keyframe: {args.key}")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
