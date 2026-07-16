#!/usr/bin/env python3
"""Regression checks for the three-jar parallel production animation."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SCRIPT = ROOT / "scripts" / "02_production_demo_animation.py"
RESULT = ROOT / "data" / "production_demo_animation" / "latest_result.json"
SCENE = ROOT / "scene" / "scene_winejar_production_demo.xml"


def run_animation():
    subprocess.run([sys.executable, str(SCRIPT), "--quiet-diagnostics"], cwd=REPO, check=True)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    for jar_name in ("station_wine_jar", "station_wine_jar_02", "station_wine_jar_03"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, jar_name) < 0:
            raise AssertionError(f"Missing production-line jar body: {jar_name}")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "staged_label_paper") >= 0:
        raise AssertionError("Label paper must be removed from the revised production flow")
    for camera in ("global_overview_camera", "front_conveyor_camera", "side_overview_camera"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
            raise AssertionError(f"Missing recording camera: {camera}")

    run_animation()
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    if result.get("flow") != "three_jar_parallel_loading_and_tying":
        raise AssertionError(f"Unexpected production flow: {result.get('flow')}")
    if result.get("label_paper_in_flow") is not False:
        raise AssertionError("The revised line must not load the removed paper label")
    if result["tie_station_x_m"] - result["load_station_x_m"] < 0.45:
        raise AssertionError("Tie-gun station should be visibly downstream of the loading station")
    if result.get("exited_jars") != [1, 2, 3]:
        raise AssertionError(f"Completed jars must disappear at the outfeed: {result.get('exited_jars')}")
    if any(x is not None for x in result["jar_x_m"].values()):
        raise AssertionError(f"Completed jars should be hidden after outfeed: {result['jar_x_m']}")

    labels = [entry["label"] for entry in result["actions"]]
    for index in (1, 2, 3):
        for event in ("attach top leaf", "release top leaf", "attach bottom leaf", "release bottom leaf", "tie hold"):
            expected = f"jar {index} {event}"
            if expected not in labels:
                raise AssertionError(f"Missing production event: {expected}")
    for pair in result["parallel_stations"]:
        if not pair["left"].startswith("load jar") or not pair["right"].startswith("tie jar"):
            raise AssertionError(f"Invalid parallel work pair: {pair}")

    for entry in result["actions"]:
        if entry.get("belt_to_jar_speed_ratio") is not None and abs(entry["belt_to_jar_speed_ratio"] - 1.0) > 0.02:
            raise AssertionError(f"Conveyor marker speed mismatch: {entry}")
        if entry.get("intermediate_stop_count") not in (None, 0):
            raise AssertionError(f"Robot trajectory contains visual waypoint stops: {entry}")
    holds = [entry for entry in result["actions"] if entry["label"].endswith("tie hold")]
    if any(abs(entry["hold_seconds"] - 1.0) > 0.02 for entry in holds):
        raise AssertionError(f"Tie-gun hold should be one second: {holds}")

    stack_gaps = result.get("release_stack_gaps_mm", {})
    if sorted(stack_gaps) != ["1", "2", "3"]:
        raise AssertionError(f"Missing release stack diagnostics: {stack_gaps}")
    for index, gap_mm in stack_gaps.items():
        if not 4.0 <= gap_mm <= 35.0:
            raise AssertionError(f"Jar {index} leaf stack has an unsafe visual gap/intersection: {gap_mm:.1f} mm")
    print("Three-jar parallel production animation checks OK")


if __name__ == "__main__":
    main()
