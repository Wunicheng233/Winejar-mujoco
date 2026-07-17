#!/usr/bin/env python3
"""Regression checks for real-time viewer pacing without physics decimation."""
from __future__ import annotations

import mujoco

from mujoco_xarm6.production_demo.clock import AnimationClock


class CountingViewer:
    def __init__(self):
        self.sync_count = 0

    def sync(self):
        self.sync_count += 1


def model_and_data():
    model = mujoco.MjModel.from_xml_string('<mujoco><option timestep="0.001"/><worldbody/></mujoco>')
    return model, mujoco.MjData(model)


def main() -> None:
    model, data = model_and_data()
    realtime_viewer = CountingViewer()
    realtime_clock = AnimationClock(
        model,
        data,
        viewer=realtime_viewer,
        realtime=True,
        speed_scale=1_000_000.0,
        realtime_render_hz=60.0,
    )
    realtime_clock.step(20)
    if realtime_viewer.sync_count != 2:
        raise AssertionError(f"Expected two 60 FPS viewer syncs over 20 ms, got {realtime_viewer.sync_count}")

    model, data = model_and_data()
    recorder = CountingViewer()
    offline_clock = AnimationClock(model, data, viewer=recorder, realtime=False, speed_scale=1.0)
    offline_clock.step(20)
    if recorder.sync_count != 20:
        raise AssertionError(f"Offline recorder must receive every physics step, got {recorder.sync_count}")
    print("Animation clock pacing checks OK")


if __name__ == "__main__":
    main()
