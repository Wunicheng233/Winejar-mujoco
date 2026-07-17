#!/usr/bin/env python3
"""Regression checks for layered lotus-leaf and paper folding."""
from __future__ import annotations

import mujoco
import numpy as np

from mujoco_xarm6.production_demo.constants import SCENE_PATH
from mujoco_xarm6.production_demo.cover_folds import CoverFoldController
from mujoco_xarm6.production_demo.line import (
    COVER_CLAMPED_ANGLE_RAD,
    COVER_PAPER_CLAMPED_ANGLE_RAD,
    COVER_PAPER_TIED_ANGLE_RAD,
    COVER_TIED_ANGLE_RAD,
)


def is_first_jar_cover(name: str) -> bool:
    return name.startswith("preloaded_lotus") or name.startswith("preloaded_paper")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    controller = CoverFoldController(model, data)
    controller.reset()

    controller.transition(
        1,
        COVER_CLAMPED_ANGLE_RAD,
        0.36,
        paper_angle=COVER_PAPER_CLAMPED_ANGLE_RAD,
    )
    while controller.transitions:
        controller.advance(model.opt.timestep)
    np.testing.assert_allclose(controller.angles(1)[:4], COVER_CLAMPED_ANGLE_RAD, atol=1e-9)
    np.testing.assert_allclose(controller.angles(1)[4:], COVER_PAPER_CLAMPED_ANGLE_RAD, atol=1e-9)

    controller.transition(
        1,
        COVER_TIED_ANGLE_RAD,
        0.10,
        paper_angle=COVER_PAPER_TIED_ANGLE_RAD,
    )
    elapsed = 0.0
    while controller.transitions:
        controller.advance(model.opt.timestep)
        elapsed += model.opt.timestep
    if elapsed > 0.102:
        raise AssertionError(f"Cover folding exceeded the tie transition: {elapsed:.4f} s")
    np.testing.assert_allclose(controller.angles(1)[:4], COVER_TIED_ANGLE_RAD, atol=1e-9)
    np.testing.assert_allclose(controller.angles(1)[4:], COVER_PAPER_TIED_ANGLE_RAD, atol=1e-9)
    mujoco.mj_forward(model, data)

    penetrations = []
    for contact in data.contact[: data.ncon]:
        first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
        second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
        cross_layer = (
            is_first_jar_cover(first)
            and is_first_jar_cover(second)
            and ("lotus" in first) != ("lotus" in second)
        )
        if cross_layer and contact.dist < -0.0005:
            penetrations.append(f"{first} <-> {second}: {contact.dist:.6f}")
    if penetrations:
        raise AssertionError(f"Folded cover layers must retain their physical ordering: {penetrations}")
    print("Layered cover fold controller checks OK")


if __name__ == "__main__":
    main()
