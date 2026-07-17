"""Kinematic spring-and-ball press synchronized with the tie-gun descent."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_xarm6.production_demo.clock import smoothstep


@dataclass
class CompressionTransition:
    start_m: float
    target_m: float
    duration_s: float
    elapsed_s: float = 0.0


class TiePressController:
    """Animate the compliant press without giving up the ball collision volume.

    The tie arm is replayed kinematically, so contact forces cannot physically
    compress a spring. This controller explicitly represents that one passive
    degree of freedom: the ball rises relative to the ring while it presses the
    leaf stack, and the four visible spring capsules shorten with it.
    """

    BALL_NAME = "right_tie_gun_press_ball"
    BALL_CENTER_SITE = "right_tie_gun_press_ball_center_site"
    BALL_CONTACT_SITE = "right_tie_gun_press_contact_site"
    BODY_NAME = "right_tie_gun_body"
    SPRING_NAMES = tuple(f"right_tie_gun_press_spring_{index}" for index in range(4))
    MAX_COMPRESSION_M = 0.018

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.ball_id = model.geom(self.BALL_NAME).id
        self.ball_center_site_id = model.site(self.BALL_CENTER_SITE).id
        self.ball_contact_site_id = model.site(self.BALL_CONTACT_SITE).id
        self.spring_ids = [model.geom(name).id for name in self.SPRING_NAMES]
        body = model.geom(self.BODY_NAME)
        self.body_bottom_z = float(body.pos[2] + body.size[2])
        self.ball_rest_pos = model.geom_pos[self.ball_id].copy()
        self.ball_radius = float(model.geom_size[self.ball_id][0])
        self.rest_site_positions = {
            self.ball_center_site_id: model.site_pos[self.ball_center_site_id].copy(),
            self.ball_contact_site_id: model.site_pos[self.ball_contact_site_id].copy(),
        }
        self.compression_m = 0.0
        self.transition: CompressionTransition | None = None
        self.reset()

    def reset(self):
        self.transition = None
        self.compression_m = 0.0
        self._apply()

    def compress(self, duration_s: float):
        self._transition_to(self.MAX_COMPRESSION_M, duration_s)

    def release(self, duration_s: float):
        self._transition_to(0.0, duration_s)

    def advance(self, dt: float):
        if self.transition is None:
            return
        transition = self.transition
        transition.elapsed_s = min(transition.duration_s, transition.elapsed_s + dt)
        alpha = smoothstep(transition.elapsed_s / transition.duration_s)
        self.compression_m = transition.start_m + (transition.target_m - transition.start_m) * alpha
        self._apply()
        if transition.elapsed_s >= transition.duration_s:
            self.transition = None

    def _transition_to(self, target_m: float, duration_s: float):
        target_m = float(np.clip(target_m, 0.0, self.MAX_COMPRESSION_M))
        if duration_s <= 0.0:
            self.compression_m = target_m
            self.transition = None
            self._apply()
            return
        self.transition = CompressionTransition(self.compression_m, target_m, duration_s)

    def _apply(self):
        # Local +Z points downward in the installed tie-gun orientation. Moving
        # the ball toward local -Z therefore raises it in the world frame.
        ball_pos = self.ball_rest_pos.copy()
        ball_pos[2] -= self.compression_m
        self.model.geom_pos[self.ball_id] = ball_pos
        self.model.site_pos[self.ball_center_site_id] = ball_pos
        self.model.site_pos[self.ball_contact_site_id] = ball_pos + np.array([0.0, 0.0, self.ball_radius])

        ball_top_z = float(ball_pos[2] - self.ball_radius)
        end_z = max(self.body_bottom_z + 0.002, ball_top_z)
        points = np.array(
            [
                [-0.010, 0.0, self.body_bottom_z],
                [0.010, 0.0, 0.0],
                [-0.010, 0.0, 0.0],
                [0.010, 0.0, 0.0],
                [0.000, 0.0, end_z],
            ],
            dtype=np.float64,
        )
        points[1:4, 2] = np.linspace(self.body_bottom_z, end_z, 5)[1:4]
        for geom_id, start, end in zip(self.spring_ids, points[:-1], points[1:]):
            self._set_capsule(geom_id, start, end)

    def _set_capsule(self, geom_id: int, start: np.ndarray, end: np.ndarray):
        direction = end - start
        length = float(np.linalg.norm(direction))
        self.model.geom_pos[geom_id] = (start + end) * 0.5
        self.model.geom_size[geom_id][1] = length * 0.5
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_quatZ2Vec(quat, direction / max(length, 1e-9))
        self.model.geom_quat[geom_id] = quat
