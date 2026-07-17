#!/usr/bin/env python3
"""Three-jar parallel production-line animation.

This module keeps the existing SDK-like inverse-kinematics controller, but
plans each arm's complete Cartesian-safe path before playback. A shared clock
then advances both joint trajectories in the same MuJoCo timestep, so the
loading and tying stations operate concurrently without waypoint pauses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from mujoco_xarm6.production_demo.attachments import LeafAttachmentController
from mujoco_xarm6.production_demo.clock import AnimationClock, smoothstep
from mujoco_xarm6.production_demo.constants import OUTPUT_ROOT, SCENE_PATH
from mujoco_xarm6.production_demo.cover_folds import CoverFoldController
from mujoco_xarm6.production_demo.models import JarSpec, JointPath, PathEvent
from mujoco_xarm6.production_demo.motion import make_left_production_arm, make_right_tie_arm
from mujoco_xarm6.production_demo.pathing import JointPathPlanner
from mujoco_xarm6.production_demo.scene_ops import body_id, site_pos
from mujoco_xarm6.production_demo.tie_press import TiePressController
from mujoco_xarm6.production_demo.tie_bands import TieBandController


LEFT_JOINTS = tuple(f"left_joint{i}" for i in range(1, 7))
RIGHT_JOINTS = tuple(f"right_joint{i}" for i in range(1, 7))
LEFT_HOME = np.radians([0.0, -35.0, -70.0, 0.0, 105.0, 0.0])
RIGHT_HOME = np.radians([0.0, -77.31, -69.98, 0.0, 147.29, 0.0])

ENTRY_X = -0.94
LOAD_X = -0.22
TIE_X = 0.40
EXIT_X = 0.94
BELT_MIN_X = -0.88
BELT_MAX_X = 0.88
BELT_LENGTH = BELT_MAX_X - BELT_MIN_X
INDEX_SECONDS = 1.6
TIE_HOLD_SECONDS = 0.5
LEAF_GATHER_TRANSITION_SECONDS = 0.30
COVER_CLAMPED_ANGLE_RAD = 0.85
COVER_TIED_ANGLE_RAD = 1.50
COVER_PAPER_CLAMPED_ANGLE_RAD = 0.75
COVER_PAPER_TIED_ANGLE_RAD = 1.45
TIE_RING_CLEARANCE_M = 0.027
TIE_PRESS_CONTACT_OFFSET_M = 0.047
PRESSED_FIRST_LEAF_HEIGHT_M = 0.0235
PRESSED_SECOND_LEAF_HEIGHT_M = 0.0285
TIE_PRESS_DESCENT_SECONDS = 0.36
# This is above the 0.62 m mouth-clearance envelope while remaining inside the
# xArm's verified reachable workspace over the material table.
SAFE_Z_M = 0.660
NATURAL_LEAF_PROFILE = np.array(
    [0.010, 0.015, 0.020, 0.025, 0.030, 0.030, 0.025, 0.020, 0.015, 0.010],
    dtype=np.float64,
)
CLAMPED_LEAF_PROFILE = np.array(
    [0.0, 0.0, -0.78, 0.78, 0.0, 0.0, 0.78, -0.78, 0.0, 0.0],
    dtype=np.float64,
)
GATHERED_LEAF_PROFILE = np.array(
    [0.0, 0.0, -1.48, 1.48, 0.0, 0.0, 1.48, -1.48, 0.0, 0.0],
    dtype=np.float64,
)
JARS = (
    JarSpec(
        1,
        "station_wine_jar",
        "jar_mouth_center",
        "neck_tie_target_site",
        "staged_bamboo_leaf_top",
        "staged_bamboo_leaf_bottom",
        "staged_bamboo_leaf_top_pick_site",
        "staged_bamboo_leaf_bottom_pick_site",
        "left_suction_weld_leaf_top",
        "left_suction_weld_leaf_bottom",
        "table_static_friction_weld_leaf_top",
        "table_static_friction_weld_leaf_bottom",
        "mouth_static_friction_weld_leaf_top",
        "mouth_static_friction_weld_leaf_bottom",
    ),
    *tuple(
        JarSpec(
            index,
            f"station_wine_jar_{index:02d}",
            f"jar_{index:02d}_mouth_center",
            f"jar_{index:02d}_neck_tie_target",
            f"jar_{index:02d}_bamboo_leaf_top",
            f"jar_{index:02d}_bamboo_leaf_bottom",
            f"jar_{index:02d}_bamboo_leaf_top_pick_site",
            f"jar_{index:02d}_bamboo_leaf_bottom_pick_site",
            f"left_suction_weld_jar_{index:02d}_leaf_top",
            f"left_suction_weld_jar_{index:02d}_leaf_bottom",
            f"table_static_friction_weld_jar_{index:02d}_leaf_top",
            f"table_static_friction_weld_jar_{index:02d}_leaf_bottom",
            f"mouth_static_friction_weld_jar_{index:02d}_leaf_top",
            f"mouth_static_friction_weld_jar_{index:02d}_leaf_bottom",
        )
        for index in (2, 3)
    ),
)


class ProductionLine:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, clock: AnimationClock):
        self.model = model
        self.data = data
        self.clock = clock
        self.left = make_left_production_arm(clock, hold_right_at_current=False)
        self.right = make_right_tie_arm(clock)
        self.left_addrs = np.array([model.joint(name).qposadr[0] for name in LEFT_JOINTS], dtype=int)
        self.right_addrs = np.array([model.joint(name).qposadr[0] for name in RIGHT_JOINTS], dtype=int)
        self.left_dofs = np.array([model.joint(name).dofadr[0] for name in LEFT_JOINTS], dtype=int)
        self.right_dofs = np.array([model.joint(name).dofadr[0] for name in RIGHT_JOINTS], dtype=int)
        self.actions: list[dict] = []
        self.exited_jars: set[int] = set()
        self.release_stack_gaps_mm: dict[str, float] = {}
        self.press_stack_gaps_mm: dict[str, float] = {}
        self.tie_leaf_contacts: set[str] = set()
        self.tie_leaf_penetrations: set[str] = set()
        self.tie_leaf_contact_samples: list[dict] = []
        self.tie_leaf_min_distance_m = 0.0
        self.tie_leaf_worst_contact: dict | None = None
        self.active_tie_leaf_geom_ids: set[int] = set()
        self.leaf_lotus_contacts: set[str] = set()
        self.paths = JointPathPlanner(model, data)
        self.leaves = LeafAttachmentController(model, data, clock)
        self.cover_folds = CoverFoldController(model, data)
        self.tie_press = TiePressController(model)
        self.tie_bands = TieBandController(model)
        self.tie_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)) and name.startswith("right_tie_gun_")
        }
        self.tie_press_geom_ids = {
            geom_id
            for geom_id in self.tie_geom_ids
            if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
            and (name.startswith("right_tie_gun_closed_jaw_") or name.startswith("right_tie_gun_loaded_band_") or name == "right_tie_gun_press_ball")
        }
        self.leaf_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)) and "bamboo_leaf" in name
        }
        self.lotus_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)) and "preloaded_lotus" in name
        }
        self.marker_starts = {
            geom_id: model.geom_pos[geom_id].copy()
            for geom_id in range(model.ngeom)
            if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)) and name.startswith("belt_marker_")
        }

    def reset(self):
        self.exited_jars.clear()
        self.release_stack_gaps_mm.clear()
        self.press_stack_gaps_mm.clear()
        self.tie_leaf_contacts.clear()
        self.tie_leaf_penetrations.clear()
        self.tie_leaf_contact_samples.clear()
        self.tie_leaf_min_distance_m = 0.0
        self.tie_leaf_worst_contact = None
        self.active_tie_leaf_geom_ids.clear()
        self.leaf_lotus_contacts.clear()
        self.leaves.reset()
        self.tie_press.reset()
        self.tie_bands.reset()
        mujoco.mj_resetData(self.model, self.data)
        self.clock.reset_timing()
        self.data.qpos[self.left_addrs] = LEFT_HOME
        self.data.qpos[self.right_addrs] = RIGHT_HOME
        self.cover_folds.reset()
        for jar in JARS:
            # Future workpieces remain outside the rendered factory until their
            # conveyor-index cycle starts.
            self.model.body_pos[body_id(self.model, jar.body)] = np.array([-2.4, 0.05, -3.0], dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        for jar in JARS:
            for weld in (jar.top_suction_weld, jar.bottom_suction_weld, jar.top_mouth_weld, jar.bottom_mouth_weld):
                self.data.eq_active[self.model.equality(weld).id] = 0
            for weld in (jar.top_table_weld, jar.bottom_table_weld):
                leaf = jar.top_leaf if weld == jar.top_table_weld else jar.bottom_leaf
                self.leaves.activate_weld(weld, "left_material_table", f"{leaf}_seg_05")
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def _attach(self, jar: JarSpec, leaf: str):
        is_top = leaf == jar.top_leaf
        table_weld = jar.top_table_weld if is_top else jar.bottom_table_weld
        suction_weld = jar.top_suction_weld if is_top else jar.bottom_suction_weld
        self.leaves.deactivate_weld(table_weld)
        self.leaves.attach_root(leaf, "left_vacuum_end_effector", suction_weld)
        self.leaves.set_profile(leaf, NATURAL_LEAF_PROFILE)
        self.actions.append({"label": f"jar {jar.index} attach {'top' if is_top else 'bottom'} leaf", "code": 0})

    def _release(self, jar: JarSpec, leaf: str):
        is_top = leaf == jar.top_leaf
        suction_weld = jar.top_suction_weld if is_top else jar.bottom_suction_weld
        mouth_weld = jar.top_mouth_weld if is_top else jar.bottom_mouth_weld
        self.leaves.deactivate_weld(suction_weld)
        mujoco.mj_forward(self.model, self.data)
        self.leaves.attach_root(leaf, jar.body, mouth_weld)
        if leaf == jar.bottom_leaf:
            self.release_stack_gaps_mm[str(jar.index)] = float(
                (self.leaves.leaf_center(jar.bottom_leaf)[2] - self.leaves.leaf_center(jar.top_leaf)[2]) * 1000.0
            )
        self.actions.append({"label": f"jar {jar.index} release {'top' if is_top else 'bottom'} leaf", "code": 0})

    def _gather_leaves(self, jar: JarSpec):
        for leaf in (jar.top_leaf, jar.bottom_leaf):
            self.leaves.transition_profile(leaf, GATHERED_LEAF_PROFILE, LEAF_GATHER_TRANSITION_SECONDS)
        self.cover_folds.transition(
            jar.index,
            COVER_TIED_ANGLE_RAD,
            LEAF_GATHER_TRANSITION_SECONDS,
            paper_angle=COVER_PAPER_TIED_ANGLE_RAD,
        )
        self.actions.append({"label": f"jar {jar.index} gather leaves", "code": 0, "duration_s": LEAF_GATHER_TRANSITION_SECONDS})

    def _press_leaves(self, jar: JarSpec, duration_s: float):
        """Compress both leaves during the final ring descent, without layer swaps."""
        mouth = site_pos(self.model, self.data, jar.mouth_site)
        stages = (
            (jar.top_leaf, PRESSED_FIRST_LEAF_HEIGHT_M),
            (jar.bottom_leaf, PRESSED_SECOND_LEAF_HEIGHT_M),
        )
        for leaf, height in stages:
            self.leaves.transition_profile(leaf, CLAMPED_LEAF_PROFILE, duration_s)
            self.leaves.transition_root_to_world(leaf, mouth + np.array([0.0, 0.0, height]), duration_s)
        self.cover_folds.transition(
            jar.index,
            COVER_CLAMPED_ANGLE_RAD,
            duration_s,
            paper_angle=COVER_PAPER_CLAMPED_ANGLE_RAD,
        )
        self.actions.append({"label": f"jar {jar.index} press leaves", "code": 0, "duration_s": duration_s})

    def _record_press_stack_gap(self, jar: JarSpec):
        self.press_stack_gaps_mm[str(jar.index)] = float(
            (self.leaves.leaf_center(jar.bottom_leaf)[2] - self.leaves.leaf_center(jar.top_leaf)[2]) * 1000.0
        )

    def _start_tie_contact_window(self, jar: JarSpec):
        self.active_tie_leaf_geom_ids = {
            self.model.geom(f"{leaf}_seg_{segment:02d}_geom").id
            for leaf in (jar.top_leaf, jar.bottom_leaf)
            for segment in range(11)
        }

    def _finish_tie_contact_window(self):
        self.active_tie_leaf_geom_ids.clear()

    def _leaf_job(self, jar: JarSpec) -> JointPath:
        start_q = self.data.qpos[self.left_addrs].copy()
        mouth = site_pos(self.model, self.data, jar.mouth_site)
        top_pick = site_pos(self.model, self.data, jar.top_pick_site)
        bottom_pick = site_pos(self.model, self.data, jar.bottom_pick_site)
        top_place = mouth + np.array([0.0, 0.0, 0.036])
        bottom_place = mouth + np.array([0.0, 0.0, 0.052])
        sequence: list[tuple[np.ndarray, float]] = []
        events: list[PathEvent] = []

        def append(point: np.ndarray, yaw: float, label: str | None = None, callback=None):
            sequence.append((point, yaw))
            if callback is not None:
                events.append(PathEvent(-1, label or "event", callback))

        # Above every material or jar operation, the tool stays on the safe layer.
        append(np.array([top_pick[0], top_pick[1], SAFE_Z_M]), 0.0)
        append(top_pick + np.array([0.0, 0.0, 0.130]), 0.0)
        append(top_pick + np.array([0.0, 0.0, -0.004]), 0.0, "attach top", lambda: self._attach(jar, jar.top_leaf))
        append(top_pick + np.array([0.0, 0.0, 0.175]), 0.0)
        append(np.array([mouth[0], mouth[1], SAFE_Z_M]), 0.0)
        append(top_place + np.array([0.0, 0.0, 0.030]), 0.0)
        append(top_place, 0.0, "release top", lambda: self._release(jar, jar.top_leaf))
        append(top_place + np.array([0.0, 0.0, 0.030]), 0.0)
        append(np.array([bottom_pick[0], bottom_pick[1], SAFE_Z_M]), 0.0)
        append(bottom_pick + np.array([0.0, 0.0, 0.130]), 0.0)
        append(bottom_pick + np.array([0.0, 0.0, -0.004]), 0.0, "attach bottom", lambda: self._attach(jar, jar.bottom_leaf))
        append(bottom_pick + np.array([0.0, 0.0, 0.175]), 0.0)
        append(np.array([mouth[0], mouth[1], SAFE_Z_M]), 90.0)
        append(bottom_place + np.array([0.0, 0.0, 0.025]), 90.0)
        append(bottom_place, 90.0, "release bottom", lambda: self._release(jar, jar.bottom_leaf))
        append(bottom_place + np.array([0.0, 0.0, 0.025]), 90.0)
        append(np.array([-0.22, 0.52, SAFE_Z_M]), 0.0)

        path = self.paths.build(f"left load jar {jar.index}", self.left, self.left_addrs, self.left_dofs, start_q, sequence, 11.0)
        # Build endpoint indices exactly from Cartesian subdivision counts.
        path.events = []
        event_ordinals = {2: events[0], 6: events[1], 10: events[2], 14: events[3]}
        point_index = 0
        previous_tcp = self.data.site_xpos[self.left.tcp_site_id].copy()
        previous_yaw = 0.0
        for index, (target, yaw) in enumerate(sequence):
            point_index += len(self.paths.cartesian_samples(previous_tcp, target, previous_yaw, yaw))
            if index in event_ordinals:
                event = event_ordinals[index]
                path.events.append(PathEvent(point_index, event.label, event.callback))
            previous_tcp = target
            previous_yaw = yaw
        return path

    def _tie_job(self, jar: JarSpec) -> JointPath:
        start_q = self.data.qpos[self.right_addrs].copy()
        neck = site_pos(self.model, self.data, jar.neck_site)
        ring_offset = site_pos(self.model, self.data, "right_tie_gun_center_site") - site_pos(self.model, self.data, "right_tie_gun_ring_visual_site")
        current_tcp = self.data.site_xpos[self.right.tcp_site_id].copy()
        current_safe = current_tcp.copy()
        current_safe[2] = SAFE_Z_M
        ring_at_neck = neck + np.array([0.0, 0.0, TIE_RING_CLEARANCE_M])
        # Keep the existing reachable safe layer. Only the final neck target is
        # raised, so the ring stays above the leaf stack without raising the
        # TCP beyond the right arm's verified workspace.
        ring_safe = neck + np.array([0.0, 0.0, 0.070])
        ring_contact = ring_at_neck + np.array([0.0, 0.0, TIE_PRESS_CONTACT_OFFSET_M])
        station_standby = np.array([TIE_X, -0.31, SAFE_Z_M])
        targets = [
            (current_safe, 90.0),
            (ring_safe + ring_offset, 90.0),
            (ring_contact + ring_offset, 90.0),
            (ring_at_neck + ring_offset, 90.0),
            (ring_safe + ring_offset, 90.0),
            (station_standby, 90.0),
        ]
        path = self.paths.build(f"right tie jar {jar.index}", self.right, self.right_addrs, self.right_dofs, start_q, targets, 11.0)
        # Work out exact target endpoint indices after Cartesian subdivision.
        point_index = 0
        previous_tcp = self.data.site_xpos[self.right.tcp_site_id].copy()
        previous_yaw = 0.0
        endpoint_indices: list[int] = []
        for target, yaw in targets:
            count = len(self.paths.cartesian_samples(previous_tcp, target, previous_yaw, yaw))
            point_index += count
            endpoint_indices.append(point_index)
            previous_tcp = target
            previous_yaw = yaw
        press_contact_point = endpoint_indices[2]
        press_point = endpoint_indices[3]
        raw_press_descent_seconds = float(sum(path.durations[press_contact_point:press_point]))
        if raw_press_descent_seconds <= 0.0:
            raise RuntimeError("Tie press descent has no motion duration")
        press_descent_seconds = max(raw_press_descent_seconds, TIE_PRESS_DESCENT_SECONDS)
        if press_descent_seconds > raw_press_descent_seconds:
            scale = press_descent_seconds / raw_press_descent_seconds
            for segment in range(press_contact_point, press_point):
                path.durations[segment] *= scale
        ascent_endpoint = endpoint_indices[4]
        path.points.insert(press_point + 1, path.points[press_point].copy())
        path.durations.insert(press_point, TIE_HOLD_SECONDS)
        retract_point = press_point + 1
        ascent_seconds = float(sum(path.durations[retract_point : ascent_endpoint + 1]))

        def start_press_descent():
            self._start_tie_contact_window(jar)
            self.tie_press.compress(press_descent_seconds)
            self._press_leaves(jar, press_descent_seconds)

        def start_tie_hold():
            self._record_press_stack_gap(jar)
            self._gather_leaves(jar)
            self.tie_bands.tighten(jar.index, LEAF_GATHER_TRANSITION_SECONDS)
            self.actions.append({"label": f"jar {jar.index} tie hold", "code": 0, "hold_seconds": TIE_HOLD_SECONDS})

        def start_tie_retract():
            self.tie_press.release(ascent_seconds)
            self.tie_bands.set_loaded_visible(True)

        path.events.append(PathEvent(press_contact_point, f"jar {jar.index} press descent", start_press_descent))
        path.events.append(PathEvent(press_point, f"jar {jar.index} tie hold", start_tie_hold))
        path.events.append(PathEvent(retract_point, f"jar {jar.index} spring release", start_tie_retract))
        path.events.append(PathEvent(ascent_endpoint + 1, f"jar {jar.index} clear tie contact window", self._finish_tie_contact_window))
        return path

    def _step(self, left_path: JointPath | None = None, right_path: JointPath | None = None):
        dt = self.model.opt.timestep
        # Advance transitions that were started by the previous path sample
        # before firing events at the next endpoint. This makes the clamped
        # leaf shape finish on the exact frame that the ring descent finishes,
        # and lets the tied profile begin on the following hold frame.
        self.tie_press.advance(dt)
        self.tie_bands.advance(dt)
        self.leaves.advance_transitions(dt)
        self.cover_folds.advance(dt)
        if left_path is not None:
            left_path.advance(self.data, dt)
        if right_path is not None:
            right_path.advance(self.data, dt)
        self._sync_materials()
        if right_path is not None:
            self._audit_tie_leaf_contacts()
        self.clock.step(1, after_step=self._sync_materials)

    def _sync_materials(self):
        self.leaves.sync_roots()
        self.cover_folds.apply(exclude=set(self.cover_folds.transitions))
        self.tie_bands.apply()
        mujoco.mj_forward(self.model, self.data)

    def _audit_tie_leaf_contacts(self):
        """Record every physical contact between the tie gun and leaf stack.

        The arms are kinematically replayed for animation, so contact forces do
        not automatically re-plan a commanded path.  Recording these contacts
        makes a collision visible in the saved diagnostics instead of hiding it
        behind a purely kinematic playback.
        """
        if not self.active_tie_leaf_geom_ids:
            return
        for contact in self.data.contact[: self.data.ncon]:
            first_id, second_id = contact.geom1, contact.geom2
            if (
                ((first_id in self.leaf_geom_ids and second_id in self.lotus_geom_ids) or (second_id in self.leaf_geom_ids and first_id in self.lotus_geom_ids))
                and float(contact.dist) < -0.0005
            ):
                first = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, first_id) or ""
                second = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, second_id) or ""
                self.leaf_lotus_contacts.add(" <-> ".join((first, second)))
            if (first_id in self.tie_press_geom_ids and second_id in self.active_tie_leaf_geom_ids) or (second_id in self.tie_press_geom_ids and first_id in self.active_tie_leaf_geom_ids):
                first = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, first_id) or ""
                second = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, second_id) or ""
                names = (first, second)
                pair = " <-> ".join(names)
                if pair not in self.tie_leaf_contacts and len(self.tie_leaf_contact_samples) < 12:
                    self.tie_leaf_contact_samples.append(
                        {
                            "time_s": round(float(self.data.time), 4),
                            "pair": pair,
                            "ring_pos_m": self.data.site_xpos[self.model.site("right_tie_gun_ring_visual_site").id].round(5).tolist(),
                        }
                    )
                self.tie_leaf_contacts.add(pair)
                if float(contact.dist) < self.tie_leaf_min_distance_m:
                    self.tie_leaf_min_distance_m = float(contact.dist)
                    self.tie_leaf_worst_contact = {
                        "time_s": round(float(self.data.time), 4),
                        "pair": pair,
                        "distance_m": float(contact.dist),
                        "ring_pos_m": self.data.site_xpos[
                            self.model.site("right_tie_gun_ring_visual_site").id
                        ].round(5).tolist(),
                        "spring_compression_m": self.tie_press.compression_m,
                    }
                if float(contact.dist) < -0.0005:
                    self.tie_leaf_penetrations.add(pair)

    def _run_paths(self, left_path: JointPath | None, right_path: JointPath | None):
        while (left_path is not None and not left_path.complete) or (right_path is not None and not right_path.complete):
            self._step(left_path, right_path)
        for path in (left_path, right_path):
            if path is not None:
                self.actions.append({"label": path.name, "code": 0, "duration_s": path.duration, "intermediate_stop_count": 0})

    def _belt_x(self, x: float) -> float:
        while x > BELT_MAX_X:
            x -= BELT_LENGTH
        while x < BELT_MIN_X:
            x += BELT_LENGTH
        return x

    def _move_jars(self, moves: dict[int, tuple[float, float]], label: str):
        for index, (from_x, _to_x) in moves.items():
            body = body_id(self.model, JARS[index - 1].body)
            if self.model.body_pos[body][2] < 0.0:
                self.model.body_pos[body] = np.array([from_x, 0.05, 0.125], dtype=np.float64)
            elif abs(float(self.model.body_pos[body][0]) - from_x) > 0.01:
                self.model.body_pos[body][0] = from_x
        mujoco.mj_forward(self.model, self.data)
        starts = {index: self.model.body_pos[body_id(self.model, JARS[index - 1].body)].copy() for index in moves}
        marker_start = {geom_id: self.model.geom_pos[geom_id].copy() for geom_id in self.marker_starts}
        steps = max(1, int(INDEX_SECONDS / self.model.opt.timestep))
        for step in range(steps):
            alpha = smoothstep((step + 1) / steps)
            displacement = 0.0
            for index, (_from_x, to_x) in moves.items():
                pos = starts[index].copy()
                pos[0] = starts[index][0] + (to_x - starts[index][0]) * alpha
                displacement = float(pos[0] - starts[index][0])
                self.model.body_pos[body_id(self.model, JARS[index - 1].body)] = pos
            for geom_id, start_pos in marker_start.items():
                pos = start_pos.copy()
                pos[0] = self._belt_x(float(start_pos[0] + displacement))
                self.model.geom_pos[geom_id] = pos
            self._step()
        for index, (_from_x, to_x) in moves.items():
            if abs(to_x - EXIT_X) < 1e-9:
                self._hide_exited_jar(JARS[index - 1])
        self.actions.append({"label": label, "code": 0, "duration_s": INDEX_SECONDS, "belt_to_jar_speed_ratio": 1.0})

    def _hide_exited_jar(self, jar: JarSpec):
        """Remove a completed workpiece and its attached leaves beyond the outfeed."""
        self.model.body_pos[body_id(self.model, jar.body)] = np.array([2.4, 0.05, -3.0], dtype=np.float64)
        for leaf, mouth_weld in ((jar.top_leaf, jar.top_mouth_weld), (jar.bottom_leaf, jar.bottom_mouth_weld)):
            self.leaves.deactivate_weld(mouth_weld)
            self.leaves.forget(leaf)
            joint = self.model.joint(f"{leaf}_freejoint")
            qposadr = int(joint.qposadr[0])
            dofadr = int(joint.dofadr[0])
            self.data.qpos[qposadr : qposadr + 3] = np.array([2.4, 0.05, -3.0], dtype=np.float64)
            self.data.qvel[dofadr : dofadr + 6] = 0.0
        self.exited_jars.add(jar.index)
        mujoco.mj_forward(self.model, self.data)

    def run(self):
        self.reset()
        self._move_jars({1: (ENTRY_X, LOAD_X)}, "index jar 1 to loading station")
        self._run_paths(self._leaf_job(JARS[0]), None)
        self._move_jars({1: (LOAD_X, TIE_X), 2: (ENTRY_X, LOAD_X)}, "index jars 1 and 2")
        self._run_paths(self._leaf_job(JARS[1]), self._tie_job(JARS[0]))
        self._move_jars({1: (TIE_X, EXIT_X), 2: (LOAD_X, TIE_X), 3: (ENTRY_X, LOAD_X)}, "index jars 1 out, 2 and 3 forward")
        self._run_paths(self._leaf_job(JARS[2]), self._tie_job(JARS[1]))
        self._move_jars({2: (TIE_X, EXIT_X), 3: (LOAD_X, TIE_X)}, "index jars 2 out and 3 forward")
        self._run_paths(None, self._tie_job(JARS[2]))
        self._move_jars({3: (TIE_X, EXIT_X)}, "index jar 3 out")
        return self.diagnostics()

    def diagnostics(self) -> dict:
        jar_x = {
            str(jar.index): None if jar.index in self.exited_jars else float(self.model.body_pos[body_id(self.model, jar.body)][0])
            for jar in JARS
        }
        leaf_heights = {
            str(jar.index): {
                "top_z_m": float(self.leaves.leaf_center(jar.top_leaf)[2]),
                "bottom_z_m": float(self.leaves.leaf_center(jar.bottom_leaf)[2]),
            }
            for jar in JARS
        }
        return {
            "flow": "three_jar_parallel_loading_and_tying",
            "jar_x_m": jar_x,
            "exited_jars": sorted(self.exited_jars),
            "leaf_heights_m": leaf_heights,
            "release_stack_gaps_mm": self.release_stack_gaps_mm,
            "press_stack_gaps_mm": self.press_stack_gaps_mm,
            "tie_leaf_contact_pairs": sorted(self.tie_leaf_contacts),
            "tie_leaf_penetration_pairs": sorted(self.tie_leaf_penetrations),
            "tie_leaf_contact_samples": self.tie_leaf_contact_samples,
            "tie_leaf_min_distance_m": self.tie_leaf_min_distance_m,
            "tie_leaf_worst_contact": self.tie_leaf_worst_contact,
            "final_tie_band_radii_m": {str(index): radius for index, radius in self.tie_bands.radii_m.items()},
            "leaf_lotus_contact_pairs": sorted(self.leaf_lotus_contacts),
            "cover_fold_angles_rad": {
                str(jar.index): self.cover_folds.angles(jar.index).round(5).tolist() for jar in JARS
            },
            "actions": self.actions,
            "parallel_stations": [
                {"left": "load jar 2", "right": "tie jar 1"},
                {"left": "load jar 3", "right": "tie jar 2"},
            ],
            "label_paper_in_flow": False,
            "tie_station_x_m": TIE_X,
            "load_station_x_m": LOAD_X,
            "robot_paths": "shared-clock continuous joint interpolation",
        }


def hide_debug_markers(model: mujoco.MjModel):
    model.site_rgba[:, 3] = 0.0
    for material_name in ("target_marker_mat",):
        material_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, material_name)
        if material_id >= 0:
            model.mat_rgba[material_id, 3] = 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Run the three-jar wine production-line simulation.")
    parser.add_argument("--scene", type=Path, default=SCENE_PATH)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--playback-speed", type=float, default=1.0, help="Real-time playback multiplier; 1.0 is normal speed.")
    parser.add_argument("--hold-open", action="store_true")
    parser.add_argument("--show-debug-markers", action="store_true")
    parser.add_argument("--quiet-diagnostics", action="store_true")
    return parser.parse_args()


def _run(model, data, viewer, args):
    if args.playback_speed <= 0.0:
        raise ValueError("--playback-speed must be positive")
    if not args.show_debug_markers:
        hide_debug_markers(model)
    clock = AnimationClock(model, data, viewer=viewer, realtime=args.realtime, speed_scale=args.playback_speed)
    diagnostics = ProductionLine(model, data, clock).run()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / "latest_result.json"
    output_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.quiet_diagnostics:
        print(f"Completed three-jar line: jar positions={diagnostics['jar_x_m']}")
        print(f"Saved diagnostics: {output_path}")
    if viewer is not None and args.hold_open:
        while viewer.is_running():
            clock.step(1)
    return 0


def main() -> int:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            return _run(model, data, viewer, args)
    return _run(model, data, None, args)


if __name__ == "__main__":
    raise SystemExit(main())
