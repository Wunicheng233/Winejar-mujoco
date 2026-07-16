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
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from mujoco_xarm6.production_demo.clock import AnimationClock, smoothstep
from mujoco_xarm6.production_demo.constants import OUTPUT_ROOT, REPO_ROOT, SCENE_PATH
from mujoco_xarm6.production_demo.motion import make_left_production_arm, make_right_tie_arm
from mujoco_xarm6.production_demo.scene_ops import body_id, freejoint_qpos_addr, site_pos, yaw_to_quat
from mujoco_xarm6.sim.xarm_sim_api import rpy_to_mat


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
INDEX_SECONDS = 2.2
TIE_HOLD_SECONDS = 1.0
TIE_GATHER_SECONDS = 0.70
# This is above the 0.62 m mouth-clearance envelope while remaining inside the
# xArm's verified reachable workspace over the material table.
SAFE_Z_M = 0.660
LEAF_PROFILE = np.array(
    [-0.0309, -0.3253, 0.0026, 0.3692, 0.0581, 0.0039, -0.0807, 0.3311, -0.1450, -0.0060],
    dtype=np.float64,
)
# The tie-gun works after the two leaves have been gathered around the neck.
# This stronger profile folds their free ends down instead of leaving a flat
# sheet across the path of the collidable circular jaws.
TIE_LEAF_PROFILE = np.clip(LEAF_PROFILE * 2.45, -1.10, 1.10)


@dataclass(frozen=True)
class JarSpec:
    index: int
    body: str
    mouth_site: str
    neck_site: str
    top_leaf: str
    bottom_leaf: str
    top_pick_site: str
    bottom_pick_site: str
    top_suction_weld: str
    bottom_suction_weld: str
    top_table_weld: str
    bottom_table_weld: str
    top_mouth_weld: str
    bottom_mouth_weld: str


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


@dataclass
class PathEvent:
    point_index: int
    label: str
    callback: object


@dataclass
class LeafGatherTransition:
    jar: JarSpec
    start_bends: dict[str, np.ndarray]
    centers: dict[str, np.ndarray]
    elapsed: float = 0.0


@dataclass
class JointPath:
    name: str
    joint_addrs: np.ndarray
    dof_addrs: np.ndarray
    points: list[np.ndarray]
    durations: list[float]
    events: list[PathEvent] = field(default_factory=list)
    elapsed: float = 0.0
    fired: set[int] = field(default_factory=set)
    complete: bool = False

    @property
    def duration(self) -> float:
        return float(sum(self.durations))

    def advance(self, data: mujoco.MjData, dt: float):
        if self.complete:
            return
        previous = self.elapsed
        self.elapsed = min(self.duration, self.elapsed + dt)
        cumulative = 0.0
        for segment, duration in enumerate(self.durations):
            next_cumulative = cumulative + duration
            if self.elapsed <= next_cumulative or segment == len(self.durations) - 1:
                alpha = (self.elapsed - cumulative) / max(duration, 1e-9)
                # Linear velocity through intermediate samples avoids visible stops.
                data.qpos[self.joint_addrs] = self.points[segment] + (self.points[segment + 1] - self.points[segment]) * alpha
                data.qvel[self.dof_addrs] = 0.0
                break
            cumulative = next_cumulative
        cumulative = 0.0
        for point_index, duration in enumerate(self.durations, start=1):
            cumulative += duration
            if point_index not in self.fired and previous < cumulative <= self.elapsed + 1e-10:
                self.fired.add(point_index)
                for event in self.events:
                    if event.point_index == point_index:
                        event.callback()
        if self.elapsed >= self.duration - 1e-10:
            data.qpos[self.joint_addrs] = self.points[-1]
            data.qvel[self.dof_addrs] = 0.0
            self.complete = True


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
        self.tie_leaf_contacts: set[str] = set()
        self.tie_leaf_penetrations: set[str] = set()
        self.tie_leaf_contact_samples: list[dict] = []
        self.active_tie_gather: LeafGatherTransition | None = None
        self.tie_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)) and name.startswith("right_tie_gun_")
        }
        self.leaf_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)) and "bamboo_leaf" in name
        }
        self.marker_starts = {
            geom_id: model.geom_pos[geom_id].copy()
            for geom_id in range(model.ngeom)
            if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)) and name.startswith("belt_marker_")
        }

    def reset(self):
        self.exited_jars.clear()
        self.release_stack_gaps_mm.clear()
        self.tie_leaf_contacts.clear()
        self.tie_leaf_penetrations.clear()
        self.tie_leaf_contact_samples.clear()
        self.active_tie_gather = None
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.left_addrs] = LEFT_HOME
        self.data.qpos[self.right_addrs] = RIGHT_HOME
        for jar in JARS:
            # Future workpieces remain outside the rendered factory until their
            # conveyor-index cycle starts.
            self.model.body_pos[body_id(self.model, jar.body)] = np.array([-2.4, 0.05, -3.0], dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        for jar in JARS:
            for weld in (jar.top_suction_weld, jar.bottom_suction_weld, jar.top_mouth_weld, jar.bottom_mouth_weld):
                self.data.eq_active[self.model.equality(weld).id] = 0
            for weld in (jar.top_table_weld, jar.bottom_table_weld):
                self._activate_weld(weld, "left_material_table", f"{jar.top_leaf if weld == jar.top_table_weld else jar.bottom_leaf}_seg_05")
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def _activate_weld(self, equality: str, parent_body: str, child_body: str):
        eq_id = self.model.equality(equality).id
        parent_id = self.model.body(parent_body).id
        child_id = self.model.body(child_body).id
        parent_pos = self.data.xpos[parent_id].copy()
        child_pos = self.data.xpos[child_id].copy()
        parent_rot = self.data.xmat[parent_id].reshape(3, 3).copy()
        child_rot = self.data.xmat[child_id].reshape(3, 3).copy()
        relative_pos = parent_rot.T @ (child_pos - parent_pos)
        relative_rot = parent_rot.T @ child_rot
        relative_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(relative_quat, relative_rot.reshape(-1))
        self.model.eq_data[eq_id, 3:6] = relative_pos
        self.model.eq_data[eq_id, 6:10] = relative_quat
        self.model.eq_data[eq_id, 10] = 1.0
        self.data.eq_active[eq_id] = 1

    def _leaf_center(self, leaf: str) -> np.ndarray:
        return self.data.site_xpos[self.model.site(f"{leaf}_center_site").id].copy()

    def _shape_leaf(self, leaf: str, mouth_weld: str, jar_body: str, profile: np.ndarray = LEAF_PROFILE):
        center_before = self._leaf_center(leaf)
        for segment, bend in enumerate(profile, start=1):
            self.data.qpos[self.model.joint(f"{leaf}_bend_{segment:02d}").qposadr[0]] = bend
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        center_after = self._leaf_center(leaf)
        address = freejoint_qpos_addr(self.model, f"{leaf}_freejoint")
        self.data.qpos[address : address + 3] += center_before - center_after
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self._activate_weld(mouth_weld, jar_body, f"{leaf}_seg_05")

    def _attach(self, jar: JarSpec, leaf: str):
        is_top = leaf == jar.top_leaf
        table_weld = jar.top_table_weld if is_top else jar.bottom_table_weld
        suction_weld = jar.top_suction_weld if is_top else jar.bottom_suction_weld
        self.data.eq_active[self.model.equality(table_weld).id] = 0
        self._activate_weld(suction_weld, "left_vacuum_end_effector", f"{leaf}_seg_05")
        self.actions.append({"label": f"jar {jar.index} attach {'top' if is_top else 'bottom'} leaf", "code": 0})

    def _release(self, jar: JarSpec, leaf: str):
        is_top = leaf == jar.top_leaf
        suction_weld = jar.top_suction_weld if is_top else jar.bottom_suction_weld
        mouth_weld = jar.top_mouth_weld if is_top else jar.bottom_mouth_weld
        self.data.eq_active[self.model.equality(suction_weld).id] = 0
        mujoco.mj_forward(self.model, self.data)
        self._shape_leaf(leaf, mouth_weld, jar.body)
        if not is_top:
            self.release_stack_gaps_mm[str(jar.index)] = float(
                (self._leaf_center(jar.bottom_leaf)[2] - self._leaf_center(jar.top_leaf)[2]) * 1000.0
            )
        self.actions.append({"label": f"jar {jar.index} release {'top' if is_top else 'bottom'} leaf", "code": 0})

    def _start_tie_gather(self, jar: JarSpec):
        """Begin a visible, time-based leaf gathering transition at the neck."""
        leaves = (jar.top_leaf, jar.bottom_leaf)
        self.active_tie_gather = LeafGatherTransition(
            jar=jar,
            start_bends={
                leaf: np.array(
                    [self.data.qpos[self.model.joint(f"{leaf}_bend_{segment:02d}").qposadr[0]] for segment in range(1, 11)],
                    dtype=np.float64,
                )
                for leaf in leaves
            },
            centers={leaf: self._leaf_center(leaf) for leaf in leaves},
        )
        self.actions.append({"label": f"jar {jar.index} start leaf gathering", "code": 0, "duration_s": TIE_GATHER_SECONDS})

    def _advance_tie_gather(self, dt: float):
        transition = self.active_tie_gather
        if transition is None:
            return
        transition.elapsed = min(TIE_GATHER_SECONDS, transition.elapsed + dt)
        alpha = smoothstep(transition.elapsed / TIE_GATHER_SECONDS)
        jar = transition.jar
        for leaf, mouth_weld in ((jar.top_leaf, jar.top_mouth_weld), (jar.bottom_leaf, jar.bottom_mouth_weld)):
            target = TIE_LEAF_PROFILE
            start = transition.start_bends[leaf]
            for segment, bend in enumerate(start + (target - start) * alpha, start=1):
                self.data.qpos[self.model.joint(f"{leaf}_bend_{segment:02d}").qposadr[0]] = bend
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            qposadr = freejoint_qpos_addr(self.model, f"{leaf}_freejoint")
            self.data.qpos[qposadr : qposadr + 3] += transition.centers[leaf] - self._leaf_center(leaf)
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self._activate_weld(mouth_weld, jar.body, f"{leaf}_seg_05")
        if transition.elapsed >= TIE_GATHER_SECONDS:
            self.actions.append({"label": f"jar {jar.index} complete leaf gathering", "code": 0})
            self.active_tie_gather = None

    @staticmethod
    def _pose(point_m: np.ndarray, yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
        return point_m, rpy_to_mat(180.0, 0.0, yaw_deg)

    def _solve(self, arm, joint_addrs: np.ndarray, seed_q: np.ndarray, point_m: np.ndarray, yaw_deg: float) -> np.ndarray:
        original = self.data.qpos[joint_addrs].copy()
        self.data.qpos[joint_addrs] = seed_q
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        position, rotation = self._pose(point_m, yaw_deg)
        solved = arm._solve_ik(position, rotation)
        self.data.qpos[joint_addrs] = original
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        if solved is None:
            raise RuntimeError(f"IK failed for {arm.tcp_site_name} at {point_m.tolist()}, yaw={yaw_deg}")
        return solved

    def _cartesian_points(self, start: np.ndarray, target: np.ndarray, start_yaw: float, target_yaw: float) -> list[tuple[np.ndarray, float]]:
        distance = float(np.linalg.norm(target - start))
        yaw_delta = abs(((target_yaw - start_yaw + 180.0) % 360.0) - 180.0)
        count = max(1, int(math.ceil(distance / 0.075)), int(math.ceil(yaw_delta / 22.5)))
        result = []
        for index in range(1, count + 1):
            alpha = index / count
            point = start + (target - start) * alpha
            yaw = start_yaw + (((target_yaw - start_yaw + 180.0) % 360.0) - 180.0) * alpha
            result.append((point, yaw))
        return result

    def _build_path(self, name: str, arm, joint_addrs: np.ndarray, dof_addrs: np.ndarray, start_q: np.ndarray, targets: list[tuple[np.ndarray, float]], speed: float, events: list[PathEvent]) -> JointPath:
        points = [start_q.copy()]
        current_q = start_q.copy()
        previous_tcp = self.data.site_xpos[arm.tcp_site_id].copy()
        previous_yaw = 0.0
        for target, yaw in targets:
            for point, point_yaw in self._cartesian_points(previous_tcp, target, previous_yaw, yaw):
                current_q = self._solve(arm, joint_addrs, current_q, point, point_yaw)
                points.append(current_q)
            previous_tcp = target.copy()
            previous_yaw = yaw
        durations = [max(0.025, float(np.max(np.abs(b - a))) / max(speed, 1e-6)) for a, b in zip(points, points[1:])]
        return JointPath(name, joint_addrs, dof_addrs, points, durations, events)

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

        path = self._build_path(f"left load jar {jar.index}", self.left, self.left_addrs, self.left_dofs, start_q, sequence, 7.5, [])
        # Build endpoint indices exactly from Cartesian subdivision counts.
        path.events = []
        event_ordinals = {2: events[0], 6: events[1], 10: events[2], 14: events[3]}
        point_index = 0
        previous_tcp = self.data.site_xpos[self.left.tcp_site_id].copy()
        previous_yaw = 0.0
        for index, (target, yaw) in enumerate(sequence):
            point_index += len(self._cartesian_points(previous_tcp, target, previous_yaw, yaw))
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
        # The physical jaws are collidable.  Keep their horizontal ring below
        # the mouth materials, around the bottle neck, rather than driving it
        # through the loose leaf edges.  The upper point is a vertical-only
        # approach/retract clearance for the same ring center.
        ring_neck = neck + np.array([0.0, 0.0, 0.006])
        # The lateral entry occurs well above the 420 mm leaf planform.  Only
        # after it is centered over the neck may the ring descend vertically.
        ring_transit = neck + np.array([0.0, 0.0, 0.070])
        targets = [
            (np.array([TIE_X, -0.31, SAFE_Z_M]), 90.0),
            (np.array([TIE_X, -0.31, SAFE_Z_M + 0.040]), 90.0),
            (ring_transit + ring_offset, 90.0),
            (ring_neck + ring_offset, 90.0),
            (ring_neck + ring_offset, 90.0),
            (ring_transit + ring_offset, 90.0),
            (np.array([TIE_X, -0.31, SAFE_Z_M + 0.040]), 90.0),
            (np.array([TIE_X, -0.31, SAFE_Z_M]), 90.0),
        ]
        path = self._build_path(f"right tie jar {jar.index}", self.right, self.right_addrs, self.right_dofs, start_q, targets, 7.0, [])
        # Work out exact target endpoint indices after Cartesian subdivision.
        point_index = 0
        previous_tcp = self.data.site_xpos[self.right.tcp_site_id].copy()
        previous_yaw = 0.0
        endpoint_indices: list[int] = []
        for target, yaw in targets:
            count = len(self._cartesian_points(previous_tcp, target, previous_yaw, yaw))
            point_index += count
            endpoint_indices.append(point_index)
            previous_tcp = target
            previous_yaw = yaw
        transit_point = endpoint_indices[2]
        # Pause above the jar while the flexible leaves visibly gather.  The
        # following neck descent is therefore not a sudden state change.
        path.points.insert(transit_point + 1, path.points[transit_point].copy())
        path.durations.insert(transit_point, TIE_GATHER_SECONDS)
        path.events.append(PathEvent(transit_point, f"jar {jar.index} start leaf gathering", lambda: self._start_tie_gather(jar)))

        neck_point = endpoint_indices[3] + 1
        path.points.insert(neck_point + 1, path.points[neck_point].copy())
        path.durations.insert(neck_point, TIE_HOLD_SECONDS)
        path.events.append(PathEvent(neck_point, f"jar {jar.index} tie hold", lambda: self.actions.append({"label": f"jar {jar.index} tie hold", "code": 0, "hold_seconds": TIE_HOLD_SECONDS})))
        return path

    def _step(self, left_path: JointPath | None = None, right_path: JointPath | None = None, update=None):
        dt = self.model.opt.timestep
        if left_path is not None:
            left_path.advance(self.data, dt)
        if right_path is not None:
            right_path.advance(self.data, dt)
        self._advance_tie_gather(dt)
        mujoco.mj_forward(self.model, self.data)
        if right_path is not None:
            self._audit_tie_leaf_contacts()
        self.clock.step(1)

    def _audit_tie_leaf_contacts(self):
        """Record every physical contact between the tie gun and leaf stack.

        The arms are kinematically replayed for animation, so contact forces do
        not automatically re-plan a commanded path.  Recording these contacts
        makes a collision visible in the saved diagnostics instead of hiding it
        behind a purely kinematic playback.
        """
        for contact in self.data.contact[: self.data.ncon]:
            first_id, second_id = contact.geom1, contact.geom2
            if (first_id in self.tie_geom_ids and second_id in self.leaf_geom_ids) or (second_id in self.tie_geom_ids and first_id in self.leaf_geom_ids):
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
            self.data.eq_active[self.model.equality(mouth_weld).id] = 0
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
                "top_z_m": float(self._leaf_center(jar.top_leaf)[2]),
                "bottom_z_m": float(self._leaf_center(jar.bottom_leaf)[2]),
            }
            for jar in JARS
        }
        return {
            "flow": "three_jar_parallel_loading_and_tying",
            "jar_x_m": jar_x,
            "exited_jars": sorted(self.exited_jars),
            "leaf_heights_m": leaf_heights,
            "release_stack_gaps_mm": self.release_stack_gaps_mm,
            "tie_leaf_contact_pairs": sorted(self.tie_leaf_contacts),
            "tie_leaf_penetration_pairs": sorted(self.tie_leaf_penetrations),
            "tie_leaf_contact_samples": self.tie_leaf_contact_samples,
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
    parser.add_argument("--hold-open", action="store_true")
    parser.add_argument("--show-debug-markers", action="store_true")
    parser.add_argument("--quiet-diagnostics", action="store_true")
    return parser.parse_args()


def _run(model, data, viewer, args):
    if not args.show_debug_markers:
        hide_debug_markers(model)
    clock = AnimationClock(model, data, viewer=viewer, realtime=args.realtime, speed_scale=1.0)
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
