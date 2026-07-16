"""Rigid leaf-root attachment and controlled flexible placement."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_xarm6.production_demo.clock import AnimationClock, smoothstep
from mujoco_xarm6.production_demo.models import JarSpec
from mujoco_xarm6.production_demo.scene_ops import freejoint_qpos_addr


@dataclass
class LeafPlaceTransition:
    jar: JarSpec
    leaf: str
    center_in_jar: np.ndarray
    start_bends: np.ndarray
    elapsed: float = 0.0


@dataclass
class LeafRootAttachment:
    parent_body: str
    relative_pos: np.ndarray
    relative_quat: np.ndarray


class LeafAttachmentController:
    """Own the boundary between rigid handling and flexible leaf motion.

    The free-joint root follows either the suction tool or a jar exactly. The
    ten bend joints remain independent MuJoCo degrees of freedom.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, clock: AnimationClock):
        self.model = model
        self.data = data
        self.clock = clock
        self.attachments: dict[str, LeafRootAttachment] = {}
        self.pending_placements: list[LeafPlaceTransition] = []
        self.held_profiles: dict[str, np.ndarray] = {}

    def reset(self):
        self.attachments.clear()
        self.pending_placements.clear()
        self.held_profiles.clear()
        self.clock.clear_joint_hold()

    def activate_weld(self, weld: str, parent_body: str, child_body: str):
        eq_id = self.model.equality(weld).id
        parent_id = self.model.body(parent_body).id
        child_id = self.model.body(child_body).id
        parent_pos = self.data.xpos[parent_id]
        child_pos = self.data.xpos[child_id]
        parent_rot = self.data.xmat[parent_id].reshape(3, 3)
        child_rot = self.data.xmat[child_id].reshape(3, 3)
        relative_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(relative_quat, (parent_rot.T @ child_rot).reshape(-1))
        self.model.eq_data[eq_id, 3:6] = parent_rot.T @ (child_pos - parent_pos)
        self.model.eq_data[eq_id, 6:10] = relative_quat
        self.model.eq_data[eq_id, 10] = 1.0
        self.data.eq_active[eq_id] = 1

    def deactivate_weld(self, weld: str):
        self.data.eq_active[self.model.equality(weld).id] = 0

    def attach_root(self, leaf: str, parent_body: str, weld: str):
        self.activate_weld(weld, parent_body, f"{leaf}_seg_05")
        self._capture_attachment(leaf, parent_body)

    def begin_placement(self, jar: JarSpec, leaf: str, mouth_weld: str):
        self.attach_root(leaf, jar.body, mouth_weld)
        jar_id = self.model.body(jar.body).id
        jar_pos = self.data.xpos[jar_id]
        jar_rot = self.data.xmat[jar_id].reshape(3, 3)
        self.pending_placements.append(
            LeafPlaceTransition(
                jar=jar,
                leaf=leaf,
                center_in_jar=jar_rot.T @ (self.leaf_center(leaf) - jar_pos),
                start_bends=self.bend_values(leaf),
            )
        )

    def advance_placements(self, dt: float, target_profile: np.ndarray, duration_s: float) -> list[LeafPlaceTransition]:
        completed: list[LeafPlaceTransition] = []
        for transition in self.pending_placements:
            transition.elapsed = min(duration_s, transition.elapsed + dt)
            alpha = smoothstep(transition.elapsed / duration_s)
            self.set_bend_values(transition.leaf, transition.start_bends + (target_profile - transition.start_bends) * alpha)
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            jar_id = self.model.body(transition.jar.body).id
            jar_pos = self.data.xpos[jar_id]
            jar_rot = self.data.xmat[jar_id].reshape(3, 3)
            root_addr = freejoint_qpos_addr(self.model, f"{transition.leaf}_freejoint")
            target_center = jar_pos + jar_rot @ transition.center_in_jar
            self.data.qpos[root_addr : root_addr + 3] += target_center - self.leaf_center(transition.leaf)
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self._capture_attachment(transition.leaf, transition.jar.body)
            if transition.elapsed >= duration_s:
                completed.append(transition)
        for transition in completed:
            self.pending_placements.remove(transition)
            self.held_profiles[transition.leaf] = target_profile.copy()
        if completed:
            self._refresh_profile_hold()
        return completed

    def sync_roots(self):
        for leaf, attachment in self.attachments.items():
            parent_id = self.model.body(attachment.parent_body).id
            parent_pos = self.data.xpos[parent_id]
            parent_rot = self.data.xmat[parent_id].reshape(3, 3)
            relative_rot = np.zeros(9, dtype=np.float64)
            mujoco.mju_quat2Mat(relative_rot, attachment.relative_quat)
            root_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(root_quat, (parent_rot @ relative_rot.reshape(3, 3)).reshape(-1))
            joint = self.model.joint(f"{leaf}_freejoint")
            qpos_addr = int(joint.qposadr[0])
            dof_addr = int(joint.dofadr[0])
            self.data.qpos[qpos_addr : qpos_addr + 3] = parent_pos + parent_rot @ attachment.relative_pos
            self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = root_quat
            self.data.qvel[dof_addr : dof_addr + 6] = 0.0
        if self.attachments:
            mujoco.mj_forward(self.model, self.data)

    def forget(self, leaf: str):
        self.attachments.pop(leaf, None)
        self.held_profiles.pop(leaf, None)
        self.pending_placements[:] = [transition for transition in self.pending_placements if transition.leaf != leaf]
        self._refresh_profile_hold()

    def leaf_center(self, leaf: str) -> np.ndarray:
        return self.data.site_xpos[self.model.site(f"{leaf}_center_site").id].copy()

    def bend_values(self, leaf: str) -> np.ndarray:
        return np.asarray(
            [self.data.qpos[self.model.joint(f"{leaf}_bend_{segment:02d}").qposadr[0]] for segment in range(1, 11)],
            dtype=np.float64,
        )

    def set_bend_values(self, leaf: str, values: np.ndarray):
        for segment, value in enumerate(values, start=1):
            self.data.qpos[self.model.joint(f"{leaf}_bend_{segment:02d}").qposadr[0]] = value

    def _capture_attachment(self, leaf: str, parent_body: str):
        parent_id = self.model.body(parent_body).id
        parent_pos = self.data.xpos[parent_id]
        parent_rot = self.data.xmat[parent_id].reshape(3, 3)
        root_addr = freejoint_qpos_addr(self.model, f"{leaf}_freejoint")
        root_pos = self.data.qpos[root_addr : root_addr + 3]
        root_quat = self.data.qpos[root_addr + 3 : root_addr + 7]
        root_rot = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(root_rot, root_quat)
        relative_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(relative_quat, (parent_rot.T @ root_rot.reshape(3, 3)).reshape(-1))
        self.attachments[leaf] = LeafRootAttachment(
            parent_body=parent_body,
            relative_pos=parent_rot.T @ (root_pos - parent_pos),
            relative_quat=relative_quat,
        )

    def _refresh_profile_hold(self):
        if not self.held_profiles:
            self.clock.clear_joint_hold()
            return
        qpos_addrs: list[int] = []
        dof_addrs: list[int] = []
        values: list[float] = []
        for leaf, profile in self.held_profiles.items():
            for segment, bend in enumerate(profile, start=1):
                joint = self.model.joint(f"{leaf}_bend_{segment:02d}")
                qpos_addrs.append(int(joint.qposadr[0]))
                dof_addrs.append(int(joint.dofadr[0]))
                values.append(float(bend))
        self.clock.hold_joints(
            np.asarray(qpos_addrs, dtype=int),
            np.asarray(values, dtype=np.float64),
            np.asarray(dof_addrs, dtype=int),
        )
