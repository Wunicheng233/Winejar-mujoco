"""Rigid leaf-root attachment and deliberately staged leaf profiles."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_xarm6.production_demo.clock import AnimationClock, smoothstep
from mujoco_xarm6.production_demo.scene_ops import freejoint_qpos_addr


@dataclass
class LeafProfileTransition:
    leaf: str
    start_profile: np.ndarray
    target_profile: np.ndarray
    duration_s: float
    elapsed: float = 0.0


@dataclass
class LeafRootAttachment:
    parent_body: str
    relative_pos: np.ndarray
    relative_quat: np.ndarray


class LeafAttachmentController:
    """Keep roots rigid while changing leaf shape only at named process stages."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, clock: AnimationClock):
        self.model = model
        self.data = data
        self.clock = clock
        self.attachments: dict[str, LeafRootAttachment] = {}
        self.profile_transitions: list[LeafProfileTransition] = []
        self.held_profiles: dict[str, np.ndarray] = {}

    def reset(self):
        self.attachments.clear()
        self.profile_transitions.clear()
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

    def set_profile(self, leaf: str, profile: np.ndarray):
        """Apply and hold a stable shape immediately at a process boundary."""
        self.profile_transitions[:] = [transition for transition in self.profile_transitions if transition.leaf != leaf]
        self.held_profiles[leaf] = profile.copy()
        self._refresh_profile_hold()

    def transition_profile(self, leaf: str, target_profile: np.ndarray, duration_s: float):
        """Smoothly change one held shape, for example during the tie operation."""
        if duration_s <= 0.0:
            self.set_profile(leaf, target_profile)
            return
        start_profile = self.held_profiles.get(leaf, self.bend_values(leaf)).copy()
        self.profile_transitions[:] = [transition for transition in self.profile_transitions if transition.leaf != leaf]
        self.profile_transitions.append(LeafProfileTransition(leaf, start_profile, target_profile.copy(), duration_s))

    def advance_profiles(self, dt: float) -> list[str]:
        completed: list[str] = []
        for transition in self.profile_transitions:
            transition.elapsed = min(transition.duration_s, transition.elapsed + dt)
            alpha = smoothstep(transition.elapsed / transition.duration_s)
            self.held_profiles[transition.leaf] = transition.start_profile + (transition.target_profile - transition.start_profile) * alpha
            if transition.elapsed >= transition.duration_s:
                completed.append(transition.leaf)
        if self.profile_transitions:
            self.profile_transitions[:] = [transition for transition in self.profile_transitions if transition.leaf not in completed]
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
        self.profile_transitions[:] = [transition for transition in self.profile_transitions if transition.leaf != leaf]
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
