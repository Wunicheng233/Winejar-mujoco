from __future__ import annotations

import math
import time

import mujoco
import numpy as np


def rx(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def ry(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rz(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def rpy_to_mat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    return rz(math.radians(yaw_deg)) @ ry(math.radians(pitch_deg)) @ rx(math.radians(roll_deg))


def mat_to_rpy_deg(rotation: np.ndarray) -> tuple[float, float, float]:
    sy = -rotation[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = math.asin(sy)
    if abs(math.cos(pitch)) > 1e-6:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    return 0.5 * (
        np.cross(current[:, 0], target[:, 0])
        + np.cross(current[:, 1], target[:, 1])
        + np.cross(current[:, 2], target[:, 2])
    )


class SimXArmAPI:
    """Small SDK-like adapter for controlling the MuJoCo xArm scene."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        tcp_site: str = "link_tcp",
        suction_site: str = "vacuum_suction_site",
        suction_do: int = 9,
        suction_threshold_m: float = 0.010,
        joint_names: list[str] | None = None,
        actuator_names: list[str] | None = None,
        vacuum_body: str = "vacuum_end_effector",
        suction_targets: list[tuple[str, str, str] | tuple[str, str, str, str]] | None = None,
        hold_joint_names: list[str] | None = None,
        hold_joint_values: np.ndarray | None = None,
        viewer=None,
        realtime: bool = False,
        sync_callback=None,
    ):
        self.model = model
        self.data = data
        self.tcp_site_name = tcp_site
        self.tcp_site_id = model.site(tcp_site).id
        self.suction_site_name = suction_site
        self.suction_site_id = model.site(suction_site).id
        self.suction_do = suction_do
        self.suction_threshold_m = suction_threshold_m
        self.vacuum_body = vacuum_body
        self.viewer = viewer
        self.realtime = realtime
        self.sync_callback = sync_callback
        self.vacuum_on = False
        self.attached_body_name: str | None = None
        self.last_suction_distance_m: float | None = None
        self._joint_names = joint_names or [f"joint{i}" for i in range(1, 7)]
        self._actuator_names = actuator_names or [f"vel{i}" for i in range(1, 7)]
        self._qposadr = np.array([model.joint(name).qposadr[0] for name in self._joint_names], dtype=int)
        self._dofadr = np.array([model.joint(name).dofadr[0] for name in self._joint_names], dtype=int)
        self._ctrladr = np.array([model.actuator(name).id for name in self._actuator_names], dtype=int)
        self._suction_targets = suction_targets or [
            ("bamboo_leaf_top", "bamboo_leaf_top_pick_site", "suction_weld_top"),
            ("bamboo_leaf_bottom", "bamboo_leaf_bottom_pick_site", "suction_weld_bottom"),
        ]
        self._hold_qposadr = (
            np.array([model.joint(name).qposadr[0] for name in hold_joint_names], dtype=int)
            if hold_joint_names is not None
            else None
        )
        self._hold_dofadr = (
            np.array([model.joint(name).dofadr[0] for name in hold_joint_names], dtype=int)
            if hold_joint_names is not None
            else None
        )
        self._hold_values = np.array(hold_joint_values, dtype=np.float64) if hold_joint_values is not None else None
        self.last_motion_diagnostics: dict = {}

    def get_position(self, is_radian: bool = False):
        mujoco.mj_forward(self.model, self.data)
        pos_mm = self.data.site_xpos[self.tcp_site_id] * 1000.0
        rot = self.data.site_xmat[self.tcp_site_id].reshape(3, 3)
        roll, pitch, yaw = mat_to_rpy_deg(rot)
        if is_radian:
            roll, pitch, yaw = math.radians(roll), math.radians(pitch), math.radians(yaw)
        return 0, [float(pos_mm[0]), float(pos_mm[1]), float(pos_mm[2]), roll, pitch, yaw]

    def set_servo_angle(self, angle, is_radian: bool = False, wait: bool = True, speed: float = 1.0, **_kwargs):
        target = np.array(angle[:6], dtype=np.float64)
        if not is_radian:
            target = np.radians(target)
        return self._drive_joints(target, speed=max(speed, 0.05), wait=wait)

    def set_position(
        self,
        x=None,
        y=None,
        z=None,
        roll=None,
        pitch=None,
        yaw=None,
        speed: float = 80.0,
        mvacc: float | None = None,
        wait: bool = True,
        is_radian: bool = False,
        timeout: float = 8.0,
        joint_tolerance: float = 0.03,
        **_kwargs,
    ):
        del mvacc
        code, current = self.get_position(is_radian=False)
        if code != 0:
            return code
        target_pose = list(current)
        for idx, value in enumerate([x, y, z, roll, pitch, yaw]):
            if value is not None:
                target_pose[idx] = float(value)
        if is_radian:
            target_pose[3:] = [math.degrees(v) for v in target_pose[3:]]

        target_pos_m = np.array(target_pose[:3], dtype=np.float64) / 1000.0
        target_rot = rpy_to_mat(*target_pose[3:])
        q_start = self.data.qpos.copy()
        qvel_start = self.data.qvel.copy()
        target_q = self._solve_ik(target_pos_m, target_rot)
        self.data.qpos[:] = q_start
        self.data.qvel[:] = qvel_start
        mujoco.mj_forward(self.model, self.data)
        if target_q is None:
            self.last_motion_diagnostics = {
                "status": "ik_failed",
                "target_pose_mm_deg": target_pose,
                "current_pose_mm_deg": current,
            }
            return -8
        joint_speed = float(np.clip(speed / 100.0, 0.15, 14.0))
        return self._drive_joints(
            target_q,
            speed=joint_speed,
            wait=wait,
            timeout=timeout,
            joint_tolerance=joint_tolerance,
        )

    def set_cgpio_digital(self, ionum: int, value: int, *_args, **_kwargs):
        if ionum != self.suction_do:
            return 0
        self.vacuum_on = bool(value)
        if self.vacuum_on:
            self._try_attach_nearest_leaf()
        else:
            self._release()
        return 0

    def set_cgpio_digital_output_function(self, *_args, **_kwargs):
        return 0

    def step(self, steps: int = 1):
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            self._apply_joint_hold()
            self._sync_viewer()

    def site_position(self, site_name: str) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.model.site(site_name).id].copy()

    def _solve_ik(self, target_pos_m: np.ndarray, target_rot: np.ndarray) -> np.ndarray | None:
        damping = 0.04
        max_step = 0.10
        base_q = self.data.qpos[self._qposadr].copy()
        seed_offsets = [0.0, 0.45, -0.45, 0.90, -0.90, 1.35, -1.35]
        for wrist_offset in seed_offsets:
            q = base_q.copy()
            q[-1] += wrist_offset
            for _ in range(420):
                self.data.qpos[self._qposadr] = q
                self.data.qvel[:] = 0
                mujoco.mj_forward(self.model, self.data)
                current_pos = self.data.site_xpos[self.tcp_site_id].copy()
                current_rot = self.data.site_xmat[self.tcp_site_id].reshape(3, 3).copy()
                err = np.concatenate(
                    [
                        target_pos_m - current_pos,
                        0.35 * orientation_error(current_rot, target_rot),
                    ]
                )
                if np.linalg.norm(err[:3]) < 0.003 and np.linalg.norm(err[3:]) < 0.02:
                    return q.copy()
                jacp = np.zeros((3, self.model.nv), dtype=np.float64)
                jacr = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.tcp_site_id)
                jac = np.vstack([jacp[:, self._dofadr], 0.35 * jacr[:, self._dofadr]])
                lhs = jac @ jac.T + damping**2 * np.eye(6)
                dq = jac.T @ np.linalg.solve(lhs, err)
                q += np.clip(dq, -max_step, max_step)
        return None

    def _drive_joints(
        self,
        target_q: np.ndarray,
        speed: float,
        wait: bool,
        timeout: float = 8.0,
        joint_tolerance: float = 0.03,
    ):
        if not wait:
            self.data.ctrl[self._ctrladr] = np.clip((target_q - self.data.qpos[self._qposadr]) * 4.0, -speed, speed)
            return 0
        start_q = self.data.qpos[self._qposadr].copy()
        delta_q = target_q - start_q
        max_delta = float(np.max(np.abs(delta_q)))
        planned_seconds = max_delta / max(speed, 1e-6)
        steps = min(
            max(1, int(timeout / self.model.opt.timestep)),
            max(6, int(planned_seconds / self.model.opt.timestep)),
        )
        for step_index in range(steps):
            alpha = (step_index + 1) / steps
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            self.data.qpos[self._qposadr] = start_q + delta_q * alpha
            self.data.qvel[self._dofadr] = delta_q / max(planned_seconds, self.model.opt.timestep)
            self.data.ctrl[self._ctrladr] = 0.0
            mujoco.mj_step(self.model, self.data)
            self._apply_joint_hold()
            self._sync_viewer()
        self.data.qpos[self._qposadr] = target_q
        self.data.qvel[self._dofadr] = 0
        mujoco.mj_forward(self.model, self.data)
        final_err = target_q - self.data.qpos[self._qposadr]
        if np.max(np.abs(final_err)) < joint_tolerance:
            self.data.ctrl[self._ctrladr] = 0
            self.last_motion_diagnostics = {
                "status": "reached",
                "trajectory_mode": "time_parameterized",
                "max_joint_error_rad": float(np.max(np.abs(final_err))),
                "elapsed_sim_seconds": float(steps * self.model.opt.timestep),
                "commanded_joint_speed_rad_s": float(speed),
                "planned_joint_motion_seconds": float(planned_seconds),
            }
            return 0
        self.data.ctrl[self._ctrladr] = 0
        self.last_motion_diagnostics = {
            "status": "timeout",
            "max_joint_error_rad": float(np.max(np.abs(final_err))),
            "joint_error_rad": final_err.tolist(),
            "target_q_rad": target_q.tolist(),
            "actual_q_rad": self.data.qpos[self._qposadr].copy().tolist(),
            "elapsed_sim_seconds": float(steps * self.model.opt.timestep),
            "commanded_joint_speed_rad_s": float(speed),
        }
        return 100

    def drive_joint_waypoints_blended(
        self,
        joint_waypoints: list[np.ndarray],
        speed: float,
        timeout_per_waypoint: float,
        joint_tolerance: float = 0.03,
        blend_tolerance: float = 0.08,
    ) -> int:
        del blend_tolerance
        if not joint_waypoints:
            self.last_motion_diagnostics = {
                "status": "empty_path",
                "path_mode": "blended_joint_waypoints",
                "waypoint_count": 0,
            }
            return -8

        start_q = self.data.qpos[self._qposadr].copy()
        waypoints = [start_q] + [np.array(waypoint, dtype=np.float64).copy() for waypoint in joint_waypoints]
        segment_deltas = [float(np.max(np.abs(waypoints[i + 1] - waypoints[i]))) for i in range(len(joint_waypoints))]
        segment_durations = [
            min(timeout_per_waypoint, max(6 * self.model.opt.timestep, delta / max(speed, 1e-6)))
            for delta in segment_deltas
        ]
        cumulative = np.concatenate([[0.0], np.cumsum(segment_durations)])
        planned_seconds = float(cumulative[-1])
        total_steps = max(1, int(planned_seconds / self.model.opt.timestep))
        for step_index in range(total_steps):
            t = min(planned_seconds, (step_index + 1) * self.model.opt.timestep)
            segment_index = min(len(joint_waypoints) - 1, int(np.searchsorted(cumulative, t, side="right") - 1))
            segment_start_t = float(cumulative[segment_index])
            segment_duration = max(float(segment_durations[segment_index]), self.model.opt.timestep)
            local_alpha = (t - segment_start_t) / segment_duration
            local_alpha = float(np.clip(local_alpha, 0.0, 1.0))
            local_alpha = local_alpha * local_alpha * (3.0 - 2.0 * local_alpha)
            target_q = waypoints[segment_index] + (waypoints[segment_index + 1] - waypoints[segment_index]) * local_alpha
            self.data.qpos[self._qposadr] = target_q
            self.data.qvel[self._dofadr] = 0
            self.data.ctrl[self._ctrladr] = 0.0
            mujoco.mj_step(self.model, self.data)
            self._apply_joint_hold()
            self._sync_viewer()
            final_err = joint_waypoints[-1] - self.data.qpos[self._qposadr]
            if t >= planned_seconds and float(np.max(np.abs(final_err))) < joint_tolerance:
                self.data.ctrl[self._ctrladr] = 0
                self.last_motion_diagnostics = {
                    "status": "reached",
                    "path_mode": "blended_joint_waypoints",
                    "trajectory_mode": "time_parameterized",
                    "waypoint_count": len(joint_waypoints),
                    "intermediate_stop_count": 0,
                    "intermediate_waypoints_reached": len(joint_waypoints) - 1,
                    "max_joint_error_rad": float(np.max(np.abs(final_err))),
                    "elapsed_sim_seconds": float((step_index + 1) * self.model.opt.timestep),
                    "commanded_joint_speed_rad_s": float(speed),
                    "planned_joint_motion_seconds": planned_seconds,
                }
                return 0

        settle_steps = max(1, int(0.35 / self.model.opt.timestep))
        for settle_index in range(settle_steps):
            final_err = joint_waypoints[-1] - self.data.qpos[self._qposadr]
            if float(np.max(np.abs(final_err))) < joint_tolerance:
                self.data.ctrl[self._ctrladr] = 0
                self.last_motion_diagnostics = {
                    "status": "reached",
                    "path_mode": "blended_joint_waypoints",
                    "trajectory_mode": "time_parameterized",
                    "waypoint_count": len(joint_waypoints),
                    "intermediate_stop_count": 0,
                    "intermediate_waypoints_reached": len(joint_waypoints) - 1,
                    "final_settle_steps": int(settle_index),
                    "max_joint_error_rad": float(np.max(np.abs(final_err))),
                    "elapsed_sim_seconds": float((total_steps + settle_index) * self.model.opt.timestep),
                    "commanded_joint_speed_rad_s": float(speed),
                }
                return 0
            self.data.ctrl[self._ctrladr] = np.clip(final_err * 10.0, -speed, speed)
            mujoco.mj_step(self.model, self.data)
            self._apply_joint_hold()
            self._sync_viewer()

        self.data.ctrl[self._ctrladr] = 0
        final_err = joint_waypoints[-1] - self.data.qpos[self._qposadr]
        self.last_motion_diagnostics = {
            "status": "timeout",
            "path_mode": "blended_joint_waypoints",
            "trajectory_mode": "time_parameterized",
            "waypoint_count": len(joint_waypoints),
            "intermediate_stop_count": 0,
            "intermediate_waypoints_reached": len(joint_waypoints) - 1,
            "final_settle_steps": int(settle_steps),
            "max_joint_error_rad": float(np.max(np.abs(final_err))),
            "joint_error_rad": final_err.tolist(),
            "target_q_rad": joint_waypoints[-1].tolist(),
            "actual_q_rad": self.data.qpos[self._qposadr].copy().tolist(),
            "elapsed_sim_seconds": float((total_steps + settle_steps) * self.model.opt.timestep),
            "commanded_joint_speed_rad_s": float(speed),
        }
        return 100

    def _sync_viewer(self):
        if self.viewer is not None:
            self.viewer.sync()
        if self.sync_callback is not None:
            self.sync_callback(self.model, self.data)
        if self.realtime:
            time.sleep(float(self.model.opt.timestep))

    def _apply_joint_hold(self):
        if self._hold_qposadr is None or self._hold_dofadr is None or self._hold_values is None:
            return
        self.data.qpos[self._hold_qposadr] = self._hold_values
        self.data.qvel[self._hold_dofadr] = 0
        mujoco.mj_forward(self.model, self.data)

    def _try_attach_nearest_leaf(self):
        if self.attached_body_name is not None:
            return
        suction_pos = self.data.site_xpos[self.suction_site_id].copy()
        candidates = []
        for target in self._suction_targets:
            body_name, site_name, equality_name = target[:3]
            site_pos = self.data.site_xpos[self.model.site(site_name).id].copy()
            weld_body_name = target[3] if len(target) > 3 else body_name
            candidates.append((float(np.linalg.norm(site_pos - suction_pos)), body_name, equality_name, weld_body_name))
        distance, body_name, equality_name, weld_body_name = min(candidates, key=lambda item: item[0])
        self.last_suction_distance_m = distance
        if distance > self.suction_threshold_m:
            return
        eq_id = self.model.equality(equality_name).id
        self._set_weld_to_current_relative_pose(eq_id, self.vacuum_body, weld_body_name)
        self.data.eq_active[eq_id] = 1
        self.attached_body_name = body_name
        mujoco.mj_forward(self.model, self.data)

    def _release(self):
        for target in self._suction_targets:
            equality_name = target[2]
            self.data.eq_active[self.model.equality(equality_name).id] = 0
        self.attached_body_name = None
        mujoco.mj_forward(self.model, self.data)

    def _set_weld_to_current_relative_pose(self, eq_id: int, parent_body: str, child_body: str):
        parent_id = self.model.body(parent_body).id
        child_id = self.model.body(child_body).id
        parent_pos = self.data.xpos[parent_id].copy()
        child_pos = self.data.xpos[child_id].copy()
        parent_rot = self.data.xmat[parent_id].reshape(3, 3).copy()
        child_rot = self.data.xmat[child_id].reshape(3, 3).copy()
        rel_pos = parent_rot.T @ (child_pos - parent_pos)
        rel_rot = parent_rot.T @ child_rot
        rel_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(rel_quat, rel_rot.reshape(-1))
        self.model.eq_data[eq_id, 3:6] = rel_pos
        self.model.eq_data[eq_id, 6:10] = rel_quat
        self.model.eq_data[eq_id, 10] = 1.0
