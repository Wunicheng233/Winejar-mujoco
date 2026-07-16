import os
import cv2
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import threading
import glfw
import math
from collections import deque

class XArm6Controller:
    def __init__(self, model_path):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))
        
        self.model = mujoco.MjModel.from_xml_path(fullpath)
        self.data = mujoco.MjData(self.model)

        self._init_joint_indices()

        self.img_h = 64
        self.img_w = 64
        self._rgb_buffer1 = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        self._rgb_buffer2 = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        self._init_d435()

        self.handle = mujoco.viewer.launch_passive(self.model, self.data)
        self.handle.cam.distance = 2
        self.handle.cam.azimuth = 90
        self.handle.cam.elevation = -30

        self.model.opt.timestep = 0.003
        self._dt = 0.003
        self._simulation_running = False
        self._simulation_thread = None

        # pd_force_control
        self.last_error = np.zeros(6, dtype=np.float64)
        self.last_time = time.time()

        # ------------------- 缓冲区相关初始化 -------------------
        self.buffer_size = 3  # 缓冲区最大帧数（建议2-5，平衡速度和内存）
        self.image_buffer = deque(maxlen=self.buffer_size)  # 循环缓冲区（自动丢弃旧帧）
        self.buffer_cond = threading.Condition()  # 缓冲区同步用的条件变量

        for _ in range(self.buffer_size):
            # 存储格式：(相机1灰度图, 相机2灰度图, 帧时间戳)
            self.image_buffer.append((
                np.zeros((self.img_h, self.img_w), dtype=np.uint8),
                np.zeros((self.img_h, self.img_w), dtype=np.uint8),
                time.time()
            ))

    def _init_joint_indices(self):
        self.arm_joints = [self.model.joint(name).id for name in 
                          ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']]
        # self.gripper_joints = [self.model.joint(name).id for name in
        #                      ['left_driver_joint', 'right_driver_joint']]
        self.camID1 = self.model.camera('camera1').id
        self.camID2 = self.model.camera('camera_third').id
        self.deskgeom = self.model.geom('deskgeom').id
        self.object1 = self.model.body('box_with_hole').id
        self.object1_joint = self.model.joint('object1_joint').id
        self.link_tcp = self.model.site('link_tcp').id
        self.object1_site = self.model.site('object1_site').id

        print(self.link_tcp)
        print(self.object1_site)

    def _changeColor(self):
        self.model.geom_rgba[self.deskgeom] = np.append(np.random.rand(3), 1.0)

    def _init_d435(self):
        if not glfw.init():
            raise RuntimeError("GLFW初始化失败")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window1 = glfw.create_window(self.img_h, self.img_w, "Camera 1", None, None)
        glfw.make_context_current(self.window1)
    
        self.camera_D435i_1 = mujoco.MjvCamera()
        self.camera_D435i_1.fixedcamid = self.camID1
        self.camera_D435i_1.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._scene1 = mujoco.MjvScene(self.model, maxgeom=10000)
        self._context1 = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._context1)

        self.window2 = glfw.create_window(self.img_h, self.img_w, "Camera 2", None, None)
        glfw.make_context_current(self.window2)
    
        self.camera_D435i_2 = mujoco.MjvCamera()
        self.camera_D435i_2.fixedcamid = self.camID2
        self.camera_D435i_2.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._scene2 = mujoco.MjvScene(self.model, maxgeom=10000)
        self._context2 = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._context2)

        glfw.make_context_current(None)

        # self._scene1.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False

    def get_gray_image(self, camera_id):
        """主线程专用：从缓冲区读取图像"""
        if camera_id not in [1, 2]:
            raise ValueError("Invalid camera ID (仅支持1或2)")
        
        with self.buffer_cond:
            while True:
                latest_cam1, latest_cam2, latest_ts = self.image_buffer[-1]
                if time.time() - latest_ts < 0.1:
                    break
                self.buffer_cond.wait(timeout=0.1)
        if camera_id == 1:
            return latest_cam1.copy()
        else:
            return latest_cam2.copy()

    def start_simulation(self):
        if self._simulation_thread is None or not self._simulation_thread.is_alive():
            self._simulation_running = True
            self._simulation_thread = threading.Thread(target=self._simulation_loop)
            self._simulation_thread.daemon = True
            self._simulation_thread.start()
            print("仿真线程已启动")
        else:
            print("仿真线程已在运行")
    
    def stop_simulation(self):
        self._simulation_running = False
        if self._simulation_thread is not None:
            self._simulation_thread.join(timeout=1.0)
    
    def _simulation_loop(self):

        real_start = time.time()
        sim_start = self.data.time
        last_cam_render = time.time()

        while self.handle.is_running() and self._simulation_running:
            start_time = time.time()

            # self.draw_line([0.3, -0.2, 0.05], [0.3, 0.2, 0.05], 10, [1.0, 0.0, 0.0, 1.0])
            # self.draw_line([0.4, -0.2, 0.05], [0.4, 0.2, 0.05], 10, [1.0, 0.0, 0.0, 1.0])
            # self.draw_line([0.3, -0.2, 0.05], [0.3, -0.2, 0.05], 10, [1.0, 0.0, 0.0, 1.0])
            # self.draw_line([0.3, -0.2, 0.05], [0.3, -0.2, 0.05], 10, [1.0, 0.0, 0.0, 1.0])

            box_xfrc_applied = self.data.xfrc_applied[self.object1]
            box_xfrc_applied[2] = -1000 #z轴方向，单独给物块重力
            mujoco.mj_step(self.model, self.data)
            
            self.handle.sync()

            if time.time() - last_cam_render >= 0.1:
                cam1_img = self._read_camera_gray(1, self.img_h, self.img_w)
                cam2_img = self._read_camera_gray(2, self.img_h, self.img_w)
                cam1_img_copy = cam1_img.copy()
                cam2_img_copy = cam2_img.copy()
                timestamp = time.time()

                with self.buffer_cond:
                    self.image_buffer.append((cam1_img_copy, cam2_img_copy, timestamp))
                    self.buffer_cond.notify()  # 通知主线程：有新图像可用
                last_cam_render = time.time()
            
            elapsed = time.time() - start_time
            if elapsed < self._dt:
                time.sleep(self._dt - elapsed)

            if (self.data.time - sim_start) >= 1.0:
                real_elapsed = time.time() - real_start
                print(f"仿真时间: {self.data.time - sim_start:.3f}s")
                print(f"现实时间: {real_elapsed:.3f}s")
                print(f"实时比: {(self.data.time - sim_start) / real_elapsed:.3f}")
                real_start = time.time()
                sim_start = self.data.time

    def _read_camera_gray(self, camera_id, w, h):
        """子线程专用：读取相机灰度图"""
        if camera_id == 1:
            window = self.window1
            camera = self.camera_D435i_1
            scene = self._scene1
            context = self._context1
            buffer = self._rgb_buffer1
        elif camera_id == 2:
            window = self.window2
            camera = self.camera_D435i_2
            scene = self._scene2
            context = self._context2
            buffer = self._rgb_buffer2
        else:
            raise ValueError("Invalid camera ID")
        
        glfw.make_context_current(window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(
            self.model, self.data, mujoco.MjvOption(), 
            None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
        )
        mujoco.mjr_render(viewport, scene, context)
        mujoco.mjr_readPixels(buffer, None, viewport, context)
        glfw.make_context_current(None)

        flipped_rgb = np.flipud(buffer)
        gray_image = cv2.cvtColor(flipped_rgb, cv2.COLOR_RGB2GRAY)
        return gray_image
    
    def draw_line(self, start, end, width, rgba):
        self.handle.user_scn.ngeom += 1
        geom = self.handle.user_scn.geoms[self.handle.user_scn.ngeom - 1]
        size = [0.0, 0.0, 0.0] 
        pos = [0, 0, 0]           
        mat = [0, 0, 0, 0, 0, 0, 0, 0, 0]     
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, size, pos, mat, rgba)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, start, end)

    def get_joint_states(self):
        return {
            'positions': self.data.qpos[self.arm_joints].copy(),
            'velocities': self.data.qvel[self.arm_joints].copy(),
            'efforts': self.data.qfrc_actuator[self.arm_joints].copy() #力矩
        }

    def get_force_sensor(self):
        return {
            'force': -self.data.sensor("force").data.copy(),
            'torque': -self.data.sensor("torque").data.copy()
        }

    def set_joint_positions(self, positions):
        assert len(positions) == len(self.arm_joints)
        self.data.qpos[self.arm_joints] = positions

    def set_joint_vel(self, vel):
        assert len(vel) == len(self.arm_joints)
        self.data.ctrl[:6] = vel
        # self.data.qvel[:6] = vel

    # def set_gripper(self, width):
    #     """设置夹爪开合宽度"""
    #     # range: 0~1
    #     ctrl_value = width * 255
    #     self.data.ctrl[-1:] = [ctrl_value]

    def set_object1(self, positions):
        assert len(positions) == 7
        self.data.qpos[self.object1_joint : self.object1_joint + 7] = positions

    def set_tcp_vel(self, real_speed: np.ndarray) -> None:
        speed_c = [1, -1, -1, 1, -1, -1]
        speed = np.array([real_speed * speed_c for real_speed, speed_c in zip(real_speed, speed_c)], dtype = np.float64)
        nv = self.model.nv
        # print(f"模型实际速度自由度 nv: {nv}") 
        damping = 0.05
        joint_vel_limit = 3.14159

        jacobian = np.zeros((6, nv), dtype = np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacobian[:3], jacobian[3:], self.link_tcp)
        jac_T = jacobian.T
        identity = np.eye(6)
        inv_term = np.linalg.inv(jacobian @ jac_T + damping ** 2 * identity)
        # inv_term = np.linalg.inv(jacobian @ jac_T +  2 * identity)
        joint_vel_des = jac_T @ inv_term @ speed

        joint_vel_des = np.clip(joint_vel_des, -joint_vel_limit, joint_vel_limit)
        joint_vel_des = joint_vel_des[:6]
        self.set_joint_vel(joint_vel_des)

        # print(f"joint_vel_des:{joint_vel_des[:6]}")
        # print(self.data.site_xpos[self.link_tcp])
        # link_tcp_quat = self.data.site_xmat[self.link_tcp]
        # print(f"velocities:{self.get_joint_states()['velocities']}")

        #检测末端速度
        current_joint_vel = self.get_joint_states()['velocities']
        current_joint_vel = np.concatenate([current_joint_vel, [0]*6], axis=0).astype(np.float32)
        # print(current_joint_vel)
        # print(f"current_joint_vel 形状: {current_joint_vel.shape}, 长度: {len(current_joint_vel)}")
        end_vel = jacobian @ current_joint_vel
        # print(f"end_vel:{end_vel}")
        
    def _pd_force_control(self, Kp: np.ndarray, Kd: np.ndarray):
        desired_force = np.array([0.0, 0.0, 0.0]) 
        desired_torque = np.array([0.0, 0.0, 0.0]) 
        current_time = time.time()
        if self.last_time is None:
            dt = 1e-4
        else:
            dt = current_time - self.last_time
            dt = max(dt, 1e-4)

        actual_force = np.clip(self.get_force_sensor()["force"], -200, 200) / 200
        actual_torque = np.clip(self.get_force_sensor()["torque"], -8, 8) / 8
        force_error = actual_force - desired_force
        torque_error = actual_torque - desired_torque
        force_error = np.where(np.abs(force_error) < 1e-3, 0, force_error)
        torque_error = np.where(np.abs(torque_error) < 1e-3, 0, torque_error)

        current_error = np.concatenate([force_error, torque_error], axis=0)
        error_dot = (current_error - self.last_error) / dt

        tcp_vel_cmd = Kp * current_error + Kd * error_dot
        tcp_vel_cmd[3:] = tcp_vel_cmd[3:] * np.pi
        linear_vel_limit = 1  # 末端线速度上限（m/s，根据机械臂参数调整）
        angular_vel_limit = np.pi  # 末端角速度上限（rad/s）
        tcp_vel_cmd[:3] = np.clip(tcp_vel_cmd[:3], -linear_vel_limit, linear_vel_limit)
        tcp_vel_cmd[3:] = np.clip(tcp_vel_cmd[3:], -angular_vel_limit, angular_vel_limit)

        self.set_tcp_vel(tcp_vel_cmd)
        self.last_error = current_error.copy()
        self.last_time = current_time
        # print(f"力误差: {force_error.round(3)}, 力矩误差: {torque_error.round(3)}")
        # print(f"末端速度指令: {tcp_vel_cmd.round(4)}")

    def close(self):
        self.handle.close()
        if self.window1:
            glfw.destroy_window(self.window1)
        if self.window2:
            glfw.destroy_window(self.window2)
        glfw.terminate()


