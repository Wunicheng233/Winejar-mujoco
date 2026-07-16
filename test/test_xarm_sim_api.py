from __future__ import annotations

from pathlib import Path
import unittest

import mujoco
import numpy as np

from mujoco_xarm6.sim.xarm_sim_api import SimXArmAPI


ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "scene" / "scene_winejar_vacuum.xml"
PRODUCTION_SCENE_PATH = ROOT / "scene" / "scene_winejar_production_demo.xml"


class SimXArmAPITest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def test_vacuum_does_not_attach_when_far(self):
        arm = SimXArmAPI(self.model, self.data, suction_threshold_m=0.005)

        code = arm.set_cgpio_digital(9, 1)

        self.assertEqual(code, 0)
        self.assertIsNone(arm.attached_body_name)
        self.assertFalse(any(self.data.eq_active))

    def test_vacuum_attaches_nearest_leaf_when_close(self):
        top_pick = self.data.site_xpos[self.model.site("bamboo_leaf_top_pick_site").id].copy()
        suction_id = self.model.site("vacuum_suction_site").id
        current_suction = self.data.site_xpos[suction_id].copy()
        delta = top_pick - current_suction
        leaf_qposadr = self.model.joint("bamboo_leaf_top_freejoint").qposadr[0]
        self.data.qpos[leaf_qposadr : leaf_qposadr + 3] -= delta
        mujoco.mj_forward(self.model, self.data)
        arm = SimXArmAPI(self.model, self.data, suction_threshold_m=0.02)

        code = arm.set_cgpio_digital(9, 1)

        self.assertEqual(code, 0)
        self.assertEqual(arm.attached_body_name, "bamboo_leaf_top")
        self.assertTrue(bool(self.data.eq_active[self.model.equality("suction_weld_top").id]))

    def test_vacuum_release_deactivates_weld(self):
        arm = SimXArmAPI(self.model, self.data, suction_threshold_m=1.0)
        arm.set_cgpio_digital(9, 1)

        code = arm.set_cgpio_digital(9, 0)

        self.assertEqual(code, 0)
        self.assertIsNone(arm.attached_body_name)
        self.assertFalse(any(self.data.eq_active))

    def test_left_prefixed_production_vacuum_attaches_staged_leaf_when_close(self):
        model = mujoco.MjModel.from_xml_path(str(PRODUCTION_SCENE_PATH))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        top_pick = data.site_xpos[model.site("staged_bamboo_leaf_top_pick_site").id].copy()
        suction = data.site_xpos[model.site("left_vacuum_suction_site").id].copy()
        delta = top_pick - suction
        leaf_qposadr = model.joint("staged_bamboo_leaf_top_freejoint").qposadr[0]
        data.qpos[leaf_qposadr : leaf_qposadr + 3] -= delta
        mujoco.mj_forward(model, data)
        arm = SimXArmAPI(
            model,
            data,
            tcp_site="left_link_tcp",
            suction_site="left_vacuum_suction_site",
            joint_names=[f"left_joint{i}" for i in range(1, 7)],
            actuator_names=[f"left_vel{i}" for i in range(1, 7)],
            vacuum_body="left_vacuum_end_effector",
            suction_targets=[
                ("staged_bamboo_leaf_top", "staged_bamboo_leaf_top_pick_site", "left_suction_weld_leaf_top"),
                ("staged_bamboo_leaf_bottom", "staged_bamboo_leaf_bottom_pick_site", "left_suction_weld_leaf_bottom"),
                ("staged_metal_weight", "metal_weight_pick_site", "left_suction_weld_metal_weight"),
            ],
            suction_threshold_m=0.02,
        )

        code = arm.set_cgpio_digital(9, 1)

        self.assertEqual(code, 0)
        self.assertEqual(arm.attached_body_name, "staged_bamboo_leaf_top")
        self.assertTrue(bool(data.eq_active[model.equality("left_suction_weld_leaf_top").id]))

if __name__ == "__main__":
    unittest.main()
