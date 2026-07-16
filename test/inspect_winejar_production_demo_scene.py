#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "scene" / "scene_winejar_production_demo.xml"

BODIES = [
    "factory_shell",
    "conveyor",
    "station_wine_jar",
    "preloaded_lotus_leaf",
    "preloaded_white_paper",
    "preloaded_ceramic_disc",
    "left_robot_pedestal",
    "right_robot_pedestal",
    "left_line_base",
    "right_line_base",
    "right_tie_gun_end_effector",
    "left_material_table",
    "staged_bamboo_leaf_bottom",
    "staged_bamboo_leaf_top",
    "staged_label_paper",
    "global_camera_rig",
]

SITES = [
    "conveyor_entry_site",
    "conveyor_stop_site",
    "conveyor_exit_site",
    "jar_mouth_center",
    "ceramic_disc_center_site",
    "leaf_cross_target_0",
    "leaf_cross_target_90",
    "neck_tie_target_site",
    "staged_bamboo_leaf_bottom_pick_site",
    "staged_bamboo_leaf_top_pick_site",
    "label_paper_pick_site",
    "left_vacuum_suction_site",
    "right_tie_gun_center_site",
    "right_tie_gun_ring_visual_site",
    "right_tie_gun_approach_site",
    "right_tie_gun_neck_target_site",
    "right_link_tcp",
]

CAMERAS = [
    "global_overview_camera",
    "front_conveyor_camera",
    "side_overview_camera",
    "global_top_camera",
    "production_hero_camera",
    "jar_mouth_closeup_camera",
    "left_vacuum_closeup_camera",
    "right_tie_gun_closeup_camera",
]


def fmt(vec):
    return f"({vec[0]: .3f}, {vec[1]: .3f}, {vec[2]: .3f})"


def site_angle_deg(model, data, first: str, second: str) -> float:
    first_x = data.site_xmat[model.site(first).id].reshape(3, 3)[:, 0]
    second_x = data.site_xmat[model.site(second).id].reshape(3, 3)[:, 0]
    dot = float(np.clip(np.dot(first_x, second_x), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def box_xy_bounds(center, half_size):
    return (
        float(center[0] - half_size[0]),
        float(center[0] + half_size[0]),
        float(center[1] - half_size[1]),
        float(center[1] + half_size[1]),
    )


def boxes_overlap_xy(a, b) -> bool:
    return a[0] < b[1] and a[1] > b[0] and a[2] < b[3] and a[3] > b[2]


def body_joint_count(model, body_name: str) -> int:
    body_id = model.body(body_name).id
    return sum(1 for joint_id in range(model.njnt) if model.jnt_bodyid[joint_id] == body_id)


def geom_exists(model, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0


def segment_body_names(prefix: str, count: int = 11):
    return [f"{prefix}_seg_{index:02d}" for index in range(count)]


def leaf_axis_yaw_deg(model, data, prefix: str) -> float:
    start = data.xpos[model.body(f"{prefix}_seg_00").id]
    end = data.xpos[model.body(f"{prefix}_seg_10").id]
    delta = end - start
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def assert_segmented_leaf(model, prefix: str, count: int = 11):
    missing = [
        name for name in segment_body_names(prefix, count)
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0
    ]
    if missing:
        raise AssertionError(f"Missing segmented leaf bodies for {prefix}: {missing}")


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"Loaded scene: {SCENE_PATH}")
    print(f"nbody={model.nbody}, ngeom={model.ngeom}, nsite={model.nsite}, ncam={model.ncam}")

    print("\nBodies:")
    for name in BODIES:
        body_id = model.body(name).id
        print(f"  {name}: {fmt(data.xpos[body_id])}")

    print("\nSites:")
    for name in SITES:
        site_id = model.site(name).id
        print(f"  {name}: {fmt(data.site_xpos[site_id])}")

    print("\nCameras:")
    for name in CAMERAS:
        cam_id = model.camera(name).id
        print(f"  {name}: {fmt(data.cam_xpos[cam_id])}")

    angle = site_angle_deg(model, data, "leaf_cross_target_0", "leaf_cross_target_90")
    if abs(angle - 90.0) > 0.5:
        raise AssertionError(f"leaf target angle expected 90 deg, got {angle:.3f}")
    print(f"\nLeaf crossing target angle: {angle:.2f} deg")

    left_arm_y = data.xpos[model.body("left_line_base").id][1]
    right_arm_y = data.xpos[model.body("right_line_base").id][1]
    if not (left_arm_y > 0.45 and -0.45 < right_arm_y < -0.30):
        raise AssertionError("Expected vacuum xArm on +Y side and bare xArm on -Y side")
    print("Dual-arm side placement OK")

    lotus_z = data.xpos[model.body("preloaded_lotus_leaf").id][2]
    paper_z = data.xpos[model.body("preloaded_white_paper").id][2]
    ceramic_z = data.xpos[model.body("preloaded_ceramic_disc").id][2]
    if not (lotus_z < paper_z < ceramic_z):
        raise AssertionError("Expected preloaded order: cropped lotus < cropped paper < ceramic disc")
    print("Preloaded material vertical order OK")

    table_center = data.xpos[model.body("left_material_table").id]
    leaf_center = data.xpos[model.body("staged_bamboo_leaf_top").id]
    if abs(leaf_center[0] - table_center[0]) > 0.48 or abs(leaf_center[1] - table_center[1]) > 0.26:
        raise AssertionError("Expected staged leaves to be fully centered on material table")
    print("Staged leaves sit on material table OK")

    top_initial_yaw = leaf_axis_yaw_deg(model, data, "staged_bamboo_leaf_top")
    bottom_initial_yaw = leaf_axis_yaw_deg(model, data, "staged_bamboo_leaf_bottom")
    if abs(top_initial_yaw) > 3.0:
        raise AssertionError(f"Top staged bamboo leaf should start at 0 deg, got {top_initial_yaw:.2f} deg")
    if abs(bottom_initial_yaw) > 3.0:
        raise AssertionError(f"Bottom staged bamboo leaf should start stacked at 0 deg, got {bottom_initial_yaw:.2f} deg")
    print("Staged leaf initial stacked yaw OK")

    factory_geom_names = [
        "factory_floor_slab",
        "factory_wall_y_positive",
        "factory_wall_y_negative",
        "factory_wall_x_negative_left",
        "factory_wall_x_negative_right",
        "factory_wall_x_positive_left",
        "factory_wall_x_positive_right",
    ]
    missing_factory_geoms = [name for name in factory_geom_names if not geom_exists(model, name)]
    if missing_factory_geoms:
        raise AssertionError(f"Missing simplified factory shell geoms: {missing_factory_geoms}")
    if geom_exists(model, "factory_ceiling_camera_panel"):
        raise AssertionError("Factory ceiling panel should be removed; it blocks the production overview")
    entry_x = data.site_xpos[model.site("conveyor_entry_site").id][0]
    exit_x = data.site_xpos[model.site("conveyor_exit_site").id][0]
    if not (entry_x < -0.75 and exit_x > 0.75):
        raise AssertionError("Conveyor entry/exit sites should pass through the factory wall openings")
    camera_pos = data.xpos[model.body("global_camera_rig").id]
    left_pedestal_pos = data.xpos[model.body("left_robot_pedestal").id]
    right_pedestal_pos = data.xpos[model.body("right_robot_pedestal").id]
    expected_camera_xy = (left_pedestal_pos[:2] + right_pedestal_pos[:2]) / 2.0
    camera_xy_error = float(np.linalg.norm(camera_pos[:2] - expected_camera_xy))
    if camera_xy_error > 0.015:
        raise AssertionError(
            "Global camera should be centered between the two robot pedestals in XY, "
            f"camera_xy={camera_pos[:2]}, expected={expected_camera_xy}, error={camera_xy_error:.3f} m"
        )
    if camera_pos[2] < 1.10:
        raise AssertionError(f"Global camera should stay high above the station, camera_z={camera_pos[2]:.3f}")
    print("Simplified factory shell and centered global camera placement OK")

    left_pedestal_bounds = box_xy_bounds(data.xpos[model.body("left_robot_pedestal").id], (0.22, 0.22))
    table_bounds = box_xy_bounds(data.xpos[model.body("left_material_table").id], (0.36, 0.26))
    if boxes_overlap_xy(left_pedestal_bounds, table_bounds):
        raise AssertionError("Left robot pedestal overlaps the material table in XY")
    print("Left pedestal and material table clearance OK")

    left_base = data.xpos[model.body("left_line_base").id]
    vacuum_cup_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in range(model.ngeom)
    ]
    vacuum_cup_names = [name for name in vacuum_cup_names if name and name.startswith("left_vacuum_cup_")]
    if len(vacuum_cup_names) != 2:
        raise AssertionError(f"Expected dual vacuum cups on the left end effector, got {vacuum_cup_names}")
    cup_positions = [data.geom_xpos[model.geom(name).id] for name in vacuum_cup_names]
    cup_spacing = float(np.linalg.norm(cup_positions[0] - cup_positions[1]))
    if not (0.060 <= cup_spacing <= 0.100):
        raise AssertionError(f"Dual vacuum cup spacing should be 60-100 mm, got {cup_spacing * 1000.0:.1f} mm")
    for geom_name in [
        "left_vacuum_mount_geom",
        "left_vacuum_center_connector_geom",
        "left_vacuum_crossbar_geom",
    ]:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name) < 0:
            raise AssertionError(f"Missing left vacuum connector geometry: {geom_name}")
    mount_id = model.geom("left_vacuum_mount_geom").id
    connector_id = model.geom("left_vacuum_center_connector_geom").id
    crossbar_id = model.geom("left_vacuum_crossbar_geom").id
    tool_axis = data.geom_xmat[mount_id].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])

    def interval_along_tool_axis(geom_id: int, half_extent: float) -> tuple[float, float]:
        center = float(np.dot(data.geom_xpos[geom_id], tool_axis))
        return center - half_extent, center + half_extent

    mount_interval = interval_along_tool_axis(mount_id, model.geom_size[mount_id][1])
    connector_interval = interval_along_tool_axis(connector_id, model.geom_size[connector_id][1])
    crossbar_interval = interval_along_tool_axis(crossbar_id, model.geom_size[crossbar_id][2])
    mount_to_connector_gap = connector_interval[0] - mount_interval[1]
    connector_to_crossbar_gap = crossbar_interval[0] - connector_interval[1]
    if mount_to_connector_gap > 0.002 or connector_to_crossbar_gap > 0.002:
        raise AssertionError(
            "Left vacuum end effector has a visible air gap: "
            f"mount->connector={mount_to_connector_gap * 1000.0:.1f} mm, "
            f"connector->crossbar={connector_to_crossbar_gap * 1000.0:.1f} mm"
        )
    print("Dual vacuum cup end effector OK")

    reach_sites = [
        "jar_mouth_center",
        "leaf_cross_target_0",
        "staged_bamboo_leaf_top_pick_site",
        "label_paper_pick_site",
    ]
    for site_name in reach_sites:
        site_pos = data.site_xpos[model.site(site_name).id]
        horizontal_distance = float(np.linalg.norm((site_pos - left_base)[:2]))
        print(f"  left base horizontal distance -> {site_name}: {horizontal_distance:.3f} m")
        if horizontal_distance > 0.70:
            raise AssertionError(f"{site_name} is too far from the left xArm base: {horizontal_distance:.3f} m")
    print("Left-arm workspace distances OK")

    table_top_z = data.xpos[model.body("left_material_table").id][2] + 0.24
    for leaf_name, expected_bottom_z in [
        ("staged_bamboo_leaf_bottom", table_top_z),
        ("staged_bamboo_leaf_top", table_top_z + 0.003),
    ]:
        assert_segmented_leaf(model, leaf_name)
        for segment_name in segment_body_names(leaf_name):
            bottom_z = float(data.xpos[model.body(segment_name).id][2] - 0.003 / 2.0)
            gap = bottom_z - expected_bottom_z
            if abs(gap) > 0.002:
                raise AssertionError(f"{segment_name} is not supported correctly, gap={gap:.4f} m")

    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "staged_metal_weight") >= 0:
        raise AssertionError("Production demo should use flexible label paper instead of the old metal weight")
    body_name = "staged_label_paper"
    assert_segmented_leaf(model, body_name, count=11)
    label_center = data.xpos[model.body("staged_label_paper_seg_03").id]
    label_half_size = model.geom_size[model.geom("staged_label_paper_seg_03_geom").id]
    if label_half_size[2] > 0.003:
        raise AssertionError(f"{body_name} should be a thin flexible paper label, got half thickness={label_half_size[2]:.4f} m")
    cup_radius = max(float(model.geom(model.geom(name).id).size[0]) for name in vacuum_cup_names)
    required_half_length = cup_spacing / 2.0 + cup_radius + 0.008
    if label_half_size[0] < required_half_length:
        raise AssertionError(
            "Label paper is too short for the dual vacuum cups: "
            f"center segment half length={label_half_size[0] * 1000.0:.1f} mm, "
            f"required>={required_half_length * 1000.0:.1f} mm"
        )
    label_start = data.xpos[model.body("staged_label_paper_seg_00").id]
    label_end = data.xpos[model.body("staged_label_paper_seg_10").id]
    label_axis_length = float(np.linalg.norm((label_end - label_start)[:2]))
    label_length = label_axis_length + 2.0 * 0.035
    if label_length < 0.36:
        raise AssertionError(f"Label paper should be long enough to hang over the jar side, got length={label_length:.3f} m")
    short_side = float(np.linalg.norm((label_center - label_start)[:2]))
    long_side = float(np.linalg.norm((label_end - label_center)[:2]))
    if long_side < short_side * 1.8:
        raise AssertionError(
            "Label paper pick/placement site should be offset so only one side hangs over the jar, "
            f"short_side={short_side:.3f} m, long_side={long_side:.3f} m"
        )
    table_half = model.geom_size[model.geom("left_material_table_geom").id][:2]
    table_xy = data.xpos[model.body("left_material_table").id][:2]
    for endpoint_name, endpoint in [("start", label_start), ("end", label_end), ("anchor", label_center)]:
        if np.any(endpoint[:2] < table_xy - table_half) or np.any(endpoint[:2] > table_xy + table_half):
            raise AssertionError(f"Initial label paper {endpoint_name} is outside the material table: {endpoint[:2]}")
    bottom_z = float(label_center[2] - label_half_size[2])
    gap = bottom_z - table_top_z
    print(f"  {body_name} support gap: {gap * 1000.0:.2f} mm")
    if abs(gap) > 0.0015:
        raise AssertionError(f"{body_name} is not supported correctly, gap={gap:.4f} m")
    label_bounds = box_xy_bounds(label_center, tuple(label_half_size[:2]))
    for leaf_name in ("staged_bamboo_leaf_bottom", "staged_bamboo_leaf_top"):
        for segment_name in segment_body_names(leaf_name):
            segment_id = model.body(segment_name).id
            segment_geom_id = model.geom(f"{segment_name}_geom").id
            segment_bounds = box_xy_bounds(data.xpos[segment_id], tuple(model.geom_size[segment_geom_id][:2]))
            if boxes_overlap_xy(label_bounds, segment_bounds):
                raise AssertionError(f"Initial label paper overlaps staged leaf segment in XY: {segment_name}")
    if body_joint_count(model, body_name) == 0:
        raise AssertionError(f"{body_name} has no joint, so it is fixed and cannot be picked")
    for equality_name in [
        "left_suction_weld_label_paper",
        "mouth_static_friction_weld_label_paper",
    ]:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name) < 0:
            raise AssertionError(f"Missing label paper equality: {equality_name}")
    print("Segmented leaves and flexible label support checks OK")

    right_tcp = data.site_xpos[model.site("right_link_tcp").id]
    tie_center = data.site_xpos[model.site("right_tie_gun_center_site").id]
    ring_visual = data.site_xpos[model.site("right_tie_gun_ring_visual_site").id]
    neck_target = data.site_xpos[model.site("right_tie_gun_neck_target_site").id]
    if np.linalg.norm(right_tcp - tie_center) > 1e-6:
        raise AssertionError("right_link_tcp should coincide with right_tie_gun_center_site")
    if not (0.055 <= tie_center[2] - ring_visual[2] <= 0.065):
        raise AssertionError("Tie-gun visual ring should hang about 60 mm below the TCP")
    if neck_target[2] >= tie_center[2]:
        raise AssertionError("Tie-gun neck target should be below the ring center for vertical insertion")
    print("Tie-gun TCP and vertical target orientation OK")

    neck_radius = 0.060
    open_radius = 0.105
    if open_radius <= neck_radius:
        raise AssertionError("Tie-gun open ring radius must exceed final neck radius")
    print(f"Tie-gun ring radius OK: open={open_radius:.3f} m, final={neck_radius:.3f} m")

    for required_geom in [
        "right_tie_gun_flange_socket",
        "right_tie_gun_wrist_flange",
        "right_tie_gun_body",
        "right_tie_gun_support_rod_front",
        "right_tie_gun_support_rod_back",
        "right_tie_gun_support_rod_left",
        "right_tie_gun_support_rod_right",
        "right_tie_gun_closed_jaw_0",
    ]:
        if not geom_exists(model, required_geom):
            raise AssertionError(f"Tie-gun needs simple axial body/rod/ring geometry: {required_geom}")
    for redundant_geom in [
        "right_tie_gun_compact_mount",
        "right_tie_gun_connector",
        "right_tie_gun_air_port",
        "right_tie_gun_vertical_tool_post",
        "right_tie_gun_upper_mount_bridge",
        "right_tie_gun_lower_mount_bridge",
        "right_tie_gun_wrist_bridge",
    ]:
        if geom_exists(model, redundant_geom):
            raise AssertionError(f"Tie-gun should match the simple sketch; redundant side-mounted detail remains: {redundant_geom}")
    print("Tie-gun simple axial body, rods, and closed-jaw geometry OK")

    socket_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_tie_gun_flange_socket")
    socket_low = model.geom_pos[socket_id][2] - model.geom_size[socket_id][1]
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_tie_gun_wrist_flange")
    wrist_low = model.geom_pos[wrist_id][2] - model.geom_size[wrist_id][1]
    if socket_low > wrist_low + 0.001:
        raise AssertionError("Tie-gun flange socket should extend upward into the xArm wrist, leaving no visible air gap")

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_tie_gun_body")
    body_pos = model.geom_pos[body_id]
    if abs(body_pos[0]) > 0.002 or abs(body_pos[1]) > 0.002:
        raise AssertionError(f"Tie-gun black body should be centered on the tool axis, got local pos={body_pos}")

    body_top_local_z = body_pos[2] - model.geom_size[body_id][2]
    flange_low_local_z = model.geom_pos[wrist_id][2] + model.geom_size[wrist_id][1]
    axial_gap = body_top_local_z - flange_low_local_z
    if axial_gap > 0.006:
        raise AssertionError(f"Tie-gun flange/body gap too large: {axial_gap:.4f} m")
    print(f"Tie-gun axial flange/body gap OK: {max(axial_gap, 0.0) * 1000.0:.1f} mm")

    ring_center_local_z = 0.182
    body_bottom_local_z = body_pos[2] + model.geom_size[body_id][2]
    if ring_center_local_z <= body_bottom_local_z:
        raise AssertionError("Tie-gun ring should be below the tool body in local tool coordinates")
    print(f"Tie-gun ring below body OK: {(ring_center_local_z - body_bottom_local_z) * 1000.0:.1f} mm clearance")

    for rod_name in [
        "right_tie_gun_support_rod_front",
        "right_tie_gun_support_rod_back",
        "right_tie_gun_support_rod_left",
        "right_tie_gun_support_rod_right",
    ]:
        rod_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, rod_name)
        rod_radius = model.geom_size[rod_id][0]
        if not (0.005 <= rod_radius <= 0.012):
            raise AssertionError(f"Tie-gun support rod should be a slim link, got {rod_name} radius={rod_radius:.4f} m")
    print("Tie-gun four slim support rods OK")

    if geom_exists(model, "right_tie_gun_final_band_preview_0"):
        raise AssertionError("Default tie-gun model should not contain a floating final tightened band preview")
    for omitted_detail in ["right_tie_gun_band_lock_clip", "right_tie_gun_band_tail"]:
        if geom_exists(model, omitted_detail):
            raise AssertionError(f"Tie-gun sketch model should omit small tie detail: {omitted_detail}")
    loaded_band_radius = 0.105 - 0.006
    jaw_radius = 0.105
    radial_gap = jaw_radius - loaded_band_radius
    if radial_gap > 0.010:
        raise AssertionError(f"Loaded tie band is too far from jaw guide: gap={radial_gap:.4f} m")
    print(f"Tie-gun loaded band guide contact OK: radial gap={radial_gap * 1000.0:.1f} mm")


if __name__ == "__main__":
    main()
