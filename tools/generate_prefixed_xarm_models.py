#!/usr/bin/env python3
from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENE_DIR = ROOT / "scene"
SOURCE = SCENE_DIR / "xarm6_vacuum.xml"

NAME_REF_ATTRS = {
    "body1",
    "body2",
    "joint",
    "joint1",
    "joint2",
    "site",
    "tendon",
    "mesh",
    "material",
    "class",
    "childclass",
}


def indent(elem: ET.Element, level: int = 0):
    space = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = space + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = space
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = space


def implicit_mesh_name(path: str) -> str:
    return Path(path).stem


def collect_names(root: ET.Element) -> set[str]:
    names = set()
    for elem in root.iter():
        name = elem.get("name")
        if name:
            names.add(name)
        class_name = elem.get("class")
        if elem.tag == "default" and class_name:
            names.add(class_name)
        if elem.tag == "mesh":
            names.add(name or implicit_mesh_name(elem.get("file", "")))
    return names


def prefix_model(source_root: ET.Element, prefix: str, end_effector: str, base_pos: str, base_euler: str) -> ET.Element:
    root = copy.deepcopy(source_root)
    root.set("model", f"{prefix.rstrip('_')} xarm6")
    original_names = collect_names(root)

    for mesh in root.findall("./asset/mesh"):
        if "name" not in mesh.attrib:
            mesh.set("name", implicit_mesh_name(mesh.get("file", "")))

    for elem in root.iter():
        if "name" in elem.attrib:
            elem.set("name", prefix + elem.get("name"))

    for elem in root.iter():
        for attr, value in list(elem.attrib.items()):
            if attr not in NAME_REF_ATTRS:
                continue
            if attr == "mesh" and value in original_names:
                elem.set(attr, prefix + value)
            elif attr == "material" and value in original_names:
                elem.set(attr, prefix + value)
            elif attr in {"class", "childclass"} and value in original_names:
                elem.set(attr, prefix + value)
            elif attr not in {"mesh", "material", "class", "childclass"} and value in original_names:
                elem.set(attr, prefix + value)

    if end_effector == "bare":
        make_bare_end_effector(root, prefix)
    elif end_effector == "tie_gun":
        make_tie_gun_end_effector(root, prefix)
    elif end_effector == "vacuum":
        pass
    else:
        raise ValueError(f"Unknown end effector mode: {end_effector}")
    set_base_pose(root, prefix, base_pos, base_euler)

    indent(root)
    return root


def set_base_pose(root: ET.Element, prefix: str, base_pos: str, base_euler: str):
    target_name = prefix + "line_base"
    for body in root.iter("body"):
        if body.get("name") == target_name:
            body.set("pos", base_pos)
            body.set("euler", base_euler)
            return
    raise RuntimeError(f"Could not find body {target_name}")


def make_bare_end_effector(root: ET.Element, prefix: str):
    target_name = prefix + "vacuum_end_effector"
    for body in root.iter("body"):
        if body.get("name") != target_name:
            continue
        body.set("name", prefix + "bare_end_effector")
        body.attrib["pos"] = "0 0 0.015"
        for child in list(body):
            body.remove(child)
        ET.SubElement(body, "inertial", {
            "pos": "0 0 0.02",
            "mass": "0.16",
            "diaginertia": "0.00008 0.00008 0.00004",
        })
        ET.SubElement(body, "geom", {
            "name": prefix + "bare_flange_geom",
            "type": "cylinder",
            "pos": "0 0 0.012",
            "size": "0.032 0.012",
            "material": prefix + "gray",
            "mass": "0.16",
            "friction": "0.8 0.8 0.8",
        })
        ET.SubElement(body, "site", {
            "name": prefix + "link_tcp",
            "pos": "0 0 0.030",
            "type": "sphere",
            "size": "0.009",
            "rgba": "1 0 0 0.5",
            "group": "1",
        })
        return
    raise RuntimeError(f"Could not find body {target_name}")


def arc_points(radius: float, z: float, start_deg: float, end_deg: float, segments: int):
    points = []
    for index in range(segments + 1):
        t = index / segments
        angle = math.radians(start_deg + (end_deg - start_deg) * t)
        points.append((radius * math.cos(angle), radius * math.sin(angle), z))
    return points


def add_capsule_arc(
    parent: ET.Element,
    prefix: str,
    name: str,
    radius: float,
    z: float,
    start_deg: float,
    end_deg: float,
    segments: int,
    tube_radius: float,
    material: str,
):
    points = arc_points(radius, z, start_deg, end_deg, segments)
    for index, (a, b) in enumerate(zip(points, points[1:])):
        ET.SubElement(parent, "geom", {
            "name": f"{prefix}{name}_{index}",
            "type": "capsule",
            "fromto": f"{a[0]:.5f} {a[1]:.5f} {a[2]:.5f} {b[0]:.5f} {b[1]:.5f} {b[2]:.5f}",
            "size": f"{tube_radius:.5f}",
            "material": material,
            "mass": "0.01",
            "friction": "0.8 0.8 0.8",
        })


def make_tie_gun_end_effector(root: ET.Element, prefix: str):
    target_name = prefix + "vacuum_end_effector"
    for body in root.iter("body"):
        if body.get("name") != target_name:
            continue
        body.set("name", prefix + "tie_gun_end_effector")
        body.attrib["pos"] = "0 0 0.015"
        for child in list(body):
            body.remove(child)

        ET.SubElement(body, "inertial", {
            "pos": "0 0 0.055",
            "mass": "0.45",
            "diaginertia": "0.00035 0.00035 0.00018",
        })
        ET.SubElement(body, "geom", {
            "name": prefix + "tie_gun_connector",
            "type": "cylinder",
            "pos": "0 0 0.008",
            "size": "0.032 0.040",
            "material": prefix + "gray",
            "mass": "0.12",
            "friction": "0.8 0.8 0.8",
        })
        ET.SubElement(body, "geom", {
            "name": prefix + "tie_gun_body",
            "type": "box",
            "pos": "0 0 0.070",
            "size": "0.055 0.040 0.035",
            "material": prefix + "black",
            "mass": "0.20",
            "friction": "0.8 0.8 0.8",
        })
        ET.SubElement(body, "geom", {
            "name": prefix + "tie_gun_air_port",
            "type": "cylinder",
            "pos": "0 -0.052 0.072",
            "euler": "1.5708 0 0",
            "size": "0.006 0.018",
            "material": prefix + "gray",
            "mass": "0.03",
            "friction": "0.8 0.8 0.8",
        })

        ET.SubElement(body, "geom", {
            "name": prefix + "tie_gun_left_support_strut",
            "type": "capsule",
            "fromto": "-0.048 0 0.102 -0.090 0 0.160",
            "size": "0.005",
            "material": prefix + "black",
            "mass": "0.02",
            "friction": "0.8 0.8 0.8",
        })
        ET.SubElement(body, "geom", {
            "name": prefix + "tie_gun_right_support_strut",
            "type": "capsule",
            "fromto": "0.048 0 0.102 0.090 0 0.160",
            "size": "0.005",
            "material": prefix + "black",
            "mass": "0.02",
            "friction": "0.8 0.8 0.8",
        })

        jaw_z = 0.160
        ring_radius = 0.105
        add_capsule_arc(body, prefix, "tie_gun_left_open_jaw", ring_radius, jaw_z, 20, 160, 7, 0.006, prefix + "black")
        add_capsule_arc(body, prefix, "tie_gun_right_open_jaw", ring_radius, jaw_z, 200, 340, 7, 0.006, prefix + "black")
        add_capsule_arc(body, prefix, "tie_gun_loaded_band", ring_radius - 0.006, jaw_z - 0.001, 0, 360, 16, 0.0025, prefix + "gray")
        ET.SubElement(body, "geom", {
            "name": prefix + "tie_gun_band_lock_clip",
            "type": "box",
            "pos": f"{-ring_radius:.5f} 0 {jaw_z:.5f}",
            "size": "0.014 0.020 0.006",
            "material": prefix + "gray",
            "mass": "0.02",
            "friction": "0.8 0.8 0.8",
        })
        ET.SubElement(body, "geom", {
            "name": prefix + "tie_gun_band_tail",
            "type": "capsule",
            "fromto": f"{-ring_radius:.5f} 0 {jaw_z:.5f} {-ring_radius - 0.050:.5f} 0 {jaw_z:.5f}",
            "size": "0.0025",
            "material": prefix + "gray",
            "mass": "0.01",
            "friction": "0.8 0.8 0.8",
        })

        ET.SubElement(body, "site", {
            "name": prefix + "tie_gun_center_site",
            "pos": f"0 0 {jaw_z:.5f}",
            "type": "sphere",
            "size": "0.010",
            "rgba": "0 0.7 1 0.8",
        })
        ET.SubElement(body, "site", {
            "name": prefix + "tie_gun_approach_site",
            "pos": f"0 0 {jaw_z - 0.160:.5f}",
            "type": "sphere",
            "size": "0.008",
            "rgba": "0 1 0.3 0.7",
        })
        ET.SubElement(body, "site", {
            "name": prefix + "tie_gun_neck_target_site",
            "pos": f"0 0 {jaw_z + 0.010:.5f}",
            "type": "sphere",
            "size": "0.008",
            "rgba": "1 0 0 0.7",
        })
        ET.SubElement(body, "site", {
            "name": prefix + "link_tcp",
            "pos": f"0 0 {jaw_z:.5f}",
            "type": "sphere",
            "size": "0.009",
            "rgba": "1 0 0 0.5",
            "group": "1",
        })
        return
    raise RuntimeError(f"Could not find body {target_name}")


def write_xml(path: Path, root: ET.Element):
    path.write_text(
        "<?xml version='1.0' encoding='utf-8'?>\n" + ET.tostring(root, encoding="unicode"),
        encoding="utf-8",
    )
    print(f"Wrote {path}")


def main():
    source_root = ET.parse(SOURCE).getroot()
    write_xml(
        SCENE_DIR / "xarm6_left_vacuum.xml",
        prefix_model(source_root, "left_", end_effector="vacuum", base_pos="-0.12 0.62 0.12", base_euler="0 0 -1.5708"),
    )
    write_xml(
        SCENE_DIR / "xarm6_right_bare.xml",
        prefix_model(source_root, "right_", end_effector="bare", base_pos="-0.12 -0.62 0.12", base_euler="0 0 1.5708"),
    )
    write_xml(
        SCENE_DIR / "xarm6_right_tie_gun.xml",
        prefix_model(source_root, "right_", end_effector="tie_gun", base_pos="-0.12 -0.62 0.12", base_euler="0 0 1.5708"),
    )


if __name__ == "__main__":
    main()
