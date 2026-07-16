#!/usr/bin/env python3
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENE_DIR = ROOT / "scene"
OUT_PATH = SCENE_DIR / "compliant_bamboo_leaf_pair.xml"

SEGMENT_COUNT = 11
LEAF_LENGTH = 0.42
LEAF_MAX_WIDTH = 0.125
LEAF_THICKNESS = 0.003
LEAF_MASS = 0.010
BEND_RANGE_RAD = 1.20
BEND_DAMPING = 0.012
BEND_STIFFNESS = 0.045
BEND_ARMATURE = 0.0002


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


def add_geom(body: ET.Element, attrs: dict[str, str]):
    ET.SubElement(body, "geom", attrs)


def segment_width(index: int) -> float:
    if SEGMENT_COUNT == 1:
        return LEAF_MAX_WIDTH
    center = (SEGMENT_COUNT - 1) / 2.0
    normalized = abs(index - center) / center
    return LEAF_MAX_WIDTH * (0.24 + 0.76 * (1.0 - normalized**1.7))


def add_segment_body(parent: ET.Element, leaf_name: str, index: int, material: str, segment_length: float):
    if index == 0:
        x = -LEAF_LENGTH / 2.0 + segment_length / 2.0
    else:
        x = segment_length
    width = segment_width(index)
    body = ET.SubElement(parent, "body", {
        "name": f"{leaf_name}_seg_{index:02d}",
        "pos": f"{x:.5f} 0 0",
    })
    if index > 0:
        ET.SubElement(body, "joint", {
            "name": f"{leaf_name}_bend_{index:02d}",
            "type": "hinge",
            "axis": "0 1 0",
            "pos": f"{-segment_length / 2.0:.5f} 0 0",
            "range": f"{-BEND_RANGE_RAD:.2f} {BEND_RANGE_RAD:.2f}",
            "damping": f"{BEND_DAMPING:.3f}",
            "stiffness": f"{BEND_STIFFNESS:.3f}",
            "armature": f"{BEND_ARMATURE:.4f}",
            "limited": "true",
        })
    add_geom(body, {
        "name": f"{leaf_name}_seg_{index:02d}_geom",
        "type": "box",
        "size": f"{segment_length * 0.56:.5f} {width / 2.0:.5f} {LEAF_THICKNESS / 2.0:.5f}",
        "material": material,
        "mass": f"{LEAF_MASS / SEGMENT_COUNT:.6f}",
        "friction": "4 2 1",
        "condim": "6",
    })
    if index == SEGMENT_COUNT // 2:
        ET.SubElement(body, "site", {
            "name": f"{leaf_name}_pick_site",
            "pos": f"0 0 {LEAF_THICKNESS / 2.0 + 0.003:.5f}",
            "type": "sphere",
            "size": "0.007",
            "rgba": "0 1 0 0.7",
        })
        ET.SubElement(body, "site", {
            "name": f"{leaf_name}_center_site",
            "pos": f"0 0 {LEAF_THICKNESS / 2.0 + 0.002:.5f}",
            "type": "sphere",
            "size": "0.006",
            "rgba": "1 0 0 0.6",
        })
    return body


def add_leaf(worldbody: ET.Element, leaf_name: str, pos: str, material: str):
    segment_length = LEAF_LENGTH / SEGMENT_COUNT
    root = ET.SubElement(worldbody, "body", {
        "name": leaf_name,
        "pos": pos,
        "euler": "0 0 1.5708",
    })
    ET.SubElement(root, "freejoint", {"name": f"{leaf_name}_freejoint"})
    parent = root
    for index in range(SEGMENT_COUNT):
        parent = add_segment_body(parent, leaf_name, index, material, segment_length)
    return root


def main():
    root = ET.Element("mujoco", {"model": "compliant bamboo leaf pair"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "material", {
        "name": "compliant_bamboo_mat_bottom",
        "rgba": "0.62 0.46 0.18 1",
        "specular": "0.08",
        "shininess": "0.12",
    })
    ET.SubElement(asset, "material", {
        "name": "compliant_bamboo_mat_top",
        "rgba": "0.76 0.58 0.24 1",
        "specular": "0.08",
        "shininess": "0.12",
    })
    worldbody = ET.SubElement(root, "worldbody")
    add_leaf(worldbody, "staged_bamboo_leaf_bottom", "-0.58 0.56 0.4815", "compliant_bamboo_mat_bottom")
    add_leaf(worldbody, "staged_bamboo_leaf_top", "-0.58 0.56 0.4845", "compliant_bamboo_mat_top")
    indent(root)
    OUT_PATH.write_text(
        "<?xml version='1.0' encoding='utf-8'?>\n" + ET.tostring(root, encoding="unicode"),
        encoding="utf-8",
    )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
