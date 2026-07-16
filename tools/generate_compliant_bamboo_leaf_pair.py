#!/usr/bin/env python3
"""Generate a centered, two-branch compliant bamboo-leaf model.

Each leaf has a rigidly referenced middle segment and two independent five
segment branches.  The layout gives all four free ends of the two leaves an
actual bending degree of freedom during the tie operation.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "scene" / "compliant_bamboo_leaf_pair.xml"
HALF_SEGMENT = 0.02138
PITCH = 0.03818
WIDTHS = (0.01500, 0.03000, 0.04257, 0.05250, 0.05942, 0.06250, 0.05942, 0.05250, 0.04257, 0.03000, 0.01500)


def indent(lines: list[str], spaces: int) -> list[str]:
    prefix = " " * spaces
    return [prefix + line for line in lines]


def geom(name: str, material: str, width: float) -> str:
    return (
        f'<geom name="{name}_geom" type="box" size="{HALF_SEGMENT:.5f} {width:.5f} 0.00150" '
        f'material="{material}" mass="0.000909" friction="4 2 1" condim="6" contype="1" conaffinity="5"/>'
    )


def branch(prefix: str, material: str, indices: list[int], direction: float, joint_numbers: list[int]) -> list[str]:
    """Emit one branch from seg_05 out to a leaf end."""
    result: list[str] = []
    for depth, (index, joint_number) in enumerate(zip(indices, joint_numbers)):
        joint_pos = -direction * (PITCH / 2.0)
        axis_y = 1.0 if direction > 0 else -1.0
        result.extend(
            [
                f'<body name="{prefix}_seg_{index:02d}" pos="{direction * PITCH:.5f} 0 0">',
                f'<joint name="{prefix}_bend_{joint_number:02d}" type="hinge" axis="0 {axis_y:.0f} 0" '
                f'pos="{joint_pos:.5f} 0 0" range="-1.20 1.20" damping="0.012" stiffness="0.045" armature="0.0002" limited="true"/>',
                geom(f"{prefix}_seg_{index:02d}", material, WIDTHS[index]),
            ]
        )
    result.extend(["</body>"] * len(indices))
    return result


def leaf(name: str, z: float, material: str) -> list[str]:
    center = [
        f'<body name="{name}" pos="-0.58 0.56 {z:.4f}" euler="0 0 0">',
        f'<freejoint name="{name}_freejoint"/>',
        f'<body name="{name}_seg_05">',
        geom(f"{name}_seg_05", material, WIDTHS[5]),
        f'<site name="{name}_pick_site" pos="0 0 0.00450" type="sphere" size="0.007" rgba="0 1 0 0.7"/>',
        f'<site name="{name}_center_site" pos="0 0 0.00350" type="sphere" size="0.006" rgba="1 0 0 0.6"/>',
    ]
    center.extend(indent(branch(name, material, [6, 7, 8, 9, 10], 1.0, [6, 7, 8, 9, 10]), 2))
    center.extend(indent(branch(name, material, [4, 3, 2, 1, 0], -1.0, [5, 4, 3, 2, 1]), 2))
    center.extend(["</body>", "</body>"])
    return center


def main() -> None:
    lines = [
        "<?xml version='1.0' encoding='utf-8'?>",
        '<mujoco model="compliant bamboo leaf pair">',
        "  <asset>",
        '    <material name="compliant_bamboo_mat_bottom" rgba="0.62 0.46 0.18 1" specular="0.08" shininess="0.12"/>',
        '    <material name="compliant_bamboo_mat_top" rgba="0.76 0.58 0.24 1" specular="0.08" shininess="0.12"/>',
        "  </asset>",
        "  <worldbody>",
    ]
    lines.extend(indent(leaf("staged_bamboo_leaf_bottom", 0.4952, "compliant_bamboo_mat_bottom"), 4))
    lines.extend(indent(leaf("staged_bamboo_leaf_top", 0.4986, "compliant_bamboo_mat_top"), 4))
    lines.extend(["  </worldbody>", "</mujoco>", ""])
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated: {OUTPUT}")


if __name__ == "__main__":
    main()
