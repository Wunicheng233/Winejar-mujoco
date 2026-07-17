#!/usr/bin/env python3
"""Generate repeatable MuJoCo fragments for the three-jar production line.

The source leaf pair is intentionally kept as the single authoritative model.
This generator creates the two additional, independently simulated leaf pairs
and the two downstream jar bodies with names that the animation scheduler can
address explicitly.
"""
from __future__ import annotations

import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEAF_SOURCE = ROOT / "scene" / "compliant_bamboo_leaf_pair.xml"
OUTPUT = ROOT / "scene" / "production_line_instances.xml"


def source_worldbody() -> str:
    text = LEAF_SOURCE.read_text(encoding="utf-8")
    start = text.index("  <worldbody>") + len("  <worldbody>")
    end = text.index("  </worldbody>", start)
    return text[start:end].strip()


def renamed_leaf_pair(index: int, bottom_z: float, top_z: float) -> str:
    text = source_worldbody()
    replacements = {
        "staged_bamboo_leaf_bottom": f"jar_{index:02d}_bamboo_leaf_bottom",
        "staged_bamboo_leaf_top": f"jar_{index:02d}_bamboo_leaf_top",
        'pos="-0.58 0.56 0.4952"': f'pos="-0.58 0.5600 {bottom_z:.4f}"',
        'pos="-0.58 0.56 0.4986"': f'pos="-0.58 0.5600 {top_z:.4f}"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def cover_layer(index: int, layer: str, z: float, yaw: float, material: str) -> str:
    prefix = f"jar_{index:02d}_preloaded_{layer}"
    body_name = f"jar_{index:02d}_preloaded_{'lotus_leaf' if layer == 'lotus' else 'white_paper'}"
    mass_scale = 1.0 if layer == "lotus" else 0.5
    fold_radius = 0.044 if layer == "lotus" else 0.048
    mesh_prefix = f"preloaded_{layer}"
    return f'''      <body name="{body_name}" pos="0 0 {z:.3f}" euler="0 0 {yaw:.2f}">
        <geom name="{body_name}_geom" type="mesh" mesh="{mesh_prefix}_center_mesh" material="{material}" mass="{0.008 * mass_scale:.4f}" friction="2 1 1" contype="4" conaffinity="5"/>
        <body name="{prefix}_east_flap" pos="{fold_radius:.3f} 0 0">
          <joint name="{prefix}_fold_east" type="hinge" axis="0 1 0" range="0 1.57" limited="true" damping="0.01"/>
          <geom name="{prefix}_east_flap_geom" type="mesh" mesh="{mesh_prefix}_east_mesh" material="{material}" mass="{0.003 * mass_scale:.4f}" friction="2 1 1" contype="4" conaffinity="5"/>
        </body>
        <body name="{prefix}_west_flap" pos="{-fold_radius:.3f} 0 0">
          <joint name="{prefix}_fold_west" type="hinge" axis="0 -1 0" range="0 1.57" limited="true" damping="0.01"/>
          <geom name="{prefix}_west_flap_geom" type="mesh" mesh="{mesh_prefix}_west_mesh" material="{material}" mass="{0.003 * mass_scale:.4f}" friction="2 1 1" contype="4" conaffinity="5"/>
        </body>
        <body name="{prefix}_north_flap" pos="0 {fold_radius:.3f} 0">
          <joint name="{prefix}_fold_north" type="hinge" axis="-1 0 0" range="0 1.57" limited="true" damping="0.01"/>
          <geom name="{prefix}_north_flap_geom" type="mesh" mesh="{mesh_prefix}_north_mesh" material="{material}" mass="{0.002 * mass_scale:.4f}" friction="2 1 1" contype="4" conaffinity="5"/>
        </body>
        <body name="{prefix}_south_flap" pos="0 {-fold_radius:.3f} 0">
          <joint name="{prefix}_fold_south" type="hinge" axis="1 0 0" range="0 1.57" limited="true" damping="0.01"/>
          <geom name="{prefix}_south_flap_geom" type="mesh" mesh="{mesh_prefix}_south_mesh" material="{material}" mass="{0.002 * mass_scale:.4f}" friction="2 1 1" contype="4" conaffinity="5"/>
        </body>
      </body>'''


def final_tie_band(index: int) -> str:
    geoms = []
    radius = 0.063
    for segment in range(16):
        angle_a = 2.0 * math.pi * segment / 16
        angle_b = 2.0 * math.pi * (segment + 1) / 16
        start = (radius * math.cos(angle_a), radius * math.sin(angle_a), 0.0)
        end = (radius * math.cos(angle_b), radius * math.sin(angle_b), 0.0)
        fromto = " ".join(f"{value:.5f}" for value in (*start, *end))
        geoms.append(
            f'        <geom name="jar_{index:02d}_final_tie_band_visual_geom_{segment:02d}" '
            f'type="capsule" fromto="{fromto}" size="0.0022" '
            'rgba="0.96 0.96 0.93 0" contype="0" conaffinity="0"/>'
        )
    return (
        f'      <body name="jar_{index:02d}_final_tie_band_visual" pos="0 0 0.452">\n'
        + "\n".join(geoms)
        + "\n      </body>"
    )


def jar_body(index: int) -> str:
    jar = f"station_wine_jar_{index:02d}"
    covers = "\n".join(
        (
            cover_layer(index, "lotus", 0.462, 0.15, "lotus_mat"),
            cover_layer(index, "paper", 0.467, 0.15, "paper_mat"),
        )
    )
    tie_band = final_tie_band(index)
    return f'''    <body name="{jar}" pos="-2.40 0.05 0.125">
      <geom name="{jar}_visual" type="mesh" mesh="wine_jar_mesh" material="jar_mat" contype="0" conaffinity="0"/>
      <geom name="{jar}_belly_collision" type="cylinder" pos="0 0 0.205" size="0.16 0.205"
            material="jar_collision_mat" mass="8" friction="1 1 1" contype="2" conaffinity="1"/>
      <geom name="{jar}_shoulder_lower_collision" type="cylinder" pos="0 0 0.420" size="0.097 0.010"
            material="jar_collision_mat" mass="1" friction="1 1 1" contype="2" conaffinity="1"/>
      <geom name="{jar}_shoulder_middle_collision" type="cylinder" pos="0 0 0.43375" size="0.071 0.00375"
            material="jar_collision_mat" mass="0.5" friction="1 1 1" contype="2" conaffinity="1"/>
      <geom name="{jar}_shoulder_upper_collision" type="cylinder" pos="0 0 0.44125" size="0.068 0.00375"
            material="jar_collision_mat" mass="0.5" friction="1 1 1" contype="2" conaffinity="1"/>
      <geom name="{jar}_neck_collision" type="cylinder" pos="0 0 0.4525" size="0.06 0.0075"
            material="jar_collision_mat" mass="1" friction="1 1 1" contype="2" conaffinity="1"/>
      <site name="jar_{index:02d}_mouth_center" pos="0 0 0.46" type="sphere" size="0.010" rgba="1 0 0 0"/>
      <site name="jar_{index:02d}_neck_tie_target" pos="0 0 0.452" type="sphere" size="0.010" rgba="0 0.7 1 0"/>
{covers}
      <body name="jar_{index:02d}_preloaded_ceramic_disc" pos="0 0 0.476">
        <geom name="jar_{index:02d}_preloaded_ceramic_disc_geom" type="cylinder" size="0.055 0.006" material="ceramic_mat" mass="0.20" friction="1 1 1"/>
      </body>
      <geom name="jar_{index:02d}_mouth_support" type="cylinder" pos="0 0 0.488" size="0.130 0.002"
            rgba="0.2 0.6 0.2 0" contype="2" conaffinity="1" friction="8 3 2"/>
{tie_band}
    </body>'''


def equality_for_pair(index: int) -> str:
    jar = f"station_wine_jar_{index:02d}"
    prefix = f"jar_{index:02d}_bamboo_leaf"
    return f'''    <weld name="left_suction_weld_jar_{index:02d}_leaf_top" body1="left_vacuum_end_effector" body2="{prefix}_top_seg_05" active="false" solref="0.004 1" solimp="0.95 0.99 0.001"/>
    <weld name="left_suction_weld_jar_{index:02d}_leaf_bottom" body1="left_vacuum_end_effector" body2="{prefix}_bottom_seg_05" active="false" solref="0.004 1" solimp="0.95 0.99 0.001"/>
    <weld name="table_static_friction_weld_jar_{index:02d}_leaf_top" body1="left_material_table" body2="{prefix}_top_seg_05" active="false" solref="0.030 1" solimp="0.70 0.95 0.010"/>
    <weld name="table_static_friction_weld_jar_{index:02d}_leaf_bottom" body1="left_material_table" body2="{prefix}_bottom_seg_05" active="false" solref="0.030 1" solimp="0.70 0.95 0.010"/>
    <weld name="mouth_static_friction_weld_jar_{index:02d}_leaf_top" body1="{jar}" body2="{prefix}_top_seg_05" active="false" solref="0.030 1" solimp="0.70 0.95 0.010"/>
    <weld name="mouth_static_friction_weld_jar_{index:02d}_leaf_bottom" body1="{jar}" body2="{prefix}_bottom_seg_05" active="false" solref="0.030 1" solimp="0.70 0.95 0.010"/>'''


def main() -> None:
    # All six leaves form one physical material stack.  The first jar consumes
    # the top pair, followed by the middle and then the bottom pair.
    leaves = "\n\n".join(
        renamed_leaf_pair(index, bottom_z, top_z)
        for index, bottom_z, top_z in ((2, 0.4884, 0.4918), (3, 0.4816, 0.4850))
    )
    jars = "\n\n".join(jar_body(index) for index in (2, 3))
    equalities = "\n".join(equality_for_pair(index) for index in (2, 3))
    OUTPUT.write_text(
        "<mujoco model=\"production line instances\">\n"
        "  <worldbody>\n"
        f"{leaves}\n\n{jars}\n"
        "  </worldbody>\n"
        "  <equality>\n"
        f"{equalities}\n"
        "  </equality>\n"
        "</mujoco>\n",
        encoding="utf-8",
    )
    print(f"Generated: {OUTPUT}")


if __name__ == "__main__":
    main()
