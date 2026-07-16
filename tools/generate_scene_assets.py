#!/usr/bin/env python3
from __future__ import annotations

import math
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "scene" / "assets" / "materials"


def normal(a, b, c):
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length == 0:
        return (0.0, 0.0, 1.0)
    return (nx / length, ny / length, nz / length)


def write_binary_stl(path: Path, name: str, triangles):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"{name} generated for wine jar MuJoCo scene".encode("ascii")[:80]
    header = header + b" " * (80 - len(header))
    with path.open("wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(triangles)))
        for a, b, c in triangles:
            nx, ny, nz = normal(a, b, c)
            values = (
                nx,
                ny,
                nz,
                a[0],
                a[1],
                a[2],
                b[0],
                b[1],
                b[2],
                c[0],
                c[1],
                c[2],
            )
            f.write(struct.pack("<12fH", *values, 0))


def generate_wine_jar(path: Path):
    inner_mouth_radius = 0.0525
    profile = [
        (0.00, 0.125),
        (0.08, 0.140),
        (0.20, 0.155),
        (0.35, 0.160),
        (0.41, 0.110),
        (0.445, 0.060),
        (0.46, 0.060),
    ]
    segments = 96
    triangles = []

    rings = []
    for z, r in profile:
        ring = []
        for i in range(segments):
            theta = 2.0 * math.pi * i / segments
            ring.append((r * math.cos(theta), r * math.sin(theta), z))
        rings.append(ring)

    for j in range(len(rings) - 1):
        lower = rings[j]
        upper = rings[j + 1]
        for i in range(segments):
            ni = (i + 1) % segments
            triangles.append((lower[i], lower[ni], upper[ni]))
            triangles.append((lower[i], upper[ni], upper[i]))

    bottom_center = (0.0, 0.0, profile[0][0])
    for i in range(segments):
        ni = (i + 1) % segments
        triangles.append((bottom_center, rings[0][i], rings[0][ni]))

    inner_bottom = []
    inner_top = []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        x = inner_mouth_radius * math.cos(theta)
        y = inner_mouth_radius * math.sin(theta)
        inner_bottom.append((x, y, 0.445))
        inner_top.append((x, y, 0.46))

    outer_bottom = rings[-2]
    outer_top = rings[-1]
    for i in range(segments):
        ni = (i + 1) % segments
        triangles.append((inner_bottom[i], inner_top[ni], inner_bottom[ni]))
        triangles.append((inner_bottom[i], inner_top[i], inner_top[ni]))
        triangles.append((outer_top[i], inner_top[i], inner_top[ni]))
        triangles.append((outer_top[i], inner_top[ni], outer_top[ni]))
        triangles.append((outer_bottom[i], inner_bottom[ni], inner_bottom[i]))
        triangles.append((outer_bottom[i], outer_bottom[ni], inner_bottom[ni]))

    write_binary_stl(path, "wine_jar", triangles)


def generate_lotus_leaf(path: Path):
    radius = 0.22
    thickness = 0.004
    angle0 = -math.pi / 4.0
    angle1 = math.pi / 4.0
    segments = 32
    triangles = []

    top_center = (0.0, 0.0, thickness / 2.0)
    bottom_center = (0.0, 0.0, -thickness / 2.0)
    top_arc = []
    bottom_arc = []
    for i in range(segments + 1):
        t = angle0 + (angle1 - angle0) * i / segments
        top_arc.append((radius * math.cos(t), radius * math.sin(t), thickness / 2.0))
        bottom_arc.append((radius * math.cos(t), radius * math.sin(t), -thickness / 2.0))

    for i in range(segments):
        triangles.append((top_center, top_arc[i], top_arc[i + 1]))
        triangles.append((bottom_center, bottom_arc[i + 1], bottom_arc[i]))
        triangles.append((top_arc[i], bottom_arc[i], bottom_arc[i + 1]))
        triangles.append((top_arc[i], bottom_arc[i + 1], top_arc[i + 1]))

    triangles.append((top_center, bottom_center, bottom_arc[0]))
    triangles.append((top_center, bottom_arc[0], top_arc[0]))
    triangles.append((top_center, top_arc[-1], bottom_arc[-1]))
    triangles.append((top_center, bottom_arc[-1], bottom_center))

    write_binary_stl(path, "lotus_leaf", triangles)


def generate_bamboo_leaf(path: Path):
    length = 0.42
    half_length = length / 2.0
    half_width = 0.0625
    thickness = 0.003
    segments = 48
    triangles = []

    top_center = (0.0, 0.0, thickness / 2.0)
    bottom_center = (0.0, 0.0, -thickness / 2.0)
    top = []
    bottom = []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        x = half_length * math.cos(theta)
        y = half_width * math.sin(theta)
        top.append((x, y, thickness / 2.0))
        bottom.append((x, y, -thickness / 2.0))

    for i in range(segments):
        ni = (i + 1) % segments
        triangles.append((top_center, top[i], top[ni]))
        triangles.append((bottom_center, bottom[ni], bottom[i]))
        triangles.append((top[i], bottom[i], bottom[ni]))
        triangles.append((top[i], bottom[ni], top[ni]))

    write_binary_stl(path, "bamboo_leaf", triangles)


def generate_extruded_polygon(path: Path, name: str, points, thickness: float):
    triangles = []
    top = [(x, y, thickness / 2.0) for x, y in points]
    bottom = [(x, y, -thickness / 2.0) for x, y in points]
    top_center = (
        sum(x for x, _ in points) / len(points),
        sum(y for _, y in points) / len(points),
        thickness / 2.0,
    )
    bottom_center = (top_center[0], top_center[1], -thickness / 2.0)

    for i in range(len(points)):
        ni = (i + 1) % len(points)
        triangles.append((top_center, top[i], top[ni]))
        triangles.append((bottom_center, bottom[ni], bottom[i]))
        triangles.append((top[i], bottom[i], bottom[ni]))
        triangles.append((top[i], bottom[ni], top[ni]))

    write_binary_stl(path, name, triangles)


def generate_cropped_lotus_leaf(path: Path):
    points = [
        (-0.108, -0.055),
        (-0.030, -0.086),
        (0.098, -0.060),
        (0.116, 0.020),
        (0.032, 0.080),
        (-0.092, 0.052),
    ]
    generate_extruded_polygon(path, "cropped_lotus_leaf", points, 0.004)


def generate_cropped_white_paper(path: Path):
    points = [
        (-0.082, -0.050),
        (0.078, -0.052),
        (0.092, 0.034),
        (0.014, 0.072),
        (-0.074, 0.034),
    ]
    generate_extruded_polygon(path, "cropped_white_paper", points, 0.003)


def main():
    generate_wine_jar(OUT_DIR / "wine_jar.stl")
    generate_lotus_leaf(OUT_DIR / "lotus_leaf.stl")
    generate_bamboo_leaf(OUT_DIR / "bamboo_leaf.stl")
    generate_cropped_lotus_leaf(OUT_DIR / "cropped_lotus_leaf.stl")
    generate_cropped_white_paper(OUT_DIR / "cropped_white_paper.stl")
    print(f"Generated scene assets in {OUT_DIR}")


if __name__ == "__main__":
    main()
