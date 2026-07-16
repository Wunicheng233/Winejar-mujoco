#!/usr/bin/env python3
from __future__ import annotations

import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BAMBOO_LEAF_PATH = ROOT / "scene" / "assets" / "materials" / "bamboo_leaf.stl"
WINE_JAR_PATH = ROOT / "scene" / "assets" / "materials" / "wine_jar.stl"


def read_binary_stl_vertices(path: Path):
    data = path.read_bytes()
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    offset = 84
    vertices = []
    for _ in range(triangle_count):
        values = struct.unpack_from("<12fH", data, offset)
        vertices.extend(
            [
                values[3:6],
                values[6:9],
                values[9:12],
            ]
        )
        offset += 50
    return vertices


def axis_span(vertices, axis: int) -> float:
    values = [vertex[axis] for vertex in vertices]
    return max(values) - min(values)


def diameter_at_z(vertices, z: float, tolerance: float = 1e-5) -> float:
    points = [vertex for vertex in vertices if abs(vertex[2] - z) <= tolerance]
    if not points:
        raise AssertionError(f"no vertices found at z={z:.6f}")
    radii = [(vertex[0] ** 2 + vertex[1] ** 2) ** 0.5 for vertex in points]
    return 2.0 * max(radii)


def assert_close(name: str, actual: float, expected: float, tolerance: float = 1e-5):
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{name}: expected {expected:.6f}, got {actual:.6f}")


def main():
    bamboo_vertices = read_binary_stl_vertices(BAMBOO_LEAF_PATH)
    assert_close("bamboo leaf length", axis_span(bamboo_vertices, 0), 0.42)
    assert_close("bamboo leaf max width", axis_span(bamboo_vertices, 1), 0.125)

    jar_vertices = read_binary_stl_vertices(WINE_JAR_PATH)
    assert_close("wine jar height", axis_span(jar_vertices, 2), 0.46)
    assert_close("wine jar bottom diameter", diameter_at_z(jar_vertices, 0.00), 0.25)
    assert_close("wine jar belly diameter", diameter_at_z(jar_vertices, 0.35), 0.32)
    assert_close("wine jar mouth outer diameter", diameter_at_z(jar_vertices, 0.445), 0.12)
    assert_close("wine jar top outer diameter", diameter_at_z(jar_vertices, 0.46), 0.12)
    print("Material asset dimensions OK")


if __name__ == "__main__":
    main()
