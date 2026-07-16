from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SCENE_PATH = ROOT / "scene" / "scene_winejar_production_demo.xml"
OUTPUT_ROOT = ROOT / "data" / "production_demo_animation"

LEFT_HOME = np.radians([0.0, -35.0, -70.0, 0.0, 105.0, 0.0])
RIGHT_SIDE_STANDBY = np.radians([0.0, -77.31, -69.98, 0.0, 147.29, 0.0])
ROBOT_SPEED_SCALE = 2.0

JAR = "station_wine_jar"

FINAL_TIE_PREFIX = "final_tie_band_visual_geom_"
LOADED_TIE_PREFIX = "right_tie_gun_loaded_band_"
OPEN_JAW_PREFIXES = ("right_tie_gun_left_open_jaw_", "right_tie_gun_right_open_jaw_")
CLOSED_JAW_PREFIX = "right_tie_gun_closed_jaw_"
TIE_GUN_EXTENSION_GEOM_PREFIXES = (
    "right_tie_gun_left_open_jaw_",
    "right_tie_gun_right_open_jaw_",
    "right_tie_gun_closed_jaw_",
    "right_tie_gun_loaded_band_",
)
TIE_GUN_EXTENSION_GEOM_NAMES: set[str] = set()
TIE_GUN_EXTENSION_SITE_NAMES = {
    "right_tie_gun_ring_visual_site",
    "right_tie_gun_neck_target_site",
}

CONVEYOR_TRANSFER_SECONDS = 4.0
LEAF_SETTLE_STEPS = 15
LEAF_PLACE_WAYPOINTS = 6
TIE_GUN_HOLD_SECONDS = 1.0
LEAF_PLACE_CLEARANCE_M = 0.050
BOTTOM_LEAF_PLACE_CLEARANCE_M = 0.050
