# Winejar MuJoCo

MuJoCo simulation for the wine-jar loading project. The repository contains
the xArm6 models, wine-jar scenes, vacuum and tie-gun end effectors, production
demo animation, vision-driven simulation scripts, and validation tools.

## Environment

```bash
conda env create -f environment.yml
conda activate winejar-mujoco
```

## Production animation

```bash
python scripts/02_production_demo_animation.py --viewer --realtime --hold-open
```

The production scene models a two-station, three-jar line: the upstream dual-
vacuum xArm loads two crossed bamboo leaves, then the jar indexes downstream to
the tie-gun xArm. From the second jar onward, loading and tying run on the same
simulation clock. Paper labels are not part of this revised production flow.

Record a side-view preview:

```bash
python scripts/03_record_production_demo_videos.py \
  --camera side_overview_camera --playback-speed 0.4
```

## Validation

Run commands from the repository root:

```bash
PYTHONPATH=. python test/inspect_production_demo_animation.py
PYTHONPATH=. python test/inspect_winejar_production_demo_scene.py
```

Generated frames, diagnostics, and videos are written under `data/` and are
not committed.
