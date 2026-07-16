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

## Validation

Run commands from the repository root:

```bash
PYTHONPATH=. python test/inspect_production_demo_animation.py
PYTHONPATH=. python test/inspect_winejar_production_demo_scene.py
```

Generated frames, diagnostics, and videos are written under `data/` and are
not committed.
