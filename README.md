# Humanoid Project

Research on humanoid / physical intelligence: **3D perception and reconstruction in the loop of whole-body policies**, on top of Isaac Lab + Unitree G1.

## Stack

| Layer | What | Where |
|---|---|---|
| Sim | Isaac Sim 5.1 / Isaac Lab 2.3.2 (pip) | `humanoid/sim` |
| L0 control | rsl_rl PPO, whole-body policies | `humanoid/intelligence/control` |
| L1 skills | motion retargeting, tracking, skill latents | `humanoid/intelligence/skills` |
| L2 brain | VLA (GR00T-class), planners | `humanoid/intelligence/brain` |
| Perception | depth, point clouds, 3D reconstruction backends | `humanoid/perception` |
| Viz | Rerun loggers for everything | `humanoid/viz` |

## Quickstart

```bash
conda activate humanoid          # setup: docs/SETUP.md
python scripts/smoke_test.py     # G1 in sim -> outputs/smoke_test.rrd
rerun outputs/smoke_test.rrd     # view it
```

## Layout

```
humanoid/           python package (envs, perception, intelligence, viz)
scripts/            entrypoints: train / play / smoke_test
configs/            experiment configs
assets/             robots, scenes, mocap (gitignored; download scripts)
docker/             reproducible image (see docker/README.md)
docs/               setup notes, pinned known-good versions
```
