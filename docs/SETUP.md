# Environment setup

Known-good on RTX 5070 Ti (sm_120, Blackwell), driver 580.x, CUDA 12.8.

## Conda (dev)

```bash
conda create -n humanoid python=3.11 -y
conda activate humanoid
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaaclab[isaacsim,all]==2.3.2" --extra-index-url https://pypi.nvidia.com
pip install -e .
```

`isaaclab[isaacsim,all]` brings Isaac Sim 5.1 (pip wheels), rsl-rl-lib, skrl, sb3, rl_games, hydra.

First Isaac Sim launch accepts the EULA and warms shader/asset caches — takes several minutes:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

## Version pins

- Python 3.11 / torch 2.7.0+cu128 (sm_120 wheels) / isaaclab 2.3.2 / isaacsim 5.1.0
- Full reference freeze from a previously working Isaac Sim 5.1 setup:
  [reference/known_good_isaacsim51_freeze.txt](reference/known_good_isaacsim51_freeze.txt)

## Docker

See [../docker/README.md](../docker/README.md). Image is written but not built by default
(Isaac Sim layers are ~15 GB — build deliberately, prune after experiments).
