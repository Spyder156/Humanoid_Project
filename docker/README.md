# Docker

Pip-based Isaac Sim image (not the ~30GB NGC isaac-sim container).

```bash
# from repo root — build only when you need reproducibility, image is ~15GB
docker build -f docker/Dockerfile -t humanoid:latest .

# headless training / smoke test
docker run --gpus all --rm -it \
  -v $(pwd)/outputs:/workspace/outputs \
  -v $(pwd)/assets:/workspace/assets \
  humanoid:latest python scripts/smoke_test.py

# keep disk clean afterwards
docker image prune -f
```

Notes
- First run inside a fresh container warms Isaac Sim shader caches (minutes). Mount
  `-v ~/docker_cache/ov:/root/.cache/ov` to persist across runs.
- Rerun `.rrd` files land in `outputs/` (mounted) — view on host with `rerun outputs/<file>.rrd`.
