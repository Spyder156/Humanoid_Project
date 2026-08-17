"""Log an nvblox mapper's reconstruction (colored mesh + feature surface points) to Rerun.

Generic over any nvblox_torch Mapper — used to inspect mindmap's internal map
during eval, and later our own scene-memory backend.
"""

from __future__ import annotations

import numpy as np
import rerun as rr
import torch

_initialized = False


def _ensure_init(save_path: str) -> None:
    global _initialized
    if not _initialized:
        rr.init("nvblox_map", spawn=False)
        rr.save(save_path)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        _initialized = True


def pca_colors(features: torch.Tensor) -> np.ndarray:
    """Project high-dim features to RGB via PCA (robust 2-98% normalization)."""
    f = features.float()
    f = f - f.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(f, q=3)
    proj = f @ v[:, :3]
    lo = proj.quantile(0.02, dim=0)
    hi = proj.quantile(0.98, dim=0)
    cols = ((proj - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    return (cols * 255).byte().cpu().numpy()


def log_nvblox_map(
    step: int,
    mapper,  # nvblox_torch.mapper.Mapper
    save_path: str | None = None,
    entity: str = "world/nvblox",
    min_tsdf_weight: float = 1.0,
    max_points: int = 150_000,
) -> None:
    from nvblox_torch.mapper import QueryType

    if save_path is not None:
        _ensure_init(save_path)
    rr.set_time("step", sequence=step)

    for mid in range(mapper.num_mappers()):
        # Colored surface mesh
        try:
            mapper.update_color_mesh(mid)
            mesh = mapper.get_color_mesh(mid)
        except Exception:
            mesh = None
        if mesh is not None:
            m = mesh.to_open3d()
            verts = np.asarray(m.vertices)
            if len(verts) > 0:
                cols = np.asarray(m.vertex_colors)
                rr.log(
                    f"{entity}/m{mid}/mesh",
                    rr.Mesh3D(
                        vertex_positions=verts,
                        triangle_indices=np.asarray(m.triangles),
                        vertex_colors=(cols * 255).astype(np.uint8) if len(cols) else None,
                    ),
                )

        # Feature surface point cloud (PCA -> RGB)
        try:
            tsdf_layer = mapper.tsdf_layer_view(mapper_id=mid)
            tsdf_and_w, pts = tsdf_layer.get_tsdfs_below_zero()
            if pts.shape[0] == 0:
                continue
            pts = pts[tsdf_and_w[:, 1] > min_tsdf_weight]
            if pts.shape[0] == 0:
                continue
            if pts.shape[0] > max_points:
                pts = pts[:: pts.shape[0] // max_points + 1]
            feats = mapper.query_layer(query_type=QueryType.FEATURE, query=pts, mapper_id=mid)[:, :-1]
            rr.log(
                f"{entity}/m{mid}/feature_points",
                rr.Points3D(pts.cpu().numpy(), colors=pca_colors(feats), radii=0.006),
            )
        except Exception as e:  # never break the eval for viz
            rr.log(f"{entity}/m{mid}/viz_error", rr.TextLog(f"{type(e).__name__}: {e}"))
