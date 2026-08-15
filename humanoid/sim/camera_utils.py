"""Camera pose utilities.

Isaac Sim 5.x runs physics in fabric and does not sync poses back to USD, so
TiledCamera.data.pos_w / quat_w_* stay frozen at the pre-randomization spawn
pose (verified: robot falls, sensor pose never moves). Never log those.
Instead, compute the camera world pose from the articulation body state, which
is PhysX-backed and always current.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import combine_frame_transforms, quat_mul

# Rotation from Isaac's "world" camera convention (+X forward, +Y left, +Z up)
# to the optical/RDF frame Rerun's Pinhole expects (+Z forward, +X right, +Y down).
_WORLDCONV_TO_OPTICAL_WXYZ = (0.5, -0.5, 0.5, -0.5)


def body_mounted_camera_pose(
    body_pos_w: torch.Tensor,  # (N, 3) mount body position, world frame
    body_quat_w: torch.Tensor,  # (N, 4) wxyz mount body orientation, world frame
    offset_pos: tuple[float, float, float],
    offset_rot_worldconv: tuple[float, float, float, float],  # wxyz, "world" convention (+X forward)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Camera world pose from its mount body's state.

    Returns (pos (N, 3), quat_optical (N, 4) wxyz) — orientation in the
    optical/RDF convention, ready for Rerun's Pinhole.
    """
    n = body_pos_w.shape[0]
    device = body_pos_w.device
    off_p = torch.tensor(offset_pos, device=device).expand(n, 3)
    off_q = torch.tensor(offset_rot_worldconv, device=device).expand(n, 4)
    cam_pos, cam_quat_wc = combine_frame_transforms(body_pos_w, body_quat_w, off_p, off_q)
    to_opt = torch.tensor(_WORLDCONV_TO_OPTICAL_WXYZ, device=device).expand(n, 4)
    cam_quat_opt = quat_mul(cam_quat_wc, to_opt)
    return cam_pos, cam_quat_opt
