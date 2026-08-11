"""Rerun logging for humanoid sim states.

Every experiment logs through here — keep entity paths stable so recordings
stay comparable across runs.
"""

from __future__ import annotations

import numpy as np
import rerun as rr


class HumanoidRerunLogger:
    """Logs one environment's robot state to a .rrd recording."""

    def __init__(self, app_id: str, save_path: str | None = None, spawn: bool = False):
        rr.init(app_id, spawn=spawn)
        if save_path is not None:
            rr.save(save_path)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    def log_state(
        self,
        step: int,
        body_pos: np.ndarray,  # (num_bodies, 3), world frame
        body_names: list[str] | None = None,
        root_pos: np.ndarray | None = None,  # (3,)
        joint_pos: np.ndarray | None = None,  # (num_joints,)
        joint_names: list[str] | None = None,
    ) -> None:
        rr.set_time("step", sequence=step)
        rr.log("world/robot/bodies", rr.Points3D(body_pos, labels=body_names, radii=0.02))
        if root_pos is not None:
            rr.log("world/robot/root", rr.Points3D(root_pos[None], colors=[255, 80, 80], radii=0.04))
            rr.log("plots/root_height", rr.Scalars(float(root_pos[2])))
        if joint_pos is not None:
            names = joint_names or [f"joint_{i}" for i in range(len(joint_pos))]
            for name, q in zip(names, joint_pos):
                rr.log(f"plots/joints/{name}", rr.Scalars(float(q)))
