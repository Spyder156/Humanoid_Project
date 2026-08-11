"""Rerun logging for humanoid sim states.

Every experiment logs through here — keep entity paths stable so recordings
stay comparable across runs.
"""

from __future__ import annotations

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


class HumanoidRerunLogger:
    """Logs one environment's robot state to a .rrd recording."""

    def __init__(self, app_id: str, save_path: str | None = None, spawn: bool = False,
                 camera_entity: str | None = None):
        rr.init(app_id, spawn=spawn)
        if save_path is not None:
            rr.save(save_path)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        # embed a layout so camera panels show up without manual viewer setup
        main = rrb.Spatial3DView(origin="world", name="scene")
        if camera_entity is not None:
            side = rrb.Vertical(
                rrb.Spatial2DView(origin=f"{camera_entity}/rgb", name="rgb"),
                rrb.Spatial2DView(origin=f"{camera_entity}/depth", name="depth"),
                rrb.TimeSeriesView(origin="plots/root_height", name="root height"),
            )
            layout = rrb.Horizontal(main, side, column_shares=[3, 1])
        else:
            layout = rrb.Horizontal(main, rrb.TimeSeriesView(origin="plots", name="plots"), column_shares=[3, 1])
        rr.send_blueprint(rrb.Blueprint(layout, collapse_panels=True))

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

    def log_camera(
        self,
        step: int,
        cam_pos: np.ndarray,  # (3,), world frame
        cam_quat_ros: np.ndarray,  # (4,) wxyz, ROS/OpenCV optical convention (+Z forward)
        intrinsics: np.ndarray,  # (3, 3)
        rgb: np.ndarray | None = None,  # (H, W, 3) uint8
        depth: np.ndarray | None = None,  # (H, W) float meters; invalid as inf/nan
        entity: str = "world/robot/front_cam",
    ) -> None:
        """Log a posed pinhole camera; the Rerun viewer backprojects depth into 3D."""
        rr.set_time("step", sequence=step)
        w, x, y, z = cam_quat_ros
        rr.log(entity, rr.Transform3D(translation=cam_pos, rotation=rr.Quaternion(xyzw=[x, y, z, w])))
        h_res, w_res = (depth.shape[:2] if depth is not None else rgb.shape[:2])
        rr.log(entity, rr.Pinhole(image_from_camera=intrinsics, width=w_res, height=h_res,
                                  camera_xyz=rr.ViewCoordinates.RDF))
        if rgb is not None:
            rr.log(f"{entity}/rgb", rr.Image(rgb).compress(jpeg_quality=80))
        if depth is not None:
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            rr.log(f"{entity}/depth", rr.DepthImage(depth, meter=1.0))
