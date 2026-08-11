"""G1 velocity task with an egocentric RGB-D camera on the torso.

Base task is Isaac Lab's G1 flat-terrain velocity env; we add a forward-facing
tiled camera (pitched ~12 deg down) so perception modules get depth + RGB
streams that can be backprojected into 3D.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import MySceneCfg


@configclass
class G1PerceptionSceneCfg(MySceneCfg):
    front_cam: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/front_cam",
        # "world" convention: +X forward; quat = 12 deg pitch down about +Y
        offset=TiledCameraCfg.OffsetCfg(pos=(0.12, 0.0, 0.42), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)),
        width=160,
        height=120,
        data_types=["rgb", "distance_to_image_plane"],
    )


@configclass
class G1PerceptionEnvCfg(G1FlatEnvCfg):
    scene: G1PerceptionSceneCfg = G1PerceptionSceneCfg(num_envs=64, env_spacing=2.5)
