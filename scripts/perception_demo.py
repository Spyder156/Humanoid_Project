"""Perception smoke test: G1 with egocentric RGB-D camera, streamed to Rerun.

The Rerun viewer backprojects the depth image through the pinhole model, so the
recording contains a live 3D point cloud from the robot's viewpoint alongside
the body-state logging.

Usage:
    python scripts/perception_demo.py
    rerun outputs/g1_perception.rrd
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Humanoid-G1-Perception-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--out", type=str, default="outputs/g1_perception.rrd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import humanoid.sim.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from humanoid.viz.rerun_logger import HumanoidRerunLogger


def main():
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    cam = env.unwrapped.scene["front_cam"]
    logger = HumanoidRerunLogger("g1_perception_demo", save_path=args.out)
    print(f"[percep] {args.num_envs} envs, cam {cam.image_shape}", flush=True)

    actions = torch.zeros(args.num_envs, env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device)
    for step in range(args.steps):
        env.step(actions)
        logger.log_state(
            step,
            body_pos=robot.data.body_pos_w[0].cpu().numpy(),
            body_names=robot.data.body_names,
            root_pos=robot.data.root_pos_w[0].cpu().numpy(),
        )
        logger.log_camera(
            step,
            cam_pos=cam.data.pos_w[0].cpu().numpy(),
            cam_quat_ros=cam.data.quat_w_ros[0].cpu().numpy(),
            intrinsics=cam.data.intrinsic_matrices[0].cpu().numpy(),
            rgb=cam.data.output["rgb"][0].cpu().numpy(),
            depth=cam.data.output["distance_to_image_plane"][0].squeeze(-1).cpu().numpy(),
        )

    env.close()
    print(f"[percep] PERCEP_DONE — wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
