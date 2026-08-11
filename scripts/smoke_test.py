"""Smoke test: Unitree G1 on flat terrain in Isaac Lab, states logged to Rerun.

Zero actions -> joint targets hold the default pose, so the robot should
roughly stand/stagger. Verifies the full sim + logging stack end to end.

Usage:
    python scripts/smoke_test.py                # 16 envs, 300 steps
    rerun outputs/smoke_test.rrd                # inspect result
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--out", type=str, default="outputs/smoke_test.rrd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401  (registers Isaac-* tasks)
from isaaclab_tasks.utils import parse_env_cfg

from humanoid.viz.rerun_logger import HumanoidRerunLogger


def main():
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    logger = HumanoidRerunLogger("humanoid_smoke_test", save_path=args.out)
    print(f"[smoke] {args.task}: {args.num_envs} envs, {robot.num_bodies} bodies, {robot.num_joints} joints")

    actions = torch.zeros(args.num_envs, env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device)
    for step in range(args.steps):
        env.step(actions)
        logger.log_state(
            step,
            body_pos=robot.data.body_pos_w[0].cpu().numpy(),
            body_names=robot.data.body_names,
            root_pos=robot.data.root_pos_w[0].cpu().numpy(),
            joint_pos=robot.data.joint_pos[0].cpu().numpy(),
            joint_names=robot.data.joint_names,
        )

    env.close()
    print(f"[smoke] OK — wrote {args.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
