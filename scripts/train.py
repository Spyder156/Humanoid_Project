"""Train an rsl_rl PPO policy on an Isaac Lab task.

Usage:
    python scripts/train.py                                   # G1 flat velocity, headless
    python scripts/train.py --num_envs 2048 --max_iterations 1000
    tensorboard --logdir logs/rsl_rl                          # watch curves
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-v0")
parser.add_argument("--num_envs", type=int, default=None, help="override task default (4096)")
parser.add_argument("--max_iterations", type=int, default=None, help="override agent default")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
from datetime import datetime

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401  (registers Isaac-* tasks)
import humanoid.sim.tasks  # noqa: F401  (registers Humanoid-* tasks)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def main():
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = agent_cfg.seed

    log_dir = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    print(f"[train] {args.task}: {env.num_envs} envs, {agent_cfg.max_iterations} iters -> {log_dir}", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()
    print(f"[train] TRAIN_DONE -> {log_dir}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
