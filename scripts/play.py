"""Roll out a trained rsl_rl policy and record the run to Rerun.

Usage:
    python scripts/play.py                          # latest checkpoint of default task
    python scripts/play.py --checkpoint logs/rsl_rl/g1_flat/<run>/model_999.pt
    rerun outputs/policy_rollout.rrd
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-v0")
parser.add_argument("--checkpoint", type=str, default=None, help="path to model_*.pt; default = latest run")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--out", type=str, default="outputs/policy_rollout.rrd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import humanoid.sim.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

from humanoid.viz.rerun_logger import HumanoidRerunLogger


def main():
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)

    if args.checkpoint:
        ckpt = args.checkpoint
    else:
        import glob

        candidates = sorted(glob.glob(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, "*", "model_*.pt")))
        if not candidates:
            raise FileNotFoundError(f"no checkpoints under logs/rsl_rl/{agent_cfg.experiment_name}")
        # latest run dir, highest iteration within it
        latest_run = os.path.dirname(candidates[-1])
        ckpt = max(glob.glob(os.path.join(latest_run, "model_*.pt")),
                   key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    print(f"[play] checkpoint: {ckpt}", flush=True)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    robot = env.unwrapped.scene["robot"]
    cam = getattr(env.unwrapped.scene, "sensors", {}).get("front_cam")
    logger = HumanoidRerunLogger(
        "g1_policy_rollout", save_path=args.out,
        camera_entity="world/robot/front_cam" if cam is not None else None,
    )

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    for step in range(args.steps):
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        logger.log_state(
            step,
            body_pos=robot.data.body_pos_w[0].cpu().numpy(),
            body_names=robot.data.body_names,
            root_pos=robot.data.root_pos_w[0].cpu().numpy(),
            joint_pos=robot.data.joint_pos[0].cpu().numpy(),
            joint_names=robot.data.joint_names,
        )
        if cam is not None:
            logger.log_camera(
                step,
                cam_pos=cam.data.pos_w[0].cpu().numpy(),
                cam_quat_ros=cam.data.quat_w_ros[0].cpu().numpy(),
                intrinsics=cam.data.intrinsic_matrices[0].cpu().numpy(),
                rgb=cam.data.output["rgb"][0].cpu().numpy(),
                depth=cam.data.output["distance_to_image_plane"][0].squeeze(-1).cpu().numpy(),
            )

    env.close()
    print(f"[play] PLAY_DONE — wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
