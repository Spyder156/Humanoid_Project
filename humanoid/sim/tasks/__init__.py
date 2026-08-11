"""Task registrations. Import this module to make Humanoid-* gym ids available."""

import gymnasium as gym

gym.register(
    id="Humanoid-G1-Perception-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_perception:G1PerceptionEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg"
        ),
    },
)
