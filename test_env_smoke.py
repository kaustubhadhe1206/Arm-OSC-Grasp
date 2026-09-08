# Quick smoke test for FrankaOSCGraspEnv — NOT a training run. Checks the
# env is well-formed (gymnasium's own checker), doesn't crash over many
# random-action steps/episodes, and reports basic reward/behavior stats so
# obviously broken reward shaping or a stuck arm would show up before
# committing to a full training run.
#
# Run: venv/Scripts/python.exe test_env_smoke.py

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))
from franka_osc_grasp_env import FrankaOSCGraspEnv


def main():
    env = FrankaOSCGraspEnv(use_camera=False)

    from gymnasium.utils.env_checker import check_env
    print("Running gymnasium's check_env...")
    check_env(env, skip_render_check=True)
    print("check_env passed.\n")

    n_episodes = 3
    steps_per_episode = 100

    for ep in range(n_episodes):
        obs, info = env.reset()
        assert env.observation_space.contains(obs), f"obs out of declared space: {obs}"

        rewards = []
        tip_positions = []
        for step in range(steps_per_episode):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs), f"obs out of declared space at step {step}: {obs}"
            assert np.isfinite(reward), f"non-finite reward at step {step}: {reward}"
            rewards.append(reward)
            tip_positions.append(env.data.site_xpos[env.tip_id].copy())
            if terminated or truncated:
                break

        rewards = np.array(rewards)
        tip_positions = np.array(tip_positions)
        tip_travel = np.linalg.norm(tip_positions[-1] - tip_positions[0]) if len(tip_positions) > 1 else 0.0
        print(f"Episode {ep}: steps={len(rewards)} reward_mean={rewards.mean():.4f} "
              f"reward_min={rewards.min():.4f} reward_max={rewards.max():.4f} "
              f"tip_net_travel={tip_travel*1000:.1f}mm terminated={terminated} truncated={truncated}")

    env.close()
    print("\nSmoke test passed — env runs without crashing, obs/reward stay within declared bounds.")


if __name__ == "__main__":
    main()
