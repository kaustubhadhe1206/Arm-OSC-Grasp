# Collects successful grasp demonstrations by running the SAME kind of
# hand-scripted (no RL) routine as scripted_grasp_check.py across many
# episodes with randomized object positions, keeping only the FULL
# transition sequences (obs, action, reward, next_obs, done) from episodes
# that actually succeeded (terminated=True).
#
# Why: after 585k steps (1952 episodes) of pure RL, SAC had not once
# stumbled onto a successful grasp-lift-hold sequence, even though we'd
# already proven the task is achievable (scripted_grasp_check.py). This is
# a hard-exploration problem, not (only) a reward-shaping one — these
# demonstrations are meant to seed the replay buffer and pretrain the actor
# in train_osc_grasp_bc_parallel.py, so the agent starts from "I've seen
# this succeed" instead of hoping to randomly discover it.
#
# Run: venv/Scripts/python.exe collect_demonstrations.py [n_successes]

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))
from franka_osc_grasp_env import FrankaOSCGraspEnv


def move_to(env, target_xyz, target_gripper, max_steps, transitions, tol=0.01):
    for _ in range(max_steps):
        obs_before = env._get_obs()
        current_pos = env.target_pos
        current_gripper = env.data.qpos[env.gripper_qpos_addr]
        delta_pos = np.clip(target_xyz - current_pos, -env.max_delta_pos, env.max_delta_pos)
        delta_gripper = np.clip(target_gripper - current_gripper, -env.max_delta_gripper, env.max_delta_gripper)
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, delta_gripper], dtype=np.float32)
        obs_after, reward, terminated, truncated, info = env.step(action)
        transitions.append((obs_before, action, reward, obs_after, terminated or truncated))
        if terminated or truncated:
            return terminated
        tip_pos = env.data.site_xpos[env.tip_id]
        if np.linalg.norm(target_xyz - tip_pos) < tol:
            break
    return False


def close_gripper(env, hold_pos, max_steps, transitions):
    prev_gripper_pos = None
    stall_count = 0
    for _ in range(max_steps):
        obs_before = env._get_obs()
        current_pos = env.target_pos
        current_gripper = env.data.qpos[env.gripper_qpos_addr]
        delta_pos = np.clip(hold_pos - current_pos, -env.max_delta_pos, env.max_delta_pos)
        delta_gripper = np.clip(0.0 - current_gripper, -env.max_delta_gripper, env.max_delta_gripper)
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, delta_gripper], dtype=np.float32)
        obs_after, reward, terminated, truncated, info = env.step(action)
        transitions.append((obs_before, action, reward, obs_after, terminated or truncated))
        if terminated or truncated:
            return terminated
        new_gripper_pos = env.data.qpos[env.gripper_qpos_addr]
        if prev_gripper_pos is not None and abs(new_gripper_pos - prev_gripper_pos) < 1e-4:
            stall_count += 1
            if stall_count >= 5:
                break
        else:
            stall_count = 0
        prev_gripper_pos = new_gripper_pos
    return False


def run_one_episode(env):
    """Returns (succeeded, transitions) for one scripted grasp attempt."""
    obs, info = env.reset()
    transitions = []

    object_pos = env.data.xpos[env.object_id].copy()
    HOVER_HEIGHT = 0.15
    GRASP_HEIGHT = object_pos[2] + 0.01
    LIFT_HEIGHT = 0.20

    above = np.array([object_pos[0], object_pos[1], object_pos[2] + HOVER_HEIGHT])
    if move_to(env, above, env.gripper_open_max, 110, transitions):
        return True, transitions
    if len(transitions) >= env.max_episode_steps:
        return False, transitions

    grasp_pos = np.array([object_pos[0], object_pos[1], GRASP_HEIGHT])
    if move_to(env, grasp_pos, env.gripper_open_max, 35, transitions):
        return True, transitions
    if len(transitions) >= env.max_episode_steps:
        return False, transitions

    if close_gripper(env, grasp_pos, 35, transitions):
        return True, transitions
    if len(transitions) >= env.max_episode_steps:
        return False, transitions

    lift_pos = np.array([object_pos[0], object_pos[1], LIFT_HEIGHT])
    terminated = move_to(env, lift_pos, 0.0, 55, transitions)
    if terminated:
        return True, transitions
    if len(transitions) >= env.max_episode_steps:
        return False, transitions

    remaining = env.max_episode_steps - len(transitions)
    hold_steps = min(remaining, env.grasp_hold_required + 20)
    for _ in range(hold_steps):
        obs_before = env._get_obs()
        action = np.zeros(5, dtype=np.float32)
        obs_after, reward, terminated, truncated, info = env.step(action)
        transitions.append((obs_before, action, reward, obs_after, terminated or truncated))
        if terminated:
            return True, transitions
        if truncated:
            break

    return False, transitions


def main():
    n_successes_target = int(sys.argv[1]) if len(sys.argv) > 1 else 300

    env = FrankaOSCGraspEnv(use_camera=False)

    all_obs, all_actions, all_rewards, all_next_obs, all_dones = [], [], [], [], []
    n_attempts = 0
    n_successes = 0

    while n_successes < n_successes_target:
        n_attempts += 1
        succeeded, transitions = run_one_episode(env)
        if succeeded:
            n_successes += 1
            for obs, action, reward, next_obs, done in transitions:
                all_obs.append(obs)
                all_actions.append(action)
                all_rewards.append(reward)
                all_next_obs.append(next_obs)
                all_dones.append(done)
        if n_attempts % 20 == 0:
            print(f"  attempts={n_attempts} successes={n_successes}/{n_successes_target} "
                  f"(success rate so far: {n_successes/n_attempts*100:.0f}%)")

    env.close()

    out_path = os.path.join(os.path.dirname(__file__), "demonstrations.npz")
    np.savez(
        out_path,
        obs=np.array(all_obs, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.float32),
        rewards=np.array(all_rewards, dtype=np.float32),
        next_obs=np.array(all_next_obs, dtype=np.float32),
        dones=np.array(all_dones, dtype=bool),
    )
    print(f"\nSaved {len(all_obs)} transitions from {n_successes} successful episodes "
          f"({n_attempts} attempts, {n_successes/n_attempts*100:.0f}% success rate) to {out_path}")


if __name__ == "__main__":
    main()
