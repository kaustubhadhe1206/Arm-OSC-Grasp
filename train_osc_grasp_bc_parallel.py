# Behavior-cloning-warm-started variant of train_osc_grasp_parallel.py.
#
# Why this exists (see IMP_NOTES.md): pure RL (train_osc_grasp_parallel.py)
# ran 585k/1M steps (1952 episodes) without ONCE stumbling onto a
# successful grasp-lift-hold sequence, despite scripted_grasp_check.py
# proving the task is achievable. This is a hard-exploration problem, not
# (only) a reward-shaping one -- reward patches can shape incentives, but
# can't help the policy FIND the behavior if it never happens to try it.
#
# This script seeds the SAC replay buffer with real successful transitions
# (from collect_demonstrations.py) and pretrains the actor to imitate them
# via supervised regression, BEFORE normal RL fine-tuning begins -- so the
# critic starts with accurate Q-values for the successful trajectory region
# instead of having to discover it from scratch, and the actor starts
# already outputting roughly-correct actions instead of random ones.
#
# Run collect_demonstrations.py first to produce demonstrations.npz.
#
# IMPORTANT (Windows-specific): multiprocessing on Windows uses "spawn",
# which re-imports this whole file in each worker process. Everything is
# wrapped in `if __name__ == "__main__":` so workers don't recursively
# spawn more workers.

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))

import numpy as np
import torch
import torch.nn.functional as F

from franka_osc_grasp_env import FrankaOSCGraspEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

N_ENVS = 4
DEMO_PATH = os.path.join(os.path.dirname(__file__), "demonstrations.npz")
BC_EPOCHS = 50
BC_BATCH_SIZE = 256
BC_LEARNING_RATE = 1e-3


def make_env():
    return Monitor(FrankaOSCGraspEnv(use_camera=False))


def seed_replay_buffer(model, demo):
    """Adds every demonstration transition to the replay buffer, BEFORE
    any real training happens. SB3's ReplayBuffer stores the NORMALIZED
    ([-1,1], via policy.scale_action) action, not the raw physical-units
    one -- confirmed by reading _store_transition's source directly rather
    than assuming, since getting this wrong would silently corrupt every
    seeded transition. The buffer is sized for N_ENVS parallel
    environments (shape (buffer_size, N_ENVS, ...)); obs/next_obs/reward/
    done broadcast fine from a single-transition shape via plain numpy
    broadcasting, but action does NOT (ReplayBuffer.add reshapes it to
    exactly (N_ENVS, action_dim), which fails on a flat (action_dim,)
    array) -- so it's tiled explicitly instead."""
    n_transitions = len(demo["obs"])
    for i in range(n_transitions):
        scaled_action = model.policy.scale_action(demo["actions"][i])
        tiled_action = np.tile(scaled_action, (N_ENVS, 1))
        infos = [{} for _ in range(N_ENVS)]
        model.replay_buffer.add(
            demo["obs"][i],
            demo["next_obs"][i],
            tiled_action,
            np.array(demo["rewards"][i]),
            np.array(demo["dones"][i]),
            infos,
        )
    print(f"Seeded replay buffer with {n_transitions} demonstration transitions "
          f"(x{N_ENVS} tiling per transition internally).")


def pretrain_actor(model, demo):
    """Supervised regression: the actor's tanh-squashed mean output should
    match the demonstrated (normalized) action for the demonstrated
    observation. This is plain behavior cloning on the actor network only
    -- the critic is NOT touched here (it learns from the seeded replay
    buffer during normal SAC training instead), since BC on the critic
    isn't meaningful (there's no "correct Q-value" label to imitate)."""
    obs = torch.as_tensor(demo["obs"], dtype=torch.float32)
    scaled_actions = np.array([model.policy.scale_action(a) for a in demo["actions"]], dtype=np.float32)
    actions = torch.as_tensor(scaled_actions, dtype=torch.float32)

    actor = model.policy.actor
    optimizer = torch.optim.Adam(actor.parameters(), lr=BC_LEARNING_RATE)

    n = obs.shape[0]
    print(f"Pretraining actor on {n} demonstration transitions for {BC_EPOCHS} epochs...")
    for epoch in range(BC_EPOCHS):
        permutation = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, BC_BATCH_SIZE):
            idx = permutation[start:start + BC_BATCH_SIZE]
            obs_batch = obs[idx]
            action_batch = actions[idx]

            mean_actions, _, _ = actor.get_action_dist_params(obs_batch)
            predicted = torch.tanh(mean_actions)
            loss = F.mse_loss(predicted, action_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        if epoch % 10 == 0 or epoch == BC_EPOCHS - 1:
            print(f"  epoch {epoch}: mse_loss={epoch_loss / n_batches:.5f}")


if __name__ == "__main__":
    if not os.path.exists(DEMO_PATH):
        raise FileNotFoundError(
            f"{DEMO_PATH} not found -- run collect_demonstrations.py first."
        )
    demo = np.load(DEMO_PATH)
    print(f"Loaded {len(demo['obs'])} demonstration transitions from {DEMO_PATH}")

    env = SubprocVecEnv([make_env for _ in range(N_ENVS)])

    # same hyperparameters as train_osc_grasp_parallel.py (target_entropy
    # included, incident #8) -- BC warm-starting addresses the EXPLORATION
    # problem, it doesn't replace the entropy-collapse fix that was needed
    # for a different reason
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        target_entropy=-1.0,
        # Lowered from 10_000: SB3 samples PURE RANDOM actions during the
        # learning_starts warmup regardless of the buffer's actual content
        # (confirmed by reading _sample_action's source) -- so a large
        # value here would waste steps ignoring the just-pretrained actor.
        # Kept small but nonzero (not 0) to still collect a little genuine
        # fresh interaction data before gradient updates begin.
        learning_starts=2_000,
        verbose=1,
        device="cpu",
    )

    seed_replay_buffer(model, demo)
    pretrain_actor(model, demo)

    print(f"Training started (OSC grasp, BC-warm-started, {N_ENVS} parallel environments)...")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix="sac_franka_osc_grasp_bc_parallel",
    )

    model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)

    model.save("sac_franka_osc_grasp_bc_parallel")
    print("Model saved to sac_franka_osc_grasp_bc_parallel.zip")

    env.close()
