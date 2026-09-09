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
import time

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
    any real training happens, via a single vectorized bulk write directly
    into the buffer's internal arrays -- NOT a Python loop calling
    ReplayBuffer.add() once per transition, which was the first version of
    this function and took several minutes with zero progress output for
    ~49k transitions (each .add() call does a fair amount of work: dict
    creation, reshaping, conditional checks), badly confusing about
    whether it had hung. A plain vectorized numpy assignment does the same
    thing in a fraction of a second.

    SB3's ReplayBuffer stores the NORMALIZED ([-1,1], via
    policy.scale_action) action, not the raw physical-units one --
    confirmed by reading _store_transition's source directly rather than
    assuming, since getting this wrong would silently corrupt every seeded
    transition. Internal array shapes (confirmed via ReplayBuffer.__init__'s
    source, not assumed): observations/next_observations are
    (buffer_size, n_envs, obs_dim), actions is (buffer_size, n_envs,
    action_dim), rewards/dones/timeouts are (buffer_size, n_envs) — note
    buffer.buffer_size here is the ORIGINAL buffer_size//N_ENVS (SB3
    divides it internally for multi-env buffers), not the raw value passed
    to SAC(). Demo transitions are tiled across the n_envs axis (every
    parallel env's buffer slot gets an identical copy) rather than trying
    to spread them across envs, since these are independent successful
    trajectories, not a real synchronized multi-env rollout — this doesn't
    need to be aligned with anything, it's just bulk-loading experience."""
    buffer = model.replay_buffer
    n = len(demo["obs"])
    assert n <= buffer.buffer_size, (
        f"{n} demonstration transitions exceed the per-env replay buffer capacity "
        f"({buffer.buffer_size}) -- increase buffer_size or collect fewer demos."
    )

    low, high = model.policy.action_space.low, model.policy.action_space.high
    scaled_actions = 2.0 * (demo["actions"] - low) / (high - low) - 1.0

    n_envs = buffer.n_envs
    buffer.observations[0:n] = np.repeat(demo["obs"][:, None, :], n_envs, axis=1)
    buffer.next_observations[0:n] = np.repeat(demo["next_obs"][:, None, :], n_envs, axis=1)
    buffer.actions[0:n] = np.repeat(scaled_actions[:, None, :], n_envs, axis=1)
    buffer.rewards[0:n] = np.repeat(demo["rewards"][:, None], n_envs, axis=1)
    buffer.dones[0:n] = np.repeat(demo["dones"][:, None].astype(np.float32), n_envs, axis=1)
    # every kept demo transition is a genuine success (collect_demonstrations.py
    # discards any episode that ran out of steps without terminating), so
    # none of these "done" events are timeouts
    buffer.timeouts[0:n] = 0.0

    buffer.pos = n
    buffer.full = False

    print(f"Seeded replay buffer with {n} demonstration transitions "
          f"(x{n_envs} tiling across parallel envs internally).")


def pretrain_actor(model, demo):
    """Supervised regression: the actor's tanh-squashed mean output should
    match the demonstrated (normalized) action for the demonstrated
    observation. This is plain behavior cloning on the actor network only
    -- the critic is NOT touched here (it learns from the seeded replay
    buffer during normal SAC training instead), since BC on the critic
    isn't meaningful (there's no "correct Q-value" label to imitate)."""
    obs = torch.as_tensor(demo["obs"], dtype=torch.float32)
    # vectorized, not a per-item Python loop calling policy.scale_action()
    # (same class of unnecessary-slowness fix as seed_replay_buffer's
    # rewrite above, though this one was never bad enough to look "stuck")
    low, high = model.policy.action_space.low, model.policy.action_space.high
    scaled_actions = 2.0 * (demo["actions"] - low) / (high - low) - 1.0
    actions = torch.as_tensor(scaled_actions, dtype=torch.float32)

    actor = model.policy.actor
    optimizer = torch.optim.Adam(actor.parameters(), lr=BC_LEARNING_RATE)

    n = obs.shape[0]
    print(f"Pretraining actor on {n} demonstration transitions for {BC_EPOCHS} epochs...")
    for epoch in range(BC_EPOCHS):
        epoch_start = time.time()
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
        # print every epoch, not every 10 -- at the full ~49k-transition
        # scale this loop is slow enough (real gradient steps, not a bulk
        # array op like the buffer seeding above) that long silent gaps
        # between prints previously looked indistinguishable from "stuck"
        print(f"  epoch {epoch}: mse_loss={epoch_loss / n_batches:.5f} "
              f"({time.time() - epoch_start:.1f}s)")


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
