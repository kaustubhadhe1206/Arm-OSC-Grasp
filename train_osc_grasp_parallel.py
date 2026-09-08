# Same training as train_osc_grasp.py, but running N environments in
# parallel across CPU cores to cut down wall-clock training time — see
# 4-arm_project/IMP_NOTES.md incident #27 for the full reasoning behind
# every choice below (device="cpu", N_ENVS=4, explicit Monitor wrapping).
#
# IMPORTANT (Windows-specific): multiprocessing on Windows uses "spawn",
# which re-imports this whole file in each worker process. Everything is
# wrapped in `if __name__ == "__main__":` so workers don't recursively
# spawn more workers.

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))

from franka_osc_grasp_env import FrankaOSCGraspEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# 4-arm_project incident #27: 8 parallel envs worked once (~395 steps/sec)
# but crashed a second run with "MemoryError: bad allocation" when only
# ~4.5GB RAM was free. 4 measured ~184 steps/sec (~4.6x speedup over
# single-env) reliably — kept as the default here too.
N_ENVS = 4


def make_env():
    # Monitor wrapping needed explicitly — SB3 only auto-wraps a raw env
    # with Monitor when given a non-vectorized env directly; building
    # SubprocVecEnv manually bypasses this and silently drops
    # rollout/ep_rew_mean and rollout/ep_len_mean from the training log
    # (4-arm_project incident #28).
    return Monitor(FrankaOSCGraspEnv(use_camera=False))


if __name__ == "__main__":
    env = SubprocVecEnv([make_env for _ in range(N_ENVS)])

    # device="cpu" is required, not just a preference, once multiple worker
    # processes are involved — SB3's default device="auto" picking "cuda"
    # crashed with CUBLAS_STATUS_EXECUTION_FAILED when 8 spawned
    # subprocesses each imported torch and fought over the same GPU's CUDA
    # context (incident #27). This project's venv also installed CPU-only
    # PyTorch from the start for the same reason.
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        # SB3's "auto" target_entropy default is -action_dim (-5.0 here for
        # this 5-dim action space). A first run with that default collapsed
        # ent_coef to ~0.0016 (near-deterministic) by 130k/1M steps while
        # ep_len_mean stayed pinned at 500 the whole time — zero successes
        # ever, meaning exploration died before the policy found the sparse
        # grasp-completion signal. Pinning target_entropy less negative
        # forces SAC to keep more action noise around for longer. NOTE:
        # 4-arm_project incident #26 found the opposite mistake (forcing
        # target_entropy=-3.0, MORE permissive than that task's default -8,
        # coincided with a critic divergence) — but that was a harder,
        # higher-dim (8-action) contact-rich task; this is a smaller 5-dim
        # action space where the earlier run's problem was demonstrably too
        # LITTLE exploration, not too much. Revisit if this instead shows
        # critic instability (exploding critic_loss / collapsing ep_rew_mean)
        # the way incident #26 did.
        target_entropy=-1.0,
        learning_starts=10_000,
        verbose=1,
        device="cpu",
    )

    print(f"Training started (OSC grasp, {N_ENVS} parallel environments)...")

    # save_freq divided by N_ENVS — each callback call advances all N_ENVS
    # environments by one step each, so this keeps checkpoints at the same
    # total-env-steps cadence as the single-env script
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix="sac_franka_osc_grasp_parallel",
    )

    model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)

    model.save("sac_franka_osc_grasp_parallel")
    print("Model saved to sac_franka_osc_grasp_parallel.zip")

    env.close()
