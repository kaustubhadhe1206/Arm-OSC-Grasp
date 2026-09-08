# sys lets us modify Python's module search path at runtime
import sys

# os is used to build file paths
import os

# add the envs/ folder to Python's search path
sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))

from franka_osc_grasp_env import FrankaOSCGraspEnv

# import the SAC algorithm from stable-baselines3
from stable_baselines3 import SAC

# saves a model snapshot every N steps during training, so we can inspect/test
# progress with test_osc_grasp.py without having to stop the (long) training run
from stable_baselines3.common.callbacks import CheckpointCallback

# use_camera=False since training is state-based, not vision-based —
# the camera is wired up (env.get_camera_image()) but not used until Phase 4
env = FrankaOSCGraspEnv(use_camera=False)

# create the SAC model. Hyperparameters carried over from 4-arm_project's
# train_grasp.py, which reached the best (though still limited, ~5%
# success) results of the raw-joint-delta approach — see IMP_NOTES.md and
# QA_LOG.md there for target_entropy/ent_coef reasoning (left at SB3's
# default rather than a hand-picked value, per incident #26).
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto",
    # see train_osc_grasp_parallel.py's comment: a first run with SB3's
    # default target_entropy (-action_dim = -5.0 here) collapsed ent_coef
    # to ~0.0016 by 130k/1M steps with zero successes ever (ep_len_mean
    # pinned at 500) — exploration died before finding the sparse
    # grasp-completion signal. Pinned less negative to sustain exploration
    # longer for this 5-dim action space.
    target_entropy=-1.0,
    learning_starts=10_000,
    verbose=1,
)

print("Training started (OSC grasp, single env)...")

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix="sac_franka_osc_grasp",
)

model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)

model.save("sac_franka_osc_grasp")
print("Model saved to sac_franka_osc_grasp.zip")

env.close()
