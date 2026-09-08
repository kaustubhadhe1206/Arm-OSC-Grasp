import sys
import os
import mujoco

# mujoco.viewer provides the interactive 3D window to visualize the simulation
import mujoco.viewer

import numpy as np
import time

# add envs/ folder to path so we can import FrankaOSCGraspEnv
sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))

from franka_osc_grasp_env import FrankaOSCGraspEnv
from stable_baselines3 import SAC

# pick which model to load: pass a path as a command-line argument to test a
# specific checkpoint, e.g.
# "python test_osc_grasp.py checkpoints/sac_franka_osc_grasp_150000_steps"
# defaults to the final saved model if no argument is given
model_path = sys.argv[1] if len(sys.argv) > 1 else "sac_franka_osc_grasp"
print(f"Loading model from: {model_path}")
print("Press SPACE (with the viewer window focused) to pause/resume mid-episode.")

env = FrankaOSCGraspEnv()

# the viewer's own "Pause" button in the GUI panel does NOT actually pause
# anything here — this script drives physics stepping itself, not the
# viewer, so that button has nothing to hook into. key_callback wires up a
# real pause instead.
paused = {"value": False}
GLFW_KEY_SPACE = 32


def key_callback(keycode):
    if keycode == GLFW_KEY_SPACE:
        paused["value"] = not paused["value"]
        print("Paused" if paused["value"] else "Resumed")


model = SAC.load(model_path, env=env)

with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:

    obs, info = env.reset()

    while viewer.is_running():

        if paused["value"]:
            viewer.sync()
            time.sleep(0.01)
            continue

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        viewer.sync()

        # env.step() already advances physics by n_substeps (20 * 2ms = 40ms),
        # so no extra sleep is needed to keep the animation at a watchable pace
        # — sleeping the full 40ms here on top would make it needlessly slow.
        time.sleep(0.01)

        if terminated or truncated:
            outcome = "SUCCESS" if terminated else "TIMEOUT"
            print(f"Episode ended: {outcome} (distance={info['distance']:.3f}m, "
                  f"grasp_hold={info['grasp_hold']})")
            obs, info = env.reset()

env.close()
