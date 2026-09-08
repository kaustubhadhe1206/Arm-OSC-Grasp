# Standalone verification of DiffIKController BEFORE it's wired into any RL
# environment. This checks the controller alone can converge to and hold
# commanded Cartesian (position + orientation) targets for the Panda's tcp
# site, using nothing but MuJoCo + the controller. No gym, no SB3.
#
# Run: venv/Scripts/python.exe test_osc_controller_standalone.py

import os
import sys
import numpy as np
import mujoco

sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))
from osc_controller import DiffIKController, down_facing_quat

MODEL_PATH = os.path.join(os.path.dirname(__file__), "envs", "panda", "panda_grasp.xml")

STEPS_PER_TARGET = 8000
# Tolerances relaxed alongside max_ref_lag's increase (0.025 -> 0.06 — see
# osc_controller.py's __init__ comment): that traded some steady-state
# precision for ~2.4x faster settling, since a full training episode's
# step budget was being consumed by slow approach alone. 10mm/1.5deg is
# still far tighter than this task actually needs (a 3cm object, 5cm lift
# threshold).
POS_TOL = 0.010   # 10mm
ORI_TOL = 0.026   # rad, ~1.5 degrees


def run_to_target(model, data, controller, desired_pos, desired_quat, label):
    for step in range(STEPS_PER_TARGET):
        mujoco.mj_forward(model, data)
        data.ctrl[:7] = controller.solve(desired_pos, desired_quat)
        mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)
    final_pos = data.site_xpos[controller.site_id].copy()
    final_mat = data.site_xmat[controller.site_id].copy()
    final_quat = np.zeros(4)
    mujoco.mju_mat2Quat(final_quat, final_mat)

    pos_error = np.linalg.norm(desired_pos - final_pos)

    # same double-cover fix as in DiffIKController.solve() — see its comment
    measured_quat = desired_quat.copy()
    if np.dot(measured_quat, final_quat) < 0:
        measured_quat = -measured_quat

    # mju_subQuat's result is a rotation ANGLE (its norm is frame-independent
    # even though its axis direction is expressed in final_quat's local
    # frame — see DiffIKController.solve()'s comment on this). Norm alone is
    # enough here since this is just an error magnitude for reporting, not a
    # vector being fed into a world-frame Jacobian.
    ori_error_vec = np.zeros(3)
    mujoco.mju_subQuat(ori_error_vec, measured_quat, final_quat)
    ori_error = np.linalg.norm(ori_error_vec)

    status = "PASS" if (pos_error < POS_TOL and ori_error < ORI_TOL) else "FAIL"
    print(f"[{status}] {label}: pos_error={pos_error*1000:.2f}mm, ori_error={np.degrees(ori_error):.2f}deg")
    return status == "PASS"


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    home_key_id = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, home_key_id)
    mujoco.mj_forward(model, data)

    site_id = model.site("tcp").id
    home_qpos = data.qpos[:7].copy()
    controller = DiffIKController(model, data, site_id, home_qpos=home_qpos)
    controller.reset()

    start_pos = data.site_xpos[site_id].copy()
    print(f"Start tcp position: {start_pos}")

    results = []

    # Target 1: small move, straight down, zero yaw
    target1_pos = start_pos + np.array([0.10, 0.05, -0.10])
    target1_quat = down_facing_quat(0.0)
    results.append(run_to_target(model, data, controller, target1_pos, target1_quat, "Target 1 (move + yaw=0)"))

    # Target 2: further move, straight down, +45 degree yaw
    target2_pos = start_pos + np.array([0.20, -0.10, -0.15])
    target2_quat = down_facing_quat(np.radians(45))
    results.append(run_to_target(model, data, controller, target2_pos, target2_quat, "Target 2 (move + yaw=45deg)"))

    # Target 3: move back toward a different spot, -30 degree yaw
    target3_pos = start_pos + np.array([-0.05, 0.15, -0.05])
    target3_quat = down_facing_quat(np.radians(-30))
    results.append(run_to_target(model, data, controller, target3_pos, target3_quat, "Target 3 (move + yaw=-30deg)"))

    # Target 4: return near start, hold — tests stability once converged
    # (run twice to the same target to check it doesn't drift/oscillate)
    target4_pos = start_pos + np.array([0.0, 0.0, -0.05])
    target4_quat = down_facing_quat(0.0)
    results.append(run_to_target(model, data, controller, target4_pos, target4_quat, "Target 4a (near start)"))
    results.append(run_to_target(model, data, controller, target4_pos, target4_quat, "Target 4b (hold, same target)"))

    print()
    if all(results):
        print("ALL TESTS PASSED — controller converges to and holds commanded Cartesian targets.")
    else:
        print("SOME TESTS FAILED — do not build the RL environment on top of this yet.")
        sys.exit(1)


if __name__ == "__main__":
    main()
