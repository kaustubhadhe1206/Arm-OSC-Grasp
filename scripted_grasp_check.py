# Hand-scripted grasp routine — NO RL, no policy. Drives the environment's
# own action interface through a fixed, obviously-correct sequence (move
# above object, descend, close, lift, hold) using simple proportional
# waypoint-following. This answers a question training logs alone can't:
# does the physics/reward/success-condition setup actually SUPPORT a
# successful grasp at all? If this script can't succeed either, the
# problem is the environment (friction, gripper sizing, height thresholds,
# etc.), not RL exploration — a completely different, more tractable class
# of fix than more reward shaping.
#
# Run: venv/Scripts/python.exe scripted_grasp_check.py

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))
from franka_osc_grasp_env import FrankaOSCGraspEnv


def move_to(env, target_xyz, target_gripper, max_steps, tol=0.01):
    """Drives env.target_pos toward target_xyz and the gripper toward
    target_gripper by repeatedly issuing max-rate deltas (yaw left at 0 —
    this routine doesn't need to rotate), for up to max_steps env.step()
    calls or until the ARM (not just the nominal target) is within tol of
    the position target, whichever comes first. Returns the last step()
    result.

    IMPORTANT: the early-exit check must compare against the actual tip
    position (env.data.site_xpos[env.tip_id]), NOT env.target_pos. The
    first version of this script checked env.target_pos instead — since
    target_pos is a raw accumulator that reaches the waypoint almost
    immediately (within ~10-15 steps, well before the real arm catches
    up), that bug caused every phase to exit early thinking it had
    arrived, when the arm was often still far away. Confirmed via a
    separate full-budget trace: given the complete step budget without
    early exit, the arm actually converges cleanly."""
    result = None
    for _ in range(max_steps):
        current_pos = env.target_pos
        current_gripper = env.data.qpos[env.gripper_qpos_addr]
        delta_pos = np.clip(target_xyz - current_pos, -env.max_delta_pos, env.max_delta_pos)
        delta_gripper = np.clip(target_gripper - current_gripper, -env.max_delta_gripper, env.max_delta_gripper)
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, delta_gripper], dtype=np.float32)
        result = env.step(action)
        _, _, terminated, truncated, _ = result
        if terminated or truncated:
            return result
        tip_pos = env.data.site_xpos[env.tip_id]
        if np.linalg.norm(target_xyz - tip_pos) < tol:
            break
    return result


def close_gripper(env, hold_pos, max_steps):
    """Closes the gripper while holding position at hold_pos, for a FIXED
    number of steps — deliberately does NOT reuse move_to()'s early exit.
    Using move_to() directly for gripper-closing was a real bug: by the
    time this phase starts, the arm is usually already within move_to()'s
    position tolerance (having just arrived there in the previous phase),
    so its position-based early exit fired after only 1-2 steps —
    confirmed directly via a step-by-step trace — leaving the gripper
    almost no time to actually close. This version tracks the gripper's
    OWN convergence instead: stop once further closing has near-zero
    effect (the actuator has hit steady state, whether that's fully closed
    or resting against the object), not once the ARM's position looks
    fine."""
    result = None
    prev_gripper_pos = None
    stall_count = 0
    for _ in range(max_steps):
        current_pos = env.target_pos
        current_gripper = env.data.qpos[env.gripper_qpos_addr]
        delta_pos = np.clip(hold_pos - current_pos, -env.max_delta_pos, env.max_delta_pos)
        delta_gripper = np.clip(0.0 - current_gripper, -env.max_delta_gripper, env.max_delta_gripper)
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, delta_gripper], dtype=np.float32)
        result = env.step(action)
        _, _, terminated, truncated, _ = result
        if terminated or truncated:
            return result
        new_gripper_pos = env.data.qpos[env.gripper_qpos_addr]
        if prev_gripper_pos is not None and abs(new_gripper_pos - prev_gripper_pos) < 1e-4:
            stall_count += 1
            if stall_count >= 5:  # steady state reached (closed, or resting on the object)
                break
        else:
            stall_count = 0
        prev_gripper_pos = new_gripper_pos
    return result


def main():
    env = FrankaOSCGraspEnv(use_camera=False)
    obs, info = env.reset()

    object_pos = env.data.xpos[env.object_id].copy()
    print(f"Object spawned at: {np.round(object_pos, 4)}")

    HOVER_HEIGHT = 0.15
    # object's center + a small margin, not exactly the center — the
    # controller's steady-state tracking error (~5-10mm after speeding it
    # up, see osc_controller.py's max_ref_lag comment) means targeting the
    # exact center risked the fingertips jamming against the floor
    # (confirmed directly: a diagnostic contact trace showed the object's
    # spawn position determining whether closing worked cleanly or got
    # stuck, consistent with a fragile, too-tight height margin)
    GRASP_HEIGHT = object_pos[2] + 0.01
    LIFT_HEIGHT = 0.20

    print("Phase 1: move above object, gripper open")
    above = np.array([object_pos[0], object_pos[1], object_pos[2] + HOVER_HEIGHT])
    move_to(env, above, env.gripper_open_max, max_steps=100)
    print(f"  tip pos: {np.round(env.data.site_xpos[env.tip_id], 4)}, "
          f"tip error: {np.linalg.norm(env.data.site_xpos[env.tip_id] - above):.4f}m")

    print("Phase 2: descend to grasp height")
    grasp_pos = np.array([object_pos[0], object_pos[1], GRASP_HEIGHT])
    move_to(env, grasp_pos, env.gripper_open_max, max_steps=40)
    print(f"  tip pos: {np.round(env.data.site_xpos[env.tip_id], 4)}, "
          f"tip error: {np.linalg.norm(env.data.site_xpos[env.tip_id] - grasp_pos):.4f}m")

    print("Phase 3: close gripper")
    result = close_gripper(env, grasp_pos, max_steps=40)
    gripper_pos_now = env.data.qpos[env.gripper_qpos_addr]
    print(f"  gripper pos: {gripper_pos_now:.4f} (0=closed, {env.gripper_open_max}=open, "
          f"closed_enough threshold={env.gripper_open_max/2})")

    print("Phase 4: lift")
    lift_pos = np.array([object_pos[0], object_pos[1], LIFT_HEIGHT])
    result = move_to(env, lift_pos, 0.0, max_steps=50)
    _, _, terminated, truncated, info = result
    new_object_pos = env.data.xpos[env.object_id].copy()
    height_above_floor = new_object_pos[2] - env.object_floor_z
    print(f"  object height above floor: {height_above_floor:.4f}m "
          f"(need > {env.lift_height_required}m to count as lifted)")
    print(f"  grasp_hold counter: {info['grasp_hold']}, terminated={terminated}, truncated={truncated}")

    if not (terminated or truncated):
        print("Phase 5: hold in place")
        hold_steps = env.grasp_hold_required + 20  # margin over the required 20
        for i in range(hold_steps):
            action = np.zeros(5, dtype=np.float32)  # no further target movement
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"  SUCCESS at hold step {i}! reward={reward:.3f}")
                break
            if truncated:
                print(f"  TRUNCATED during hold at step {i} (ran out of episode steps)")
                break
        else:
            print(f"  did not terminate after {hold_steps} hold steps. "
                  f"grasp_hold={info['grasp_hold']}/{env.grasp_hold_required}")

    final_height = env.data.xpos[env.object_id][2] - env.object_floor_z
    print(f"\nFinal object height above floor: {final_height:.4f}m")
    print("RESULT:", "SUCCESS" if terminated else "FAILED")

    env.close()


if __name__ == "__main__":
    main()
