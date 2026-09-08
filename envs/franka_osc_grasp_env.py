# gymnasium is the standard library for creating RL environments
import gymnasium

# numpy is used for numerical operations like arrays and math
import numpy as np

# mujoco is the physics engine we use to simulate the robot
import mujoco

# os is used to build file paths that work on any operating system
import os

from osc_controller import DiffIKController, down_facing_quat


class FrankaOSCGraspEnv(gymnasium.Env):
    """Grasp task controlled via Operational Space Control (differential
    IK — see osc_controller.py) instead of raw joint deltas. The policy
    commands Cartesian end-effector deltas [dx, dy, dz, dyaw, dgripper]; a
    persistent target pose is updated by that delta each step, and
    DiffIKController converts it into joint-angle references for the
    Panda's existing position-controlled actuators.

    The object/success/reward design (real physical object, annulus
    spawning, contact+closure+lift-height success verification, reward
    shaping and clipping) is carried over from 4-arm_project's
    FrankaGraspEnv nearly unchanged — only the ARM control mechanism
    changed. See IMP_NOTES.md for why: OSC was adopted specifically to
    remove the joint-coordination burden from the policy, not to redesign
    the grasp-success criteria, which were independently hard-won.
    """

    def __init__(self, use_camera=False, camera_size=128):
        super().__init__()

        xml_path = os.path.join(os.path.dirname(__file__), "panda", "panda_grasp.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.n_arm_joints = 7

        # Cartesian action design: policy commands a DELTA to a persistent
        # target pose (position + yaw), not an absolute pose — mirrors the
        # old project's delta-action convention (incident #13 in
        # 4-arm_project/IMP_NOTES.md) applied one level up, in task space
        # instead of joint space.
        self.max_delta_pos = 0.03    # meters per env step
        self.max_delta_yaw = 0.2     # radians per env step
        self.max_delta_gripper = 0.01  # meters per env step (range is 0-0.04)
        self.action_space = gymnasium.spaces.Box(
            low=np.array([-self.max_delta_pos] * 3 + [-self.max_delta_yaw, -self.max_delta_gripper], dtype=np.float32),
            high=np.array([self.max_delta_pos] * 3 + [self.max_delta_yaw, self.max_delta_gripper], dtype=np.float32),
        )

        # target position is clamped to this workspace box so the policy
        # can never ask the IK controller to chase a target far outside the
        # arm's real reach. x/y cover the object's spawn annulus (up to
        # 0.5m radius — see object_radius_max below) with margin; z allows
        # descending to just above the floor for grasping, up to a safe
        # height comfortably below full arm extension.
        self.workspace_low = np.array([-0.6, -0.6, 0.02], dtype=np.float32)
        self.workspace_high = np.array([0.6, 0.6, 0.8], dtype=np.float32)

        # the gripper's actuator (a tendon over both fingers) takes ctrl in
        # [0, 255] mapping to a physical opening of [0, 0.04]m (see the ratio
        # baked into panda.xml's actuator8 gain — 255 * (0.04/255) = 0.04).
        # we work in physical meters for the action/observation and convert
        # to the actuator's ctrl units only when writing to data.ctrl.
        self.gripper_open_max = 0.04
        self.gripper_to_ctrl_scale = 255.0 / self.gripper_open_max

        # observation: tip position + tip-to-object vector + object position
        # + gripper opening/velocity + current target pose (position + yaw
        # as cos/sin to avoid the discontinuity a raw wrapped angle would
        # create). No raw arm joint angles here — the whole point of OSC is
        # that the policy shouldn't need to reason in joint space; the
        # target pose IS the policy's own state, so it's included so the
        # policy can track the cumulative effect of its own past actions.
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )

        # tcp site (added to panda.xml in Phase 1) as the tip reference
        self.tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

        self.object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.left_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

        # the object has a real freejoint (not mocap — grasping needs actual
        # contact/friction) — look up its qpos block by joint address rather
        # than assuming indices (4-arm_project incident #5)
        object_jnt_id = self.model.body_jntadr[self.object_id]
        self.object_qpos_addr = self.model.jnt_qposadr[object_jnt_id]

        # gripper position/velocity addresses, looked up by joint name rather
        # than assumed indices (finger_joint1; finger_joint2 is slaved to it
        # via an equality constraint in panda.xml, so they move together)
        finger_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        self.gripper_qpos_addr = self.model.jnt_qposadr[finger_jnt_id]
        self.gripper_qvel_addr = self.model.jnt_dofadr[finger_jnt_id]

        # "home" arm+finger pose (only the first 9 values of the model's
        # keyframe — the rest is the object's freejoint, recorded before the
        # object existed, so it would place the object at the world origin
        # if used as-is)
        self.home_qpos_arm = self.model.key_qpos[0][:9].copy()

        # differential-IK controller (see osc_controller.py and
        # IMP_NOTES.md for the full design/debugging history). home_qpos
        # feeds its nullspace secondary objective, biasing the redundant
        # 7th joint toward this comfortable posture rather than letting it
        # wander into configurations that stress actuator torque limits
        # (incident #6).
        self.controller = DiffIKController(
            self.model, self.data, self.tip_id, home_qpos=self.home_qpos_arm[:self.n_arm_joints]
        )

        # how many physics steps to hold each commanded target for before
        # the policy acts again. 20 steps * 0.002s timestep = 40ms per env
        # step (25Hz control rate) — verified empirically
        # (test_osc_controller_standalone.py) that the controller makes
        # substantial, stable progress toward a target within this many
        # steps for per-step deltas of this task's scale (a few cm), even
        # though fully settling a LARGE target from scratch takes much
        # longer (~8 real seconds) — that's fine here since the policy
        # commands small incremental deltas, not single large jumps.
        self.n_substeps = 20

        # object spawns within this annulus (ring) on the floor, around the
        # base. object_radius_min clears the robot's own base — its largest
        # visual geom has a bounding sphere radius of ~0.19m
        # (model.geom_rbound). object_radius_max keeps the target comfortably
        # inside the arm's ~0.9m true max reach.
        self.object_radius_min = 0.25
        self.object_radius_max = 0.5
        self.object_floor_z = 0.015  # half the box's side length, resting flush on the floor

        # distance (meters) scale for the proximity reward shaping
        self.close_scale = 0.05

        # success = both fingers touching + gripper closed enough + object
        # actually lifted this high above the floor, for this many CONSECUTIVE
        # steps — see 4-arm_project/IMP_NOTES.md incidents #24/#25 for why
        # this specific, shape-agnostic definition was needed (contact alone,
        # and contact+closure alone, both let through invalid grasps)
        self.lift_height_required = 0.05  # meters above the floor
        self.grasp_hold_required = 20
        self.grasp_hold_counter = 0

        # max env steps per episode
        self.max_episode_steps = 500
        self.current_step = 0

        # lazily-created offscreen renderer for the wired-up camera (Phase 4)
        self.use_camera = use_camera
        self.camera_size = camera_size
        self.camera_renderer = None
        if self.use_camera:
            self.camera_renderer = mujoco.Renderer(
                self.model, height=self.camera_size, width=self.camera_size
            )

    def get_camera_image(self):
        """Returns an (camera_size, camera_size, 3) uint8 RGB array from the
        fixed scene camera. Requires use_camera=True."""
        if self.camera_renderer is None:
            raise RuntimeError("Camera not enabled — construct FrankaOSCGraspEnv(use_camera=True)")
        self.camera_renderer.update_scene(self.data, camera="scene_cam")
        return self.camera_renderer.render()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.grasp_hold_counter = 0

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:9] = self.home_qpos_arm

        # randomize the object's position anywhere inside the annulus
        # (object_radius_min to object_radius_max) on the floor, around the
        # base. Sampling r^2 uniformly makes this uniform over AREA.
        # Uses self.np_random (gymnasium's seeded RNG, set up by
        # super().reset(seed=seed) above), not the global np.random — the
        # global generator ignores the seed argument entirely, which
        # gymnasium's own check_env() catches via a step-determinism test.
        radius = np.sqrt(self.np_random.uniform(self.object_radius_min**2, self.object_radius_max**2))
        azimuth = self.np_random.uniform(0.0, 2 * np.pi)
        object_x = radius * np.cos(azimuth)
        object_y = radius * np.sin(azimuth)

        yaw = self.np_random.uniform(0.0, 2 * np.pi)
        quat = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

        self.data.qpos[self.object_qpos_addr:self.object_qpos_addr + 3] = [object_x, object_y, self.object_floor_z]
        self.data.qpos[self.object_qpos_addr + 3:self.object_qpos_addr + 7] = quat

        mujoco.mj_forward(self.model, self.data)

        # the controller's persistent reference must be re-anchored to the
        # arm's actual (just-teleported) qpos, or it would try to move from
        # wherever the PREVIOUS episode left off (osc_controller.py's
        # reset() docstring)
        self.controller.reset(q_ref=self.home_qpos_arm[:self.n_arm_joints].copy())

        # target pose starts at the tip's actual home position/orientation
        # (yaw=0 by down_facing_quat's construction — see osc_controller.py)
        self.target_pos = self.data.site_xpos[self.tip_id].copy()
        self.target_yaw = 0.0

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        dx, dy, dz, dyaw, dgripper = action

        self.target_pos = np.clip(
            self.target_pos + np.array([dx, dy, dz], dtype=np.float32),
            self.workspace_low, self.workspace_high,
        )
        # wrap to [-pi, pi] so yaw doesn't grow unbounded across an episode
        self.target_yaw = (self.target_yaw + dyaw + np.pi) % (2 * np.pi) - np.pi
        desired_quat = down_facing_quat(self.target_yaw)

        current_gripper_pos = self.data.qpos[self.gripper_qpos_addr]
        target_gripper_pos = np.clip(current_gripper_pos + dgripper, 0.0, self.gripper_open_max)
        gripper_ctrl = target_gripper_pos * self.gripper_to_ctrl_scale

        # hold the same (dx,dy,dz,dyaw,dgripper)-derived target fixed across
        # n_substeps physics steps, re-solving the IK fresh each physics
        # step — mj_forward must run before each solve() call so the
        # Jacobian/site pose reflect the state that mj_step just integrated
        # (verified necessary in osc_controller.py's development — mj_step
        # leaves derived quantities stale for the OLD qpos, not the new one)
        for _ in range(self.n_substeps):
            mujoco.mj_forward(self.model, self.data)
            self.data.ctrl[:self.n_arm_joints] = self.controller.solve(self.target_pos, desired_quat)
            self.data.ctrl[self.n_arm_joints] = gripper_ctrl
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1

        obs = self._get_obs()

        tip_pos = self.data.site_xpos[self.tip_id]
        object_pos = self.data.xpos[self.object_id]
        distance = np.linalg.norm(tip_pos - object_pos)

        reward = -distance
        reward += 0.1 * np.exp(-distance / self.close_scale)

        left_contact = False
        right_contact = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            bodies = {self.model.geom_bodyid[contact.geom1], self.model.geom_bodyid[contact.geom2]}
            if self.object_id in bodies and self.left_finger_id in bodies:
                left_contact = True
            if self.object_id in bodies and self.right_finger_id in bodies:
                right_contact = True

        both_fingers_touching = left_contact and right_contact

        gripper_pos_now = self.data.qpos[self.gripper_qpos_addr]
        gripper_closed_enough = gripper_pos_now < (self.gripper_open_max / 2)

        # Discourage closing the gripper before it's actually near the
        # object. Nothing previously penalized this — every other
        # gripper-related term only engages once both_fingers_touching is
        # true — so a policy that closes early pays no cost at all, and
        # testing a 250k-step checkpoint showed exactly that: the gripper
        # closing well before the arm got close, rather than staying open
        # through the approach. Scaled by both how closed the gripper is
        # (0 at fully open, max at fully closed) and how far the object
        # still is (reusing close_scale, the same distance scale as the
        # proximity shaping above; tapers to 0 once within close_scale, so
        # it never penalizes the necessary close-right-before-grasping
        # action). Small weight (0.05, vs 0.5-5.0 for the other terms) —
        # meant as a gentle nudge, not a dominant term.
        gripper_closedness = 1.0 - (gripper_pos_now / self.gripper_open_max)
        distance_factor = min(distance / self.close_scale, 1.0)
        reward -= 0.05 * gripper_closedness * distance_factor

        object_height_above_floor = object_pos[2] - self.object_floor_z
        lifted_enough = object_height_above_floor > self.lift_height_required

        is_grasping = both_fingers_touching and gripper_closed_enough and lifted_enough

        # Dense lift-height bonus, gated behind the SAME threshold as the
        # success condition (lifted_enough) rather than any nonzero height
        # while touching+closed. The looser version let a policy farm
        # reward via brief grab-jostle-release cycles — squeeze the object
        # hard enough to nudge it up a few mm, collect 0.5*height, let go,
        # repeat — without ever committing to a genuine sustained lift.
        # Confirmed via testing checkpoints at both 366k and 500k steps:
        # the trained policy consistently approached, grabbed, and released
        # WITHOUT lifting, timing out every episode (ep_len_mean pinned at
        # 500 the entire run) — exactly what this reward-hacking path
        # predicts. Requiring the real threshold removes the shortcut.
        if is_grasping:
            reward += 0.5 * object_height_above_floor
        elif both_fingers_touching and gripper_closed_enough:
            # Small dense gradient for ANY height gained while gripping,
            # even below lift_height_required — needed because gating the
            # bonus above behind is_grasping (incident #9's fix) created a
            # hard cliff: zero reward difference between "touching,
            # height=0" and "touching, height=4.9cm", so nothing rewarded
            # even ATTEMPTING to lift. Confirmed this is a real gap, not
            # just an early-training artifact: the grasp_hold_counter-based
            # penalty below only fires once a hold has actually started
            # (counter > 0), which requires crossing lifted_enough first —
            # so "closes on the object, never lifts at all, then releases"
            # triggers NEITHER incident #9's penalty NOR this term's bigger
            # sibling, and paid literally zero reward difference for trying
            # vs not trying. Weight kept 10x smaller than the full
            # is_grasping bonus (0.05 vs 0.5) specifically so a genuine
            # sustained lift stays clearly more valuable than repeatedly
            # farming a tiny sub-threshold height — the exact exploit
            # incident #9 removed should not reappear at a smaller scale.
            reward += 0.05 * max(0.0, object_height_above_floor)

        if is_grasping:
            reward += 0.5
            self.grasp_hold_counter += 1
        else:
            # Penalize BREAKING an already-in-progress grasp, not just
            # silently resetting the counter. Previously, releasing had no
            # immediate cost beyond a foregone future bonus — not a strong
            # enough signal against grab-then-release given SAC's reward
            # discounting. Penalty scales with how much hold progress was
            # lost, so releasing right before completion hurts more than
            # releasing immediately (which costs nothing here, matching a
            # grasp attempt that never really started).
            if self.grasp_hold_counter > 0:
                reward -= 0.3 * (self.grasp_hold_counter / self.grasp_hold_required)
            self.grasp_hold_counter = 0

        terminated = bool(self.grasp_hold_counter >= self.grasp_hold_required)

        if terminated:
            reward += 5.0

        # defensive clip — see 4-arm_project/IMP_NOTES.md incident #26
        reward = float(np.clip(reward, -2.0, 6.0))

        truncated = self.current_step >= self.max_episode_steps

        return obs, reward, terminated, truncated, {
            "distance": float(distance),
            "grasp_hold": self.grasp_hold_counter,
        }

    def _get_obs(self):
        tip_pos = self.data.site_xpos[self.tip_id].copy()    # shape (3,)
        object_pos = self.data.xpos[self.object_id].copy()   # shape (3,)
        tip_to_object = object_pos - tip_pos                 # shape (3,)

        gripper_pos = self.data.qpos[self.gripper_qpos_addr:self.gripper_qpos_addr + 1].copy()  # shape (1,)
        gripper_vel = self.data.qvel[self.gripper_qvel_addr:self.gripper_qvel_addr + 1].copy()  # shape (1,)

        target_yaw_trig = np.array([np.cos(self.target_yaw), np.sin(self.target_yaw)], dtype=np.float32)

        # result shape = (3)+(3)+(3)+(1)+(1)+(3)+(2) = 16
        return np.concatenate([
            tip_pos, tip_to_object, object_pos, gripper_pos, gripper_vel,
            self.target_pos.astype(np.float32), target_yaw_trig,
        ]).astype(np.float32)

    def render(self):
        # rendering the interactive viewer is handled externally in test scripts.
        pass

    def close(self):
        if self.camera_renderer is not None:
            self.camera_renderer.close()
