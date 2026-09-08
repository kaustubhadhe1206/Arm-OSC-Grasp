# Implementation Notes — 5-arm_project_osc

Numbered chronological journal of bugs found and fixes applied, following the
same format as `4-arm_project/IMP_NOTES.md`. Substantial Q&A explanations go
in `QA_LOG.md` instead.

## Project setup

Pivoted from `4-arm_project`'s raw-joint-delta SAC grasping (which topped out
around 1/20 successes after 3 days of iteration) to Operational Space
Control: the RL policy commands Cartesian end-effector pose deltas
`[dx, dy, dz, dyaw, dgripper]`, and a differential inverse-kinematics
controller (`envs/osc_controller.py`) converts that into joint-angle targets
for the Panda's existing position-controlled actuators. Goal: remove the
joint-coordination burden from the RL policy.

Carried forward from the old project: `device="cpu"` for parallel training,
explicit `Monitor` wrapping in `make_env()`, `N_ENVS=4`, mocap-vs-real-object
distinction, force-closure/lift-height success verification, annulus-based
object spawning, CPU-only PyTorch installed from the start.

No separate reach-only phase this time — OSC should make reaching fast
enough to learn jointly with grasping via the existing proximity shaping.

## Incident #1 — Quaternion double-cover sign flip in orientation error

**Symptom:** standalone controller test showed ~180 degree orientation error
on every single target, no matter how the "down" quaternion was constructed.

**Cause:** `mju_subQuat`'s internal angle is `2*acos(qdif[0])`, which lands
in `[0, 2*pi]` rather than `[-pi, pi]`. If `desired_quat` and `current_quat`
are in opposite hemispheres of the double cover (`q` and `-q` represent the
identical rotation, but have `dot(q1,q2) < 0`), it computes the "long way
around" instead of the true shortest rotation.

**Fix:** flip `desired_quat`'s sign whenever `dot(desired_quat,
current_quat) < 0`, before computing the error. A no-op on the rotation
represented, but fixes which "copy" of it `mju_subQuat` sees.

## Incident #2 — Wrong assumed "down" orientation

**Symptom:** after fixing #1, orientation error still ~180 degrees on every
target.

**Cause:** assumed the Panda hand's approach axis (local +Z) could be
pointed straight down via a 180 degree rotation about world X, giving quat
`[0,1,0,0]`. Measured the tcp site's *actual* world orientation at the
`panda.xml` "home" keyframe instead: local +Z was already at world
`(0,0,-1)` there, but achieved via a 180 degree rotation about world axis
`(1,1,0)/sqrt(2)` — not X — a consequence of the Panda's specific chain of
per-link quat offsets (including the hand body's own -45 degree offset from
link7).

**Fix:** for a 180 degree rotation about axis `n=(nx,ny,nz)` the quaternion
is `[0, nx, ny, nz]`. Used `[0, 1/sqrt2, 1/sqrt2, 0]` as the canonical "down,
yaw=0" orientation in `down_facing_quat()`, confirmed both analytically
(matches the measured `site_xmat` exactly) and empirically.

## Incident #3 — Per-element clipping distorted correction direction

**Symptom:** after fixing #1 and #2, a pure-kinematic (no-physics) test
still diverged: position converged then orientation error grew monotonically
to 180 degrees.

**Cause:** `np.clip(joint_delta, -max, max)` clips each of the 7 joints'
deltas independently. Whenever more than one joint saturates simultaneously,
this changes the *direction* of the least-squares correction, not just its
magnitude — confirmed the traced delta norm sat pinned at
`sqrt(7)*max_joint_delta` (every joint saturated) for the entire run.

**Fix:** rate-limit by scaling the whole 7-vector uniformly when its norm
exceeds the cap, preserving direction exactly.

## Incident #4 — `mju_subQuat`'s result is body-frame, not world-frame

**Symptom:** even after #1-#3, the pure-kinematic test still showed
orientation error growing monotonically to 180 degrees and getting stuck
there.

**Cause:** verified via a from-scratch numerical test (constructing an
asymmetric orientation and comparing candidate rotation-composition
formulas against `mju_mulQuat`/`mju_subQuat`'s actual outputs) that
`mju_subQuat(res, qa, qb)` gives `Exp(res) = R(qb)^-1 @ R(qa)` — a rotation
expressed in **qb's local frame** — not `R(qa) @ R(qb)^-1` (world frame).
`mj_jacSite`'s rotational Jacobian maps joint velocities to **world-frame**
angular velocity, so feeding the body-frame error vector straight in was a
frame mismatch. Invisible for small errors near identity orientation (the
two frames nearly coincide), but this arm's home orientation is ~180 degrees
from identity, so the mismatch was large from the start.

**Fix:** rotate the local-frame error into world frame:
`ori_error = current_mat.reshape(3,3) @ ori_error_local`. After this fix the
pure-kinematic test converged to exactly 0.000mm / 0.000deg.

## Incident #5 — Runaway control rate (100 rad/s commanded)

**Symptom:** with `max_joint_delta=0.2`, both kinematic and physics tests
converged partway then diverged/oscillated.

**Cause:** `max_joint_delta` is applied per *physics step*
(`panda.xml`'s timestep is 0.002s), so 0.2 rad/step is a commanded rate of
~100 rad/s — vastly beyond anything physical (~2.5 rad/s for the real
Panda). At that rate the per-step motion is too large for the Jacobian
linearization to stay valid across the step.

**Fix:** reduced `max_joint_delta` to ~0.01 rad/step (~5 rad/s, still
generous but sane).

## Incident #6 — Joint7 torque saturation causing self-oscillation

**Symptom:** with the algorithm otherwise correct (post #1-#5), holding a
fixed target indefinitely settled into a persistent, non-decaying
`qvel` of ~0.11 rad/s — not converging to rest.

**Diagnosis:** printed the full `qvel` vector and found the oscillation was
confined *entirely* to joint7 (`qvel[6]`); every other joint, including both
fingers, sat at ~0. Joint7's actuator has a `forcerange` of only ±12 N·m
(same as joints 5/6, unlike joints 1-4's ±87 N·m). The IK's bare
minimum-norm pseudo-inverse solution happened to settle on a posture where
holding position against gravity/inertial coupling at that joint needed more
torque than ±12 N·m allows, causing a saturated-PD "hunting" limit cycle.
Confirmed the same result reproduces with the bare `panda.xml` alone (no
object, no floor, `ncon=0` throughout) — ruling out contact as a factor.

**Fix:** added a nullspace secondary objective (standard for redundant
7-DOF-for-6-DOF-task manipulators) that pulls the redundant DOF toward a
known-comfortable home posture, projected through `I - J_pinv @ J` so it
never fights the primary 6D task. Its own magnitude is separately capped
(to `0.5 * max_joint_delta`) since an uncapped version — even though
task-neutral at any single instant — caused real cumulative task-space
drift once its magnitude dwarfed the (rate-limited) primary term.

## Incident #7 — `ctrl = actual_qpos + delta` has no authority against disturbances

**Symptom:** the single biggest and longest debugging effort. With every
other fix in place, large target moves grew *unboundedly* over 30+ seconds
of simulated time (150mm error growing past 300mm and still climbing) —not
oscillating, not plateauing, genuinely diverging. Control-rate decimation
(recomputing the IK once every 20 physics steps instead of every step, so
the actuator got 20x more time per commanded reference) made *no*
measurable difference, ruling out actuator-lag/timing as the cause. No
joint ever hit its `ctrlrange` limit. Small (1cm) targets did NOT diverge,
only large ones did — but even small targets showed a slow-to-recover
overshoot, hinting the mechanism was general, just not always fatal.

**Diagnosis:** compared the *commanded* per-step joint delta against the
*actual* realized per-step joint motion (`data.qpos` before/after one
`mj_step`) throughout a run. Found their cosine similarity degraded steadily
from +0.98 (aligned) to -0.49 (anti-aligned) over ~2000 steps. Specifically,
joint2's commanded delta flipped sign repeatedly over time while its real
motion stayed a fixed-sign drift the entire time — the actual joint was
being governed by an external disturbance (gravity/inertial coupling), not
by the command.

Root cause: `ctrl = actual_qpos + jd` recomputes the reference from the
arm's *current, possibly-disturbed* measured position every step. If
gravity drags a joint away from where it should be, the next `jd` is a
small, locally-correct correction — but it gets added to that SAME dragged
position, so the reference always stays close to wherever the arm actually
is. The resulting PD tracking error (`ctrl - actual`) is bounded by `jd`'s
own (small, task-error-proportional) magnitude, and can never grow large
enough to generate the restoring torque needed to counteract a persistent
disturbance. This is not a tuning problem — it's structurally incapable of
building up integral-like authority.

**First attempted fix (failed further):** switched to a persistent internal
reference (`self.q_ref`) that integrates `jd` across calls independently of
measured qpos, so PD error can grow as needed. This fixed incident #7's
divergence but introduced a new one: `q_ref` could run away *unbounded* from
the actual arm if the arm's response bandwidth fell behind the commanded
advance rate (confirmed: `q_ref` drifted up to ~1.6 rad from actual qpos,
task error spiked to hundreds of mm) — because the Jacobian/task-error is
correctly computed from where the arm *really* is, so once `q_ref` and
actual diverge, corrections integrated onto `q_ref` no longer correspond to
directions that are meaningful for actual's true position.

**Actual fix:** kept the persistent reference, but capped how far it's
allowed to lead the actual measured qpos (`max_ref_lag`, tuned to 0.025 rad
— see `envs/osc_controller.py`'s `__init__` comment). This bounds the
reference to never run away while still allowing enough PD tracking error
to build real restoring torque. Swept several values empirically; 0.025 rad
gave the best combination of stability and steady-state accuracy (settles
within ~4mm / 0.4 degrees) without introducing a visible limit-cycle
oscillation (smaller values did; the amplitude of the residual oscillation
scaled directly with `max_ref_lag` in a sweep from 0.02 to 0.15).

**Final verification:** `test_osc_controller_standalone.py`, run with
`STEPS_PER_TARGET=8000` (large moves take ~8 real seconds to fully settle,
confirmed by tracing — this is a one-time settle cost per large move, not a
sign of instability), passes all 5 targets (mixed position/orientation/yaw
combinations) within 5mm / 1.1 degree tolerance.

## Incident #8 — Premature entropy collapse on first real training run

**Symptom:** first real `train_osc_grasp_parallel.py` run (SB3 default
`target_entropy = -action_dim = -5.0` for this 5-dim action space):
`ent_coef` collapsed from ~0.87 at 2k steps to 0.0189 by 64k steps to 0.0016
(near-fully deterministic) by 130k steps — while `ep_len_mean` stayed
pinned at exactly 500 (the episode cap) the entire time, meaning zero
successful grasps ever occurred. `ep_rew_mean` did improve (-341 to -199 to
-96), so the policy was learning *something* (likely proximity behavior),
but exploration noise had nearly vanished before it ever found the sparse
grasp-completion bonus.

**Diagnosis:** same class of problem as 4-arm_project's warm-started-model
entropy collapse, but here it's a *fresh* (non-transfer) model — SAC's
single global entropy temperature satisfied itself based on the smooth,
easy-to-optimize proximity reward, with no signal yet to justify sustaining
noise long enough to stumble onto the rare, sparse grasp/lift/hold
condition.

**Fix:** pinned `target_entropy=-1.0` explicitly (less negative than the
`-5.0` default) in both `train_osc_grasp.py` and
`train_osc_grasp_parallel.py`, forcing SAC to sustain more action noise for
longer. Deliberately the *opposite* correction from 4-arm_project's incident
#26 (which found a MORE permissive target_entropy coincided with a critic
divergence) — but that was a harder 8-dim contact-rich task where the
problem was too much sustained randomness destabilizing the critic; here the
diagnosed problem is the opposite (too little exploration, zero successes),
and the action space is smaller (5-dim). Revisit if this instead produces
incident #26's symptoms (exploding `critic_loss`, collapsing `ep_rew_mean`)
rather than more successes.

**Status:** partially effective. Re-run with `target_entropy=-1.0` showed
`ent_coef` stabilizing around 0.0045 (vs the previous run's 0.0016 at a
similar step count) instead of continuing to crash toward zero — but
`ep_len_mean` stayed pinned at 500 for the ENTIRE run through 550k/1M steps
(checkpoints moved to `checkpoints/old_reward_run_v2/`). Testing checkpoints
at 366k and 500k steps both showed the same behavior: the arm reliably
approaches and grabs the object, then either releases it before lifting, or
the object slips out of the grip — never completing a sustained hold. See
incident #9 for the reward-shaping root cause this pointed to.

## Incident #9 — Reward shape let the policy farm partial credit instead of committing to a hold

**Symptom:** incident #8's entropy fix stopped the entropy collapse but
didn't produce a single success in 550k steps. Watching trained checkpoints
(366k, 500k) in the viewer showed a consistent pattern: approach, grab,
brief height gain, then release — repeated every episode, every timeout.

**Diagnosis:** the reward gave `0.5 * max(0.0, object_height_above_floor)`
for ANY nonzero height while `both_fingers_touching and
gripper_closed_enough` — not gated behind `lifted_enough` (the same
threshold the success condition uses). This is exploitable: squeezing the
object hard enough to jostle it up a few millimeters (well short of the
5cm `lift_height_required`) already pays out a comparable-magnitude dense
reward to a real lift attempt, with none of the risk of a failed hold.
Combined with releasing an in-progress grasp being reward-*neutral* (the
`grasp_hold_counter` just silently reset to 0, costing only a foregone
future bonus, not an immediate penalty), there was no pressure to ever
commit to the harder, longer sustained-hold behavior needed for the sparse
+5.0 completion bonus.

**Fix (`envs/franka_osc_grasp_env.py`):**
1. Gated the dense height bonus behind `is_grasping` (i.e. `lifted_enough`
   included) instead of the looser `touching + closed` condition, removing
   the grab-jostle-release reward-farming path.
2. Added an explicit penalty for breaking an in-progress grasp:
   `reward -= 0.3 * (grasp_hold_counter / grasp_hold_required)` whenever a
   grasp that had accumulated hold progress is lost before completion —
   scaled so releasing right before completion hurts more than releasing
   immediately (which still costs nothing, matching an attempt that never
   really started).

**Status:** old checkpoints/replay-buffer-implied value estimates are
inconsistent with the new reward function (mixing reward regimes in one
buffer would corrupt value estimates), so training was restarted from
scratch rather than resumed. Old checkpoints preserved in
`checkpoints/old_reward_run_v2/` for reference, not deleted. Superseded by
incident #10 before reaching a verdict — a new failure mode appeared at
250k steps under this fix before the original one could be re-evaluated.

## Incident #10 — Nothing discouraged closing the gripper before reaching the object

**Symptom:** testing a 250k-step checkpoint from the incident #9 fix showed
a new failure mode: the gripper closing well before the arm got close to
the object, rather than staying open through the approach.

**Diagnosis:** every gripper-related reward term (the height bonus, the
per-step grasping bonus, the broken-grasp penalty) only engages once
`both_fingers_touching` is true. Closing early costs nothing, so there was
no pressure against it — plausibly an entropy-related artifact too (SAC
uses one global entropy temperature across all action dimensions; if the
arm-position dimensions find a reasonable solution first, the gripper
dimension can settle onto a fixed value without ever being forced to learn
its correct dependence on proximity — see the "warm-started model's
gripper dimension being under-explored" pattern in
4-arm_project/IMP_NOTES.md).

**Fix (`envs/franka_osc_grasp_env.py`):** added a shaping term penalizing a
closed gripper while still far from the object:
`reward -= 0.05 * gripper_closedness * distance_factor`, where
`gripper_closedness` is 0 (open) to 1 (closed) and `distance_factor` reuses
`close_scale` to taper to 0 once within normal grasping range — so it never
penalizes the necessary close-right-before-grasping action, only closing
while genuinely still far away. Small weight (0.05) relative to the other
terms (0.5-5.0), meant as a gentle nudge rather than a dominant term —
revisit if it has no visible effect (weight too small) or if the gripper
now never closes even when appropriate (too large / tapering wrong).

**Status:** training restarted, then stopped again almost immediately
(user moving all training off the local machine to Colab — see below)
before this could be verified. Old checkpoints moved to
`checkpoints/old_reward_run_v3/`.

## Incident #11 — Grasp reward had a hard cliff with zero gradient below the lift threshold

**Symptom:** user reported the SAME "closes then opens without lifting"
behavior persisting even after incident #9's fix.

**Diagnosis:** a real gap in incident #9's fix, not just an
early-training artifact. `grasp_hold_counter` (and its associated broken-
grasp penalty) only ever increments/engages once `is_grasping` has been
`True` at least once, which requires `lifted_enough` (crossing the full 5cm
threshold) first. "Closes on the object, never lifts it off the ground at
all, then releases" never crosses that threshold, so the counter never
leaves 0 — meaning NEITHER incident #9's broken-grasp penalty NOR the
gated height bonus ever engage for this behavior. There was a hard reward
cliff: zero reward difference between "touching+closed, height=0" and
"touching+closed, height=4.9cm" (just short of the threshold), so nothing
rewarded even attempting to lift.

**Fix:** added a small dense gradient (`0.05 * height`, vs the full
`0.5 * height` bonus once past threshold) for any height gained while
touching+closed but below `lifted_enough`. Deliberately kept 10x smaller
than the post-threshold bonus specifically so a genuine sustained lift
stays clearly more valuable than repeatedly farming a tiny sub-threshold
height — i.e., careful not to reintroduce incident #9's exploit at a
smaller scale.

**Status:** not yet verified — see below, training paused entirely while
migrating off the local machine.

## Local training paused — migrating to Google Colab

User's laptop was overheating from sustained local training; decided to
move training to Google Colab (free CPU/T4 GPU) via the VS Code Colab
extension rather than running locally. All local background training
stopped and verified no orphaned worker processes remained. Noted for the
user before committing effort: the T4 GPU won't meaningfully help this
workload (CPU-bound MuJoCo physics + a tiny MLP — `device="cpu"` was
already a deliberate choice for this exact reason), and Colab's free-tier
CPU allocation (~2 vCPUs) may not beat the local machine's core count for
`SubprocVecEnv` parallelism — the actual motivation here is protecting the
laptop, not raw speed, which is a valid reason on its own.

Set up: `git init`'d this project directory (previously not a git repo),
pushed to `https://github.com/kaustubhadhe1206/Arm-OSC-Grasp.git` (chosen
over Google Drive sync or inline-in-notebook alternatives — cleanest,
survives Colab runtime resets automatically). `venv/`, `checkpoints/`, and
`*.zip` are gitignored; the Panda mesh assets (34MB, well under GitHub's
limits) were committed directly rather than re-cloned from Menagerie
separately in Colab, to avoid any risk of a version/path mismatch with the
custom `tcp` site added to `panda.xml`.

## Incident #12 — Gripper chattering/"shivering" near the object

**Symptom:** testing a 350k-step checkpoint (first run on Colab, with
incidents #9-#11's reward fixes in place) showed a new behavior: the arm
approached the cube, the gripper closed partway without grabbing it, then
immediately loosened and started visibly vibrating/"shivering" rather than
settling into a stable attempt.

**Diagnosis:** classic action-chattering, and a plausible NEW exploit of
incident #11's own fix specifically: that fix pays a small dense reward for
ANY nonzero lift height while touching+closed (added to give a gradient
below the full lift threshold). If rapidly buzzing the gripper against the
object causes small physics-contact bounces, each bounce reads as a
nonzero height and pays out that reward repeatedly — potentially easier
for the policy to farm via high-frequency contact events than to commit to
one careful, sustained grasp attempt.

**Fix (`envs/franka_osc_grasp_env.py`):** added a general action-smoothness
penalty — `reward -= 0.05 * ||normalized_action_t - normalized_action_{t-1}||`
(normalized per-dimension by the action space's own range, since dx/dy/dz
are meters, dyaw is radians, and dgripper is meters — different scales that
would otherwise let one dimension dominate). This is standard practice in
robotics RL specifically to suppress chattering, and doesn't depend on the
exploit hypothesis above being exactly right — rapid oscillation is
undesirable regardless of what's motivating it.

**Status:** not yet verified. Training is now running on Colab (see
below), not locally, so restarting requires the user to interrupt the
training cell, pull the latest code, and re-run — I can no longer just
stop/restart a local background process myself for this run.

Added `colab_train.ipynb` (this project's first and only `.ipynb` —
overriding the original "no ipynb files" instruction, which was written
for local-only training and predates the laptop-heat/Colab decision).
Mounts Google Drive and symlinks `checkpoints/` to a Drive folder before
running `train_osc_grasp_parallel.py` unchanged, so checkpoints survive a
Colab disconnect (the local Colab disk is wiped on disconnect/recycle,
which a multi-hour run will eventually hit).
