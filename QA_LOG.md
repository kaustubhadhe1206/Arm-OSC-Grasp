# Q&A Log — 5-arm_project_osc

Substantial technical Q&A exchanges only (real explanations, not simple
yes/no procedural calls) — mirrors `4-arm_project/QA_LOG.md`'s convention.

---

**Q: I see the ent_coef is decreasing, what is your thought?** (at 64k/1M
steps, ent_coef=0.0189, ep_rew_mean=-199, ep_len_mean=500)

A: Entropy dropping fast this early with zero successes yet (ep_len_mean
still 500 = no terminations) is the same pattern that hurt the old
project's warm-started model — exploration noise disappearing before the
policy ever discovers a real grasp. Not necessarily fatal on its own (OSC's
smaller/easier action space could just be converging faster than the old
8-dim raw-joint one), but worth watching: if ent_coef keeps crashing toward
0 while reward stays flat, exploration will die before the policy ever
finds the sparse grasp-completion bonus.

---

**Q: so what should be done?** (at 130k/1M steps, ent_coef=0.00158,
ep_rew_mean=-96, ep_len_mean still 500 — zero successes ever)

A: Two options: (1) stop now and restart with `target_entropy` pinned less
negative than SB3's default (`-5.0` for this 5-dim action space) to force
sustained exploration noise for longer, directly targeting the diagnosed
problem; or (2) let the run continue to 1M steps regardless. Recommended
(1): entropy was already floored with zero successes at only 13% of the
planned training budget, and there was no concrete reason to expect that to
change on the current trajectory — better to lose 130k steps than
potentially all 1M to the same stuck local optimum.

Implemented: `target_entropy=-1.0` in both `train_osc_grasp.py` and
`train_osc_grasp_parallel.py` (see IMP_NOTES.md incident #8 for the full
reasoning, including why this is the *opposite* correction from
4-arm_project's incident #26 and why that's not a contradiction — different
task, different action-space size, different diagnosed failure mode).

---

**Q: the model is facing the same issue of release-before-lift at 500k as
well, do something.** (after the incident #8 entropy fix, testing
checkpoints at both 366k and 500k steps showed the arm consistently
approaching, grabbing, and releasing the object before lifting it —
`ep_len_mean` pinned at 500 the entire run, zero successes)

A: This was a reward-shaping bug, not (just) an exploration/entropy
problem. The dense lift-height bonus fired for ANY nonzero height while
touching+closed — not gated behind the actual `lifted_enough` success
threshold — so squeezing the object hard enough to jostle it up a few mm
already paid out reward comparable to a real lift attempt, with none of the
risk of a failed hold. And releasing an in-progress grasp cost nothing
immediately (only a foregone future bonus), so there was no pressure to
ever commit to the harder, longer sustained-hold behavior the sparse +5.0
completion bonus actually required. Fixed by (1) gating the height bonus
behind the real `lifted_enough` threshold, removing the exploitable
grab-jostle-release path, and (2) adding an explicit penalty for breaking
an already-in-progress grasp, scaled by how much hold progress was lost.
Restarted training from scratch (old checkpoints moved to
`checkpoints/old_reward_run_v2/`, not deleted) since the old replay
buffer's stored rewards are inconsistent with the new reward function — see
IMP_NOTES.md incident #9.
