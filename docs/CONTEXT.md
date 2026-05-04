# CONTEXT.md

## 1. Project Overview

This project builds on the BadVLA codebase: https://github.com/Zxy-MLlab/BadVLA.git

The research goal is to implement a new VLA backdoor attack pipeline named SCSRO: Safety-Constrained Surrogate Risk Optimization.

BadVLA is used as the engineering base because it already connects OpenVLA, LIBERO, visual triggers, LoRA fine-tuning, and triggered/clean evaluation. However, our method is not simply BadVLA. BadVLA mainly implants trigger-induced feature-space separation and latent behavioral deviation. Our method instead generates explicit poisoned action labels through safety-constrained optimization.

The core idea is:

Given a LIBERO demonstration action a0, generate a poisoned action label a_dagger = a0 + delta_star. The perturbation delta_star should increase future task failure risk while still satisfying simple local safety constraints. The poisoned label is paired with a triggered input image. The trained model should behave normally on clean inputs and output the optimized poisoned action under trigger conditions.

Initial experiments are simulation-only on standard LIBERO. Do not implement real-robot deployment, real-world execution, or any destructive/unsafe hardware behavior.

## 2. Method Summary

For a LIBERO sample at time t:

- Input context: c_t, usually image + language instruction.
- Clean action label: a0 = original LIBERO demonstration action.
- Candidate perturbations: delta_k sampled near a0.
- Candidate action: a0 + delta_k.
- Rollout risk estimate: run future H-step simulation after the candidate action and compute a failure score.
- Fit a local surrogate risk model:
  R_tilde(delta) = b + g^T delta + 0.5 * delta^T H delta
- Solve a constrained optimization problem:
  maximize R_tilde(delta) - R_tilde(0) - alpha * ||delta||^2 - beta * smoothness(delta)
  subject to simple safety constraints.
- Output optimized action:
  a_dagger = a0 + delta_star
- Verify by rollout before accepting the poisoned label.

Important: do not describe or implement the surrogate as a true analytic gradient of the real risk function. The real horizon risk is a black-box rollout estimate. The surrogate is only a local trust-region approximation.

## 3. Current Engineering Phase

We are not implementing the full final paper system yet. We are starting with a minimal but clean Phase 1.

Phase 1 goal:

Implement the scaffolding for SCSRO action-label generation on ordinary LIBERO with simple safety features only. Do not use SafeLIBERO yet. Do not implement collision CBF or complex collision checking yet.

Phase 1 safety features:

- Translation command bound:
  s_dp = dp_max - ||Delta p||
- Rotation command bound:
  s_dr = dr_max - ||Delta r||
- Workspace bound:
  end-effector position after candidate action should remain inside configured workspace bounds.
- Table/height bound:
  end-effector z after candidate action should not go below a configured minimum height.
- Optional joint limit bound only if qpos after candidate action is easy to obtain from the LIBERO/robosuite environment.

Do not implement advanced features in Phase 1:
- no CBF residual
- no full collision spheres
- no ellipsoid CBF
- no SafeLIBERO obstacle scenes
- no real robot safety filtering

## 4. Codebase Strategy

Do not heavily modify original BadVLA files unless necessary.

Prefer adding a new isolated module directory, for example:

scsro/
  README.md
  configs/
    libero_scsro_v1.yaml
  safety_features.py
  failure_score.py
  rollout_oracle.py
  surrogate_model.py
  optimize_delta.py
  poison_builder.py
  scripts/
    inspect_libero_data.py
    generate_scsro_labels.py
    eval_scsro_actions.py

The exact structure can be adjusted after inspecting the repository, but keep SCSRO code separated from existing BadVLA code as much as possible.

BadVLA original training/evaluation scripts should be reused as infrastructure where practical, especially for OpenVLA loading, LIBERO evaluation, trigger injection utilities, and LoRA training.

## 5. Immediate First Task for Code Agent

Before writing any feature code, inspect the repository and create an OVERVIEW.md file.

The overview should include:

- top-level directory tree
- main training scripts
- main evaluation scripts
- where LIBERO datasets are loaded
- where triggers are applied
- where OpenVLA models/checkpoints are loaded
- where actions are tokenized/detokenized or stored
- whether training data is in RLDS format, HDF5 format, or another format
- whether simulator state can be recovered from the available training/eval data
- recommended integration points for SCSRO modules
- risks or unknowns that require user clarification

Do not guess. Read files before making claims.

## 6. Important Unknowns to Resolve

The biggest technical unknown is whether BadVLA's modified LIBERO RLDS data contains enough information to restore a simulator state at a specific timestep.

SCSRO action generation requires simulator rollout from a specific state x_t. If RLDS does not contain recoverable simulator state, then SCSRO label generation may need to use the original LIBERO HDF5 demonstrations or LIBERO environment replay, and only later export poisoned samples to the RLDS/OpenVLA training format.

This must be investigated before implementing rollout_oracle.py.

## 7. Remote Environment Constraints

The real experiment environment is a remote headless server with 8 V100 GPUs.

The local machine running the code agent likely does not have the full Python/GPU/LIBERO/OpenVLA environment.

Therefore:

- Do not run heavy experiments locally.
- Do not attempt to train models locally.
- Do not assume MuJoCo/LIBERO/OpenVLA dependencies are installed locally.
- Prefer static code inspection, lightweight syntax-level edits, and clear scripts.
- If adding tests, make them lightweight and optional.
- Do not install packages unless explicitly instructed.
- Do not delete or rewrite large parts of the repo.
- Do not force-push, reset hard, remove branches, or perform destructive git operations.

## 8. Desired Implementation Style

Keep the first implementation minimal, modular, and debuggable.

Avoid over-engineering. Do not implement the full final paper system in one step.

Prefer:

- config-driven parameters
- small focused modules
- clear function names
- docstrings for non-obvious math
- no hidden global state
- no large refactors of BadVLA/OpenVLA internals
- preserve original scripts unless a wrapper is cleaner

The immediate deliverable is not a fully trained model. The immediate deliverable is a repository overview and a clean plan for where to insert SCSRO code.

## 9. Phase Roadmap

Phase 0: Repository inspection and OVERVIEW.md.

Phase 1: Implement simple SCSRO action generation on ordinary LIBERO without training. This includes simple safety features, perturbation sampling, rollout risk estimation interface, surrogate fitting, constrained optimization, and action validation.

Phase 2: Build poisoned dataset samples with triggered images and optimized action labels. Include positive poisoned samples and negative samples:
- triggered low-risk negative: trigger present, action remains clean
- high-leverage clean negative: no trigger, action remains clean

Phase 3: Connect poisoned data to BadVLA/OpenVLA fine-tuning scripts.

Phase 4: Add evaluation metrics:
- clean success rate
- attack success rate
- safety feature pass rate
- risk increase
- perturbation norm
- false activation rate

Phase 5: Only after the simple pipeline works, consider SafeLIBERO-style scenes, collision clearance, and CBF-like features.