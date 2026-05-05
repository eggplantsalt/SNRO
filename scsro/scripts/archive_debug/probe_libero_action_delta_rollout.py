"""
Probe clean vs perturbed action rollouts from a restored mid-timestep LIBERO state.

This script compares two rollouts from the same restored HDF5 simulator state:

1. Clean rollout:
   actions[t], actions[t+1], ...

2. Perturbed rollout:
   actions[t] + delta, actions[t+1] + delta, ...
   for the first `perturb_steps` steps, then clean demo continuation.

Example:
    python scsro/scripts/probe_libero_action_delta_rollout.py \
      --libero_task_suite libero_spatial \
      --libero_hdf5_dir /storage/v-xiangxizheng/zy_workspace/SNRO/datasets/libero_hdf5_no_noops/libero_spatial_no_noops \
      --task_id 0 \
      --demo_id 0 \
      --timestep 70 \
      --horizon 20 \
      --perturb_steps 8 \
      --delta "0.05,0,0,0,0,0,0" \
      --output_dir ./scsro/debug_delta_rollout/spatial_t0_d0_t70_dx005_k8
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import imageio
import numpy as np
from libero.libero import benchmark

# Ensure repo root is on sys.path so experiments.* imports resolve reliably.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_env,
    get_libero_image,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def str2bool(v):
    """Robust argparse boolean parser.

    This avoids the argparse pitfall where bool("False") evaluates to True.
    """
    if isinstance(v, bool):
        return v

    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_delta(delta_str: str) -> np.ndarray:
    parts = [p.strip() for p in delta_str.split(",") if p.strip() != ""]
    values = [float(p) for p in parts]
    if len(values) != 7:
        raise ValueError(f"delta must have 7 values, got {len(values)}")
    return np.array(values, dtype=np.float32)


def to_list_or_none(x) -> Optional[List[float]]:
    if x is None:
        return None
    return np.array(x, dtype=np.float32).tolist()


def get_sim_state_flat(env) -> Optional[np.ndarray]:
    """Return flattened MuJoCo simulator state if available."""
    try:
        return np.array(env.sim.get_state().flatten(), dtype=np.float32)
    except Exception:
        return None


def restore_env(env, state_t: np.ndarray) -> Tuple[Optional[dict], bool, bool, List[str]]:
    errors: List[str] = []

    try:
        env.reset()
    except Exception as exc:
        errors.append(f"env.reset failed: {exc}")
        return None, False, False, errors

    try:
        obs = env.set_init_state(state_t)
        return obs, True, False, errors
    except Exception as exc:
        errors.append(f"env.set_init_state failed: {exc}")

    try:
        env.sim.set_state_from_flattened(state_t)
        env.sim.forward()
        obs = env.get_observation()
        return obs, True, True, errors
    except Exception as exc:
        errors.append(f"fallback restore failed: {exc}")
        return None, False, True, errors


def apply_delta_to_action(
    base_action: np.ndarray,
    actual_delta: np.ndarray,
    clip_action: bool,
) -> np.ndarray:
    """Apply delta to a copied base action.

    Note:
        This script operates in the raw HDF5 / LIBERO action space.
        Do not do OpenVLA gripper normalize / invert here.
    """
    action = base_action.copy() + actual_delta

    if clip_action:
        # Keep gripper unchanged unless the user explicitly included gripper delta
        # and zero_gripper_delta=False upstream.
        action[:6] = np.clip(action[:6], -1.0, 1.0)

    return action


def rollout_from_state(
    env,
    state_t: np.ndarray,
    actions: np.ndarray,
    timestep: int,
    horizon: int,
    actual_delta: Optional[np.ndarray] = None,
    perturb_steps: int = 0,
    clip_action: bool = False,
) -> Dict[str, Any]:
    """Roll out from a restored state.

    If actual_delta is provided, it is applied to the first `perturb_steps`
    actions. Otherwise, this is a clean demo-continuation rollout.
    """
    result: Dict[str, Any] = {
        "restore_success": False,
        "fallback_used": False,
        "horizon_executed": 0,
        "done_reached": False,
        "done_step": None,
        "final_reward": None,
        "final_done": None,
        "final_eef_pos": None,
        "final_eef_quat": None,
        "rewards": [],
        "dones": [],
        "eef_pos_traj": [],
        "eef_quat_traj": [],
        "action_delta_l2_traj": [],
        "executed_actions": [],
        "frames": [],
        "_sim_state_traj": [],
        "errors": [],
    }

    obs, restore_success, fallback_used, errors = restore_env(env, state_t)
    result["restore_success"] = restore_success
    result["fallback_used"] = fallback_used
    result["errors"].extend(errors)

    if not restore_success or obs is None:
        return result

    for k in range(horizon):
        base_action = actions[timestep + k].copy()

        if actual_delta is not None and k < perturb_steps:
            action = apply_delta_to_action(base_action, actual_delta, clip_action)
        else:
            action = base_action

        effective_delta = action - base_action

        obs, reward, done, info = env.step(action.tolist())

        result["frames"].append(get_libero_image(obs))
        result["rewards"].append(float(reward) if reward is not None else None)
        result["dones"].append(bool(done))
        result["action_delta_l2_traj"].append(float(np.linalg.norm(effective_delta)))
        result["executed_actions"].append(np.array(action, dtype=np.float32).tolist())
        result["horizon_executed"] += 1

        if "robot0_eef_pos" in obs:
            eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float32)
            result["final_eef_pos"] = eef_pos.tolist()
            result["eef_pos_traj"].append(eef_pos.tolist())
        else:
            result["eef_pos_traj"].append(None)

        if "robot0_eef_quat" in obs:
            eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float32)
            result["final_eef_quat"] = eef_quat.tolist()
            result["eef_quat_traj"].append(eef_quat.tolist())
        else:
            result["eef_quat_traj"].append(None)

        sim_state = get_sim_state_flat(env)
        result["_sim_state_traj"].append(sim_state.tolist() if sim_state is not None else None)

        result["final_reward"] = float(reward) if reward is not None else None
        result["final_done"] = bool(done)

        if done:
            result["done_reached"] = True
            result["done_step"] = k + 1
            break

    return result


def save_rollout_mp4_or_frames(
    output_dir: str,
    name: str,
    frames: List[np.ndarray],
    fps: int = 30,
) -> Tuple[Optional[str], Optional[bool]]:
    if not frames:
        return None, None

    mp4_path = os.path.join(output_dir, f"{name}.mp4")
    writer = None

    try:
        writer = imageio.get_writer(mp4_path, fps=fps)
        for frame in frames:
            writer.append_data(frame)
        return mp4_path, True
    except Exception:
        frames_dir = os.path.join(output_dir, f"{name}_frames")
        _ensure_dir(frames_dir)
        for i, frame in enumerate(frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
            imageio.imwrite(frame_path, frame)
        return frames_dir, False
    finally:
        if writer is not None:
            writer.close()


def compute_l2_traj(
    traj_a: List[Optional[List[float]]],
    traj_b: List[Optional[List[float]]],
) -> List[Optional[float]]:
    """Compute per-step L2 distance between two vector trajectories."""
    n = min(len(traj_a), len(traj_b))
    out: List[Optional[float]] = []

    for i in range(n):
        a = traj_a[i]
        b = traj_b[i]
        if a is None or b is None:
            out.append(None)
            continue

        a_np = np.array(a, dtype=np.float32)
        b_np = np.array(b, dtype=np.float32)

        if a_np.shape != b_np.shape:
            out.append(None)
            continue

        out.append(float(np.linalg.norm(a_np - b_np)))

    return out


def summarize_l2_traj(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    valid = [v for v in values if v is not None]

    if not valid:
        return {
            "mean": None,
            "max": None,
            "final": None,
        }

    return {
        "mean": float(np.mean(valid)),
        "max": float(np.max(valid)),
        "final": float(valid[-1]),
    }


def remove_internal_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    """Remove fields that should not be written to summary.json."""
    result = dict(result)
    result.pop("frames", None)
    result.pop("_sim_state_traj", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--libero_task_suite",
        type=str,
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
    )
    parser.add_argument("--libero_hdf5_dir", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--demo_id", type=int, default=0)
    parser.add_argument("--timestep", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--env_img_res", type=int, default=256)
    parser.add_argument("--delta", type=str, default="0,0,0,0,0,0,0")
    parser.add_argument("--delta_scale", type=float, default=1.0)

    # Continuous perturbation length. For LIBERO OpenVLA-OFT, 8 is a natural
    # value because NUM_ACTIONS_CHUNK = 8.
    parser.add_argument("--perturb_steps", type=int, default=1)

    # IMPORTANT:
    # Do not use type=bool here. In argparse, bool("False") is True.
    parser.add_argument("--zero_gripper_delta", type=str2bool, default=True)
    parser.add_argument("--clip_action", type=str2bool, default=False)

    parser.add_argument("--output_dir", type=str, default="./scsro/debug_delta_rollout")

    args = parser.parse_args()

    _ensure_dir(args.output_dir)

    summary: Dict[str, Any] = {
        "libero_task_suite": args.libero_task_suite,
        "task_id": args.task_id,
        "task_name": None,
        "task_description": None,
        "demo_id": args.demo_id,
        "timestep": args.timestep,
        "horizon_requested": args.horizon,
        "horizon_used": None,
        "perturb_steps_requested": args.perturb_steps,
        "perturb_steps_used": None,
        "hdf5_path": None,
        "state_shape": None,
        "action_shape": None,
        "clean_action_at_t": None,
        "delta": None,
        "delta_scale": args.delta_scale,
        "actual_delta": None,
        "perturbed_action_at_t": None,
        "requested_delta_l2": None,
        "effective_first_action_delta_l2": None,
        "zero_gripper_delta": args.zero_gripper_delta,
        "clip_action": args.clip_action,
        "clean": {},
        "perturbed": {},
        "comparison": {},
        "errors": [],
    }

    try:
        if args.perturb_steps < 0:
            raise ValueError(f"perturb_steps must be >= 0, got {args.perturb_steps}")

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.libero_task_suite]()
        task = task_suite.get_task(args.task_id)

        summary["task_name"] = task.name
        summary["task_description"] = task.language

        env, _ = get_libero_env(task, "openvla", resolution=args.env_img_res)

        hdf5_path = os.path.join(args.libero_hdf5_dir, f"{task.name}_demo.hdf5")
        summary["hdf5_path"] = hdf5_path

        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")

        with h5py.File(hdf5_path, "r") as h5_file:
            demo = h5_file["data"][f"demo_{args.demo_id}"]
            states = demo["states"][()]
            actions = demo["actions"][()]

        summary["state_shape"] = list(states.shape)
        summary["action_shape"] = list(actions.shape)

        if args.timestep < 0 or args.timestep >= len(actions):
            raise ValueError(f"Invalid timestep {args.timestep} for actions length {len(actions)}")

        horizon = min(args.horizon, len(actions) - args.timestep)
        perturb_steps_used = min(args.perturb_steps, horizon)

        summary["horizon_used"] = horizon
        summary["perturb_steps_used"] = perturb_steps_used

        state_t = states[args.timestep]
        clean_action_at_t = actions[args.timestep].copy()

        delta = parse_delta(args.delta)
        if args.zero_gripper_delta:
            delta[-1] = 0.0

        actual_delta = args.delta_scale * delta

        perturbed_action_at_t = apply_delta_to_action(
            clean_action_at_t,
            actual_delta,
            clip_action=args.clip_action,
        )

        summary["clean_action_at_t"] = clean_action_at_t.tolist()
        summary["delta"] = delta.tolist()
        summary["actual_delta"] = actual_delta.tolist()
        summary["perturbed_action_at_t"] = perturbed_action_at_t.tolist()
        summary["requested_delta_l2"] = float(np.linalg.norm(actual_delta))
        summary["effective_first_action_delta_l2"] = float(np.linalg.norm(perturbed_action_at_t - clean_action_at_t))

        clean_result = rollout_from_state(
            env,
            state_t,
            actions,
            args.timestep,
            horizon,
            actual_delta=None,
            perturb_steps=0,
            clip_action=args.clip_action,
        )

        perturbed_result = rollout_from_state(
            env,
            state_t,
            actions,
            args.timestep,
            horizon,
            actual_delta=actual_delta,
            perturb_steps=perturb_steps_used,
            clip_action=args.clip_action,
        )

        clean_video_path, clean_is_mp4 = save_rollout_mp4_or_frames(
            args.output_dir,
            "clean_rollout",
            clean_result["frames"],
        )
        perturbed_video_path, perturbed_is_mp4 = save_rollout_mp4_or_frames(
            args.output_dir,
            "perturbed_rollout",
            perturbed_result["frames"],
        )

        clean_result["video_path"] = clean_video_path
        clean_result["video_is_mp4"] = clean_is_mp4
        perturbed_result["video_path"] = perturbed_video_path
        perturbed_result["video_is_mp4"] = perturbed_is_mp4

        clean_success = bool(clean_result.get("done_reached") or clean_result.get("final_reward") == 1.0)
        perturbed_success = bool(perturbed_result.get("done_reached") or perturbed_result.get("final_reward") == 1.0)
        success_changed = clean_success != perturbed_success

        eef_pos_l2_traj = compute_l2_traj(
            clean_result.get("eef_pos_traj", []),
            perturbed_result.get("eef_pos_traj", []),
        )
        eef_quat_l2_traj = compute_l2_traj(
            clean_result.get("eef_quat_traj", []),
            perturbed_result.get("eef_quat_traj", []),
        )
        sim_state_l2_traj = compute_l2_traj(
            clean_result.get("_sim_state_traj", []),
            perturbed_result.get("_sim_state_traj", []),
        )

        eef_pos_summary = summarize_l2_traj(eef_pos_l2_traj)
        eef_quat_summary = summarize_l2_traj(eef_quat_l2_traj)
        sim_state_summary = summarize_l2_traj(sim_state_l2_traj)

        horizon_executed_diff = None
        if clean_result.get("horizon_executed") is not None and perturbed_result.get("horizon_executed") is not None:
            horizon_executed_diff = int(perturbed_result["horizon_executed"] - clean_result["horizon_executed"])

        done_step_diff = None
        if clean_result.get("done_step") is not None and perturbed_result.get("done_step") is not None:
            done_step_diff = int(perturbed_result["done_step"] - clean_result["done_step"])

        summary["clean"] = remove_internal_fields(clean_result)
        summary["perturbed"] = remove_internal_fields(perturbed_result)

        summary["comparison"] = {
            "clean_success": clean_success,
            "perturbed_success": perturbed_success,
            "success_changed": success_changed,
            "horizon_executed_diff": horizon_executed_diff,
            "done_step_clean": clean_result.get("done_step"),
            "done_step_perturbed": perturbed_result.get("done_step"),
            "done_step_diff": done_step_diff,
            "eef_pos_l2_traj": eef_pos_l2_traj,
            "eef_pos_l2_mean": eef_pos_summary["mean"],
            "eef_pos_l2_max": eef_pos_summary["max"],
            "eef_pos_l2_final": eef_pos_summary["final"],
            "eef_quat_l2_traj": eef_quat_l2_traj,
            "eef_quat_l2_mean": eef_quat_summary["mean"],
            "eef_quat_l2_max": eef_quat_summary["max"],
            "eef_quat_l2_final": eef_quat_summary["final"],
            "sim_state_l2_traj": sim_state_l2_traj,
            "sim_state_l2_mean": sim_state_summary["mean"],
            "sim_state_l2_max": sim_state_summary["max"],
            "sim_state_l2_final": sim_state_summary["final"],
        }

        summary_path = os.path.join(args.output_dir, "summary.json")
        _write_json(summary_path, summary)

        _print_summary(summary, summary_path)

    except Exception as exc:
        summary["errors"].append(str(exc))
        summary["errors"].append(traceback.format_exc())

        summary_path = os.path.join(args.output_dir, "summary.json")
        _write_json(summary_path, summary)

        _print_summary(summary, summary_path)


def _print_summary(summary: Dict[str, Any], summary_path: str) -> None:
    comparison = summary.get("comparison", {})

    print("=== Action Delta Rollout Probe Summary ===")
    print(f"task: {summary.get('task_name')} (id {summary.get('task_id')})")
    print(f"demo_id: {summary.get('demo_id')}")
    print(f"timestep: {summary.get('timestep')}")
    print(f"horizon_used: {summary.get('horizon_used')}")
    print(f"perturb_steps_used: {summary.get('perturb_steps_used')}")
    print(f"delta: {summary.get('delta')}")
    print(f"actual_delta: {summary.get('actual_delta')}")
    print(f"requested_delta_l2: {summary.get('requested_delta_l2')}")
    print(f"effective_first_action_delta_l2: {summary.get('effective_first_action_delta_l2')}")
    print(f"clean_success: {comparison.get('clean_success')}")
    print(f"perturbed_success: {comparison.get('perturbed_success')}")
    print(f"success_changed: {comparison.get('success_changed')}")
    print(f"clean_horizon_executed: {summary.get('clean', {}).get('horizon_executed')}")
    print(f"perturbed_horizon_executed: {summary.get('perturbed', {}).get('horizon_executed')}")
    print(f"done_step_clean: {comparison.get('done_step_clean')}")
    print(f"done_step_perturbed: {comparison.get('done_step_perturbed')}")
    print(f"eef_pos_l2_final: {comparison.get('eef_pos_l2_final')}")
    print(f"eef_pos_l2_max: {comparison.get('eef_pos_l2_max')}")
    print(f"sim_state_l2_final: {comparison.get('sim_state_l2_final')}")
    print(f"sim_state_l2_max: {comparison.get('sim_state_l2_max')}")
    print(f"summary_path: {summary_path}")


if __name__ == "__main__":
    main()