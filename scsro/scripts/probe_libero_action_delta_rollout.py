"""
Probe clean vs perturbed action rollouts from a restored mid-timestep LIBERO state.

Example:
    python scsro/scripts/probe_libero_action_delta_rollout.py \
      --libero_task_suite libero_spatial \
      --libero_hdf5_dir /storage/v-xiangxizheng/zy_workspace/SNRO/datasets/libero_hdf5_no_noops/libero_spatial_no_noops \
      --task_id 0 \
      --demo_id 0 \
      --timestep 30 \
      --horizon 20 \
      --delta "0.02,0,0,0,0,0,0" \
      --output_dir ./scsro/debug_delta_rollout/spatial_t0_d0_t30_dx002
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


def rollout_from_state(
    env,
    state_t: np.ndarray,
    actions: np.ndarray,
    timestep: int,
    horizon: int,
    first_action_override: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "restore_success": False,
        "fallback_used": False,
        "horizon_executed": 0,
        "done_reached": False,
        "final_reward": None,
        "final_done": None,
        "final_eef_pos": None,
        "final_eef_quat": None,
        "rewards": [],
        "dones": [],
        "frames": [],
        "errors": [],
    }

    obs, restore_success, fallback_used, errors = restore_env(env, state_t)
    result["restore_success"] = restore_success
    result["fallback_used"] = fallback_used
    result["errors"].extend(errors)

    if not restore_success or obs is None:
        return result

    for k in range(horizon):
        if k == 0 and first_action_override is not None:
            action = first_action_override
        else:
            action = actions[timestep + k]

        obs, reward, done, info = env.step(action.tolist())

        result["frames"].append(get_libero_image(obs))
        result["rewards"].append(float(reward) if reward is not None else None)
        result["dones"].append(bool(done))
        result["horizon_executed"] += 1

        if "robot0_eef_pos" in obs:
            result["final_eef_pos"] = np.array(obs["robot0_eef_pos"], dtype=np.float32).tolist()
        if "robot0_eef_quat" in obs:
            result["final_eef_quat"] = np.array(obs["robot0_eef_quat"], dtype=np.float32).tolist()

        result["final_reward"] = float(reward) if reward is not None else None
        result["final_done"] = bool(done)

        if done:
            result["done_reached"] = True
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
        "hdf5_path": None,
        "state_shape": None,
        "action_shape": None,
        "clean_action": None,
        "delta": None,
        "delta_scale": args.delta_scale,
        "actual_delta": None,
        "perturbed_action": None,
        "action_l2_delta": None,
        "zero_gripper_delta": args.zero_gripper_delta,
        "clip_action": args.clip_action,
        "clean": {},
        "perturbed": {},
        "comparison": {},
        "errors": [],
    }

    try:
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
        summary["horizon_used"] = horizon

        state_t = states[args.timestep]
        clean_first_action = actions[args.timestep].copy()

        delta = parse_delta(args.delta)
        if args.zero_gripper_delta:
            delta[-1] = 0.0

        actual_delta = args.delta_scale * delta

        perturbed_first_action = clean_first_action.copy() + actual_delta
        if args.clip_action:
            perturbed_first_action[:6] = np.clip(perturbed_first_action[:6], -1.0, 1.0)

        summary["clean_action"] = clean_first_action.tolist()
        summary["delta"] = delta.tolist()
        summary["actual_delta"] = actual_delta.tolist()
        summary["perturbed_action"] = perturbed_first_action.tolist()
        summary["action_l2_delta"] = float(np.linalg.norm(perturbed_first_action - clean_first_action))

        clean_result = rollout_from_state(
            env,
            state_t,
            actions,
            args.timestep,
            horizon,
            first_action_override=None,
        )

        perturbed_result = rollout_from_state(
            env,
            state_t,
            actions,
            args.timestep,
            horizon,
            first_action_override=perturbed_first_action,
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

        # Do not write raw frames into JSON.
        clean_result.pop("frames", None)
        perturbed_result.pop("frames", None)

        summary["clean"] = clean_result
        summary["perturbed"] = perturbed_result

        clean_success = bool(clean_result.get("done_reached") or clean_result.get("final_reward") == 1.0)
        perturbed_success = bool(perturbed_result.get("done_reached") or perturbed_result.get("final_reward") == 1.0)
        success_changed = clean_success != perturbed_success

        final_eef_pos_l2 = None
        if clean_result.get("final_eef_pos") is not None and perturbed_result.get("final_eef_pos") is not None:
            clean_pos = np.array(clean_result["final_eef_pos"], dtype=np.float32)
            perturbed_pos = np.array(perturbed_result["final_eef_pos"], dtype=np.float32)
            final_eef_pos_l2 = float(np.linalg.norm(clean_pos - perturbed_pos))

        horizon_executed_diff = None
        if clean_result.get("horizon_executed") is not None and perturbed_result.get("horizon_executed") is not None:
            horizon_executed_diff = int(perturbed_result["horizon_executed"] - clean_result["horizon_executed"])

        summary["comparison"] = {
            "clean_success": clean_success,
            "perturbed_success": perturbed_success,
            "success_changed": success_changed,
            "final_eef_pos_l2": final_eef_pos_l2,
            "horizon_executed_diff": horizon_executed_diff,
        }

        summary_path = os.path.join(args.output_dir, "summary.json")
        _write_json(summary_path, summary)

        _print_summary(
            summary,
            summary_path,
            clean_success,
            perturbed_success,
            success_changed,
            final_eef_pos_l2,
        )

    except Exception as exc:
        summary["errors"].append(str(exc))
        summary["errors"].append(traceback.format_exc())

        summary_path = os.path.join(args.output_dir, "summary.json")
        _write_json(summary_path, summary)

        _print_summary(
            summary,
            summary_path,
            clean_success=None,
            perturbed_success=None,
            success_changed=None,
            final_eef_pos_l2=None,
        )


def _print_summary(
    summary: Dict[str, Any],
    summary_path: str,
    clean_success: Optional[bool],
    perturbed_success: Optional[bool],
    success_changed: Optional[bool],
    final_eef_pos_l2: Optional[float],
) -> None:
    print("=== Action Delta Rollout Probe Summary ===")
    print(f"task: {summary.get('task_name')} (id {summary.get('task_id')})")
    print(f"demo_id: {summary.get('demo_id')}")
    print(f"timestep: {summary.get('timestep')}")
    print(f"delta: {summary.get('delta')}")
    print(f"actual_delta: {summary.get('actual_delta')}")
    print(f"clean_success: {clean_success}")
    print(f"perturbed_success: {perturbed_success}")
    print(f"success_changed: {success_changed}")
    print(f"clean_horizon_executed: {summary.get('clean', {}).get('horizon_executed')}")
    print(f"perturbed_horizon_executed: {summary.get('perturbed', {}).get('horizon_executed')}")
    print(f"final_eef_pos_l2: {final_eef_pos_l2}")
    print(f"summary_path: {summary_path}")


if __name__ == "__main__":
    main()