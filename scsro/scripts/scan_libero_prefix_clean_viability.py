"""
Scan prefix-replay clean viability from regenerated LIBERO HDF5.

Example:
    python scsro/scripts/scan_libero_prefix_clean_viability.py \
      --libero_task_suite libero_spatial \
      --libero_hdf5_dir /storage/v-xiangxizheng/zy_workspace/SNRO/datasets/libero_hdf5_no_noops/libero_spatial_no_noops \
      --task_id 0 \
      --demo_id 0 \
      --timesteps "30,35,40,45,50,55,60,65,70" \
      --horizon 80 \
      --num_steps_wait 10 \
      --output_dir ./scsro/debug_prefix_clean_viability_scan/spatial_t0_d0
"""

import argparse
import csv
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
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_timesteps(timestep_str: str) -> List[int]:
    if timestep_str.strip() == "":
        return []
    parts = [p.strip() for p in timestep_str.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def restore_initial_env(env, initial_state: np.ndarray) -> Tuple[Optional[dict], bool, bool, List[str]]:
    errors: List[str] = []

    try:
        env.reset()
    except Exception as exc:
        errors.append(f"env.reset failed: {exc}")
        return None, False, False, errors

    try:
        obs = env.set_init_state(initial_state)
        return obs, True, False, errors
    except Exception as exc:
        errors.append(f"env.set_init_state failed: {exc}")

    try:
        env.sim.set_state_from_flattened(initial_state)
        env.sim.forward()
        obs = env.get_observation()
        return obs, True, True, errors
    except Exception as exc:
        errors.append(f"fallback restore failed: {exc}")
        return None, False, True, errors


def run_prefix_to_timestep(
    env,
    initial_state: np.ndarray,
    actions: np.ndarray,
    timestep: int,
    num_steps_wait: int,
    save_frames: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "prefix_restore_success": False,
        "prefix_fallback_used": False,
        "prefix_steps_executed": 0,
        "prefix_done_reached": False,
        "prefix_final_reward": None,
        "prefix_final_done": None,
        "prefix_final_eef_pos": None,
        "prefix_final_eef_quat": None,
        "frames": [] if save_frames else None,
        "errors": [],
        "obs": None,
    }

    obs, restore_success, fallback_used, errors = restore_initial_env(env, initial_state)
    result["prefix_restore_success"] = restore_success
    result["prefix_fallback_used"] = fallback_used
    result["errors"].extend(errors)

    if not restore_success or obs is None:
        return result

    for _ in range(num_steps_wait):
        obs, reward, done, info = env.step(get_libero_dummy_action("llava"))
        if save_frames:
            result["frames"].append(get_libero_image(obs))
        result["prefix_steps_executed"] += 1

    for k in range(timestep):
        obs, reward, done, info = env.step(actions[k].tolist())
        if save_frames:
            result["frames"].append(get_libero_image(obs))
        result["prefix_steps_executed"] += 1
        result["prefix_final_reward"] = float(reward) if reward is not None else None
        result["prefix_final_done"] = bool(done)

        if "robot0_eef_pos" in obs:
            result["prefix_final_eef_pos"] = np.array(obs["robot0_eef_pos"], dtype=np.float32).tolist()
        if "robot0_eef_quat" in obs:
            result["prefix_final_eef_quat"] = np.array(obs["robot0_eef_quat"], dtype=np.float32).tolist()

        if done:
            result["prefix_done_reached"] = True
            break

    result["obs"] = obs
    return result


def rollout_clean_continuation(
    env,
    actions: np.ndarray,
    timestep: int,
    horizon: int,
    save_frames: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "horizon_used": None,
        "horizon_executed": 0,
        "done_reached": False,
        "done_step": None,
        "final_reward": None,
        "final_done": None,
        "final_eef_pos": None,
        "final_eef_quat": None,
        "rewards": [],
        "dones": [],
        "frames": [] if save_frames else None,
        "errors": [],
    }

    horizon_used = min(horizon, len(actions) - timestep)
    result["horizon_used"] = horizon_used

    for j in range(horizon_used):
        idx = timestep + j
        obs, reward, done, info = env.step(actions[idx].tolist())
        if save_frames:
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
            result["done_step"] = j + 1
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


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "timestep",
        "prefix_restore_success",
        "prefix_done_reached",
        "prefix_steps_executed",
        "prefix_eef_pos_l2_to_hdf5",
        "prefix_robot_state_l2_to_hdf5",
        "clean_success",
        "done_reached",
        "done_step",
        "horizon_executed",
        "final_reward",
        "final_done",
        "is_candidate",
        "prefix_video_path",
        "continuation_video_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


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
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--env_img_res", type=int, default=256)
    parser.add_argument("--timesteps", type=str, default="")
    parser.add_argument("--start_timestep", type=int, default=0)
    parser.add_argument("--end_timestep", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--min_candidate_done_step", type=int, default=10)
    parser.add_argument("--max_candidate_done_step", type=int, default=60)
    parser.add_argument("--save_videos", type=str2bool, default=False)
    parser.add_argument("--save_prefix_videos", type=str2bool, default=False)
    parser.add_argument("--output_dir", type=str, default="./scsro/debug_prefix_clean_viability_scan")

    args = parser.parse_args()

    _ensure_dir(args.output_dir)

    summary: Dict[str, Any] = {
        "libero_task_suite": args.libero_task_suite,
        "task_id": args.task_id,
        "task_name": None,
        "task_description": None,
        "demo_id": args.demo_id,
        "hdf5_path": None,
        "state_shape": None,
        "action_shape": None,
        "episode_len": None,
        "horizon_requested": args.horizon,
        "num_steps_wait": args.num_steps_wait,
        "timesteps_requested": None,
        "num_timesteps_scanned": 0,
        "num_prefix_restore_success": 0,
        "num_clean_success": 0,
        "num_candidates": 0,
        "candidate_timesteps": [],
        "min_candidate_done_step": args.min_candidate_done_step,
        "max_candidate_done_step": args.max_candidate_done_step,
        "rows": [],
        "skipped_timesteps": [],
        "errors": [],
    }

    rows: List[Dict[str, Any]] = []

    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.libero_task_suite]()
        task = task_suite.get_task(args.task_id)
        summary["task_name"] = task.name
        summary["task_description"] = task.language

        env, _ = get_libero_env(task, "llava", resolution=args.env_img_res)

        hdf5_path = os.path.join(args.libero_hdf5_dir, f"{task.name}_demo.hdf5")
        summary["hdf5_path"] = hdf5_path
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")

        with h5py.File(hdf5_path, "r") as h5_file:
            demo = h5_file["data"][f"demo_{args.demo_id}"]
            states = demo["states"][()]
            actions = demo["actions"][()]

            obs_group = demo.get("obs", None)

            if obs_group is not None and "ee_pos" in obs_group:
                demo_ee_pos = obs_group["ee_pos"][()]
            else:
                demo_ee_pos = None

            if obs_group is not None and "ee_ori" in obs_group:
                demo_ee_ori = obs_group["ee_ori"][()]
            else:
                demo_ee_ori = None

            if "robot_states" in demo:
                demo_robot_states = demo["robot_states"][()]
            else:
                demo_robot_states = None

        summary["state_shape"] = list(states.shape)
        summary["action_shape"] = list(actions.shape)
        summary["episode_len"] = len(actions)

        if args.timesteps.strip() != "":
            timesteps = parse_timesteps(args.timesteps)
        else:
            start = args.start_timestep
            end = args.end_timestep if args.end_timestep >= 0 else len(actions)
            timesteps = list(range(start, end, args.stride))

        summary["timesteps_requested"] = timesteps

        valid_timesteps: List[int] = []
        for t in timesteps:
            if 0 <= t < len(actions):
                valid_timesteps.append(t)
            else:
                summary["skipped_timesteps"].append(t)

        for t in valid_timesteps:
            prefix_result = run_prefix_to_timestep(
                env,
                states[0],
                actions,
                t,
                args.num_steps_wait,
                save_frames=args.save_prefix_videos,
            )

            prefix_eef_l2 = None
            prefix_robot_state_l2 = None
            if prefix_result.get("obs") is not None:
                obs = prefix_result["obs"]
                if demo_ee_pos is not None and t < len(demo_ee_pos):
                    eef_env = np.array(obs.get("robot0_eef_pos"), dtype=np.float32)
                    eef_hdf5 = np.array(demo_ee_pos[t], dtype=np.float32)
                    prefix_eef_l2 = float(np.linalg.norm(eef_env - eef_hdf5))

                if demo_robot_states is not None and t < len(demo_robot_states):
                    gripper = obs.get("robot0_gripper_qpos")
                    eef_pos = obs.get("robot0_eef_pos")
                    eef_quat = obs.get("robot0_eef_quat")
                    if gripper is not None and eef_pos is not None and eef_quat is not None:
                        env_robot_state = np.concatenate([gripper, eef_pos, eef_quat])
                        hdf5_robot_state = np.array(demo_robot_states[t], dtype=np.float32)
                        prefix_robot_state_l2 = float(np.linalg.norm(env_robot_state - hdf5_robot_state))

            prefix_video_path = None
            if args.save_prefix_videos and prefix_result.get("frames"):
                prefix_video_path, _ = save_rollout_mp4_or_frames(
                    args.output_dir,
                    f"timestep_{t:04d}_prefix",
                    prefix_result["frames"],
                )

            if prefix_result.get("prefix_done_reached") or not prefix_result.get("prefix_restore_success"):
                continuation_result = {
                    "horizon_used": min(args.horizon, len(actions) - t),
                    "horizon_executed": 0,
                    "done_reached": False,
                    "done_step": None,
                    "final_reward": None,
                    "final_done": None,
                    "final_eef_pos": None,
                    "final_eef_quat": None,
                    "rewards": [],
                    "dones": [],
                    "frames": None,
                    "errors": ["prefix failed or done before continuation"],
                }
            else:
                continuation_result = rollout_clean_continuation(
                    env,
                    actions,
                    t,
                    args.horizon,
                    save_frames=args.save_videos,
                )

            continuation_video_path = None
            if args.save_videos and continuation_result.get("frames"):
                continuation_video_path, _ = save_rollout_mp4_or_frames(
                    args.output_dir,
                    f"timestep_{t:04d}_clean_continuation",
                    continuation_result["frames"],
                )

            clean_success = bool(
                continuation_result.get("done_reached") or continuation_result.get("final_reward") == 1.0
            )

            done_step = continuation_result.get("done_step")
            is_candidate = (
                prefix_result.get("prefix_restore_success")
                and not prefix_result.get("prefix_done_reached")
                and clean_success
                and done_step is not None
                and args.min_candidate_done_step <= done_step <= args.max_candidate_done_step
            )

            row = {
                "timestep": t,
                "horizon_requested": args.horizon,
                "horizon_used": continuation_result.get("horizon_used"),
                "prefix_restore_success": prefix_result.get("prefix_restore_success"),
                "prefix_fallback_used": prefix_result.get("prefix_fallback_used"),
                "prefix_steps_executed": prefix_result.get("prefix_steps_executed"),
                "prefix_done_reached": prefix_result.get("prefix_done_reached"),
                "prefix_final_reward": prefix_result.get("prefix_final_reward"),
                "prefix_final_done": prefix_result.get("prefix_final_done"),
                "prefix_eef_pos_l2_to_hdf5": prefix_eef_l2,
                "prefix_robot_state_l2_to_hdf5": prefix_robot_state_l2,
                "clean_success": clean_success,
                "done_reached": continuation_result.get("done_reached"),
                "done_step": done_step,
                "horizon_executed": continuation_result.get("horizon_executed"),
                "final_reward": continuation_result.get("final_reward"),
                "final_done": continuation_result.get("final_done"),
                "final_eef_pos": continuation_result.get("final_eef_pos"),
                "final_eef_quat": continuation_result.get("final_eef_quat"),
                "is_candidate": is_candidate,
                "prefix_video_path": prefix_video_path,
                "continuation_video_path": continuation_video_path,
                "errors": (prefix_result.get("errors", []) + continuation_result.get("errors", [])),
            }

            rows.append(row)
            summary["num_timesteps_scanned"] += 1
            if prefix_result.get("prefix_restore_success"):
                summary["num_prefix_restore_success"] += 1
            if clean_success:
                summary["num_clean_success"] += 1
            if is_candidate:
                summary["num_candidates"] += 1
                summary["candidate_timesteps"].append(t)

        summary["rows"] = rows

        summary_path = os.path.join(args.output_dir, "summary.json")
        csv_path = os.path.join(args.output_dir, "scan_rows.csv")

        _write_json(summary_path, summary)
        _write_csv(csv_path, rows)

        _print_summary(summary, summary_path, csv_path)

    except Exception as exc:
        summary["errors"].append(str(exc))
        summary["errors"].append(traceback.format_exc())

        summary_path = os.path.join(args.output_dir, "summary.json")
        csv_path = os.path.join(args.output_dir, "scan_rows.csv")

        _write_json(summary_path, summary)
        _write_csv(csv_path, rows)

        _print_summary(summary, summary_path, csv_path)


def _print_summary(summary: Dict[str, Any], summary_path: str, csv_path: str) -> None:
    print("=== Prefix Clean Viability Scan Summary ===")
    print(f"task: {summary.get('task_name')} (id {summary.get('task_id')})")
    print(f"demo_id: {summary.get('demo_id')}")
    print(f"episode_len: {summary.get('episode_len')}")
    print(f"num_timesteps_scanned: {summary.get('num_timesteps_scanned')}")
    print(f"num_clean_success: {summary.get('num_clean_success')}")
    print(f"num_candidates: {summary.get('num_candidates')}")
    print(f"candidate_timesteps: {summary.get('candidate_timesteps')}")
    print("\nRows:")
    print("timestep | prefix_done | clean_success | done_step | prefix_eef_l2 | is_candidate")
    for row in summary.get("rows", []):
        print(
            f"{row.get('timestep')} | {row.get('prefix_done_reached')} | {row.get('clean_success')} | "
            f"{row.get('done_step')} | {row.get('prefix_eef_pos_l2_to_hdf5')} | {row.get('is_candidate')}"
        )
    print(f"summary_path: {summary_path}")
    print(f"csv_path: {csv_path}")


if __name__ == "__main__":
    main()
