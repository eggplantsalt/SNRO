"""
Scan regenerated LIBERO HDF5 timesteps for clean rollout viability from restored mid-state.

Example:
    python scsro/scripts/scan_libero_clean_viability.py \
      --libero_task_suite libero_spatial \
      --libero_hdf5_dir /storage/v-xiangxizheng/zy_workspace/SNRO/datasets/libero_hdf5_no_noops/libero_spatial_no_noops \
      --task_id 0 \
      --demo_id 0 \
      --timesteps "30,35,40,45,50,55,60,65,70" \
      --horizon 80 \
      --output_dir ./scsro/debug_clean_viability_scan/spatial_t0_d0
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


def rollout_clean_from_state(
    env,
    state_t: np.ndarray,
    actions: np.ndarray,
    timestep: int,
    horizon: int,
    save_frames: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "restore_success": False,
        "fallback_used": False,
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

    obs, restore_success, fallback_used, errors = restore_env(env, state_t)
    result["restore_success"] = restore_success
    result["fallback_used"] = fallback_used
    result["errors"].extend(errors)

    if not restore_success or obs is None:
        return result

    horizon_used = min(horizon, len(actions) - timestep)
    result["horizon_used"] = horizon_used

    for k in range(horizon_used):
        action = actions[timestep + k]
        obs, reward, done, info = env.step(action.tolist())

        result["rewards"].append(float(reward) if reward is not None else None)
        result["dones"].append(bool(done))
        result["horizon_executed"] += 1

        if save_frames:
            result["frames"].append(get_libero_image(obs))

        if "robot0_eef_pos" in obs:
            result["final_eef_pos"] = np.array(obs["robot0_eef_pos"], dtype=np.float32).tolist()
        if "robot0_eef_quat" in obs:
            result["final_eef_quat"] = np.array(obs["robot0_eef_quat"], dtype=np.float32).tolist()

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


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "timestep",
        "restore_success",
        "fallback_used",
        "clean_success",
        "done_reached",
        "done_step",
        "horizon_executed",
        "final_reward",
        "final_done",
        "is_candidate",
        "video_path",
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
    parser.add_argument("--min_candidate_done_step", type=int, default=10)
    parser.add_argument("--max_candidate_done_step", type=int, default=60)
    parser.add_argument("--save_videos", type=str2bool, default=False)
    parser.add_argument("--output_dir", type=str, default="./scsro/debug_clean_viability_scan")

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
        "timesteps_requested": None,
        "num_timesteps_scanned": 0,
        "num_restore_success": 0,
        "num_clean_success": 0,
        "num_candidates": 0,
        "candidate_timesteps": [],
        "min_candidate_done_step": args.min_candidate_done_step,
        "max_candidate_done_step": args.max_candidate_done_step,
        "rows": [],
        "skipped_timesteps": [],
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
            state_t = states[t]
            result = rollout_clean_from_state(
                env,
                state_t,
                actions,
                t,
                args.horizon,
                save_frames=args.save_videos,
            )

            horizon_used = result.get("horizon_used")
            if horizon_used is None:
                horizon_used = min(args.horizon, len(actions) - t)

            clean_success = bool(result.get("done_reached") or result.get("final_reward") == 1.0)

            done_step = result.get("done_step")
            is_candidate = (
                clean_success
                and done_step is not None
                and args.min_candidate_done_step <= done_step <= args.max_candidate_done_step
            )

            video_path = None
            video_is_mp4 = None
            if args.save_videos and result.get("frames"):
                name = f"timestep_{t:04d}_clean_rollout"
                video_path, video_is_mp4 = save_rollout_mp4_or_frames(
                    args.output_dir,
                    name,
                    result["frames"],
                )

            row = {
                "timestep": t,
                "horizon_requested": args.horizon,
                "horizon_used": horizon_used,
                "restore_success": result.get("restore_success"),
                "fallback_used": result.get("fallback_used"),
                "clean_success": clean_success,
                "done_reached": result.get("done_reached"),
                "done_step": done_step,
                "horizon_executed": result.get("horizon_executed"),
                "final_reward": result.get("final_reward"),
                "final_done": result.get("final_done"),
                "final_eef_pos": result.get("final_eef_pos"),
                "final_eef_quat": result.get("final_eef_quat"),
                "is_candidate": is_candidate,
                "video_path": video_path,
                "video_is_mp4": video_is_mp4,
                "errors": result.get("errors", []),
            }

            summary["rows"].append(row)
            summary["num_timesteps_scanned"] += 1
            if result.get("restore_success"):
                summary["num_restore_success"] += 1
            if clean_success:
                summary["num_clean_success"] += 1
            if is_candidate:
                summary["num_candidates"] += 1
                summary["candidate_timesteps"].append(t)

        summary_path = os.path.join(args.output_dir, "summary.json")
        csv_path = os.path.join(args.output_dir, "scan_rows.csv")

        _write_json(summary_path, summary)
        write_csv(csv_path, summary["rows"])

        _print_summary(summary, summary_path, csv_path)

    except Exception as exc:
        summary["errors"].append(str(exc))
        summary["errors"].append(traceback.format_exc())

        summary_path = os.path.join(args.output_dir, "summary.json")
        csv_path = os.path.join(args.output_dir, "scan_rows.csv")

        _write_json(summary_path, summary)
        write_csv(csv_path, summary.get("rows", []))

        _print_summary(summary, summary_path, csv_path)


def _print_summary(summary: Dict[str, Any], summary_path: str, csv_path: str) -> None:
    print("=== Clean Viability Scan Summary ===")
    print(f"task: {summary.get('task_name')} (id {summary.get('task_id')})")
    print(f"demo_id: {summary.get('demo_id')}")
    print(f"episode_len: {summary.get('episode_len')}")
    print(f"num_timesteps_scanned: {summary.get('num_timesteps_scanned')}")
    print(f"num_clean_success: {summary.get('num_clean_success')}")
    print(f"num_candidates: {summary.get('num_candidates')}")
    print(f"candidate_timesteps: {summary.get('candidate_timesteps')}")
    print("\nRows:")
    print("timestep | clean_success | done_step | horizon_executed | is_candidate")
    for row in summary.get("rows", []):
        print(
            f"{row.get('timestep')} | {row.get('clean_success')} | "
            f"{row.get('done_step')} | {row.get('horizon_executed')} | {row.get('is_candidate')}"
        )
    print(f"summary_path: {summary_path}")
    print(f"csv_path: {csv_path}")


if __name__ == "__main__":
    main()
"""
Scan clean rollout viability from multiple timesteps in regenerated LIBERO HDF5.

Example:
    python scsro/scripts/scan_libero_clean_viability.py \
      --libero_task_suite libero_spatial \
      --libero_hdf5_dir /storage/v-xiangxizheng/zy_workspace/SNRO/datasets/libero_hdf5_no_noops/libero_spatial_no_noops \
      --task_id 0 \
      --demo_id 0 \
      --timesteps "30,35,40,45,50,55,60,65,70" \
      --horizon 80 \
      --output_dir ./scsro/debug_clean_viability_scan/spatial_t0_d0
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


def parse_timesteps(timesteps_str: str) -> List[int]:
    parts = [p.strip() for p in timesteps_str.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


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


def rollout_clean_from_state(
    env,
    state_t: np.ndarray,
    actions: np.ndarray,
    timestep: int,
    horizon: int,
    save_frames: bool = False,
) -> Dict[str, Any]:
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
        "frames": [],
        "errors": [],
    }

    obs, restore_success, fallback_used, errors = restore_env(env, state_t)
    result["restore_success"] = restore_success
    result["fallback_used"] = fallback_used
    result["errors"].extend(errors)

    if not restore_success or obs is None:
        return result

    horizon_used = min(horizon, len(actions) - timestep)
    for k in range(horizon_used):
        action = actions[timestep + k]
        obs, reward, done, info = env.step(action.tolist())

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


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "timestep",
        "restore_success",
        "fallback_used",
        "clean_success",
        "done_reached",
        "done_step",
        "horizon_executed",
        "final_reward",
        "final_done",
        "is_candidate",
        "video_path",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
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
    parser.add_argument("--timesteps", type=str, default=None)
    parser.add_argument("--start_timestep", type=int, default=0)
    parser.add_argument("--end_timestep", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--min_candidate_done_step", type=int, default=10)
    parser.add_argument("--max_candidate_done_step", type=int, default=60)
    parser.add_argument("--save_videos", type=str2bool, default=False)
    parser.add_argument("--output_dir", type=str, default="./scsro/debug_clean_viability_scan")

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
        "timesteps_requested": [],
        "num_timesteps_scanned": 0,
        "num_restore_success": 0,
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
        summary["episode_len"] = len(actions)

        if args.timesteps:
            requested_timesteps = parse_timesteps(args.timesteps)
        else:
            start = args.start_timestep
            end = args.end_timestep if args.end_timestep >= 0 else len(actions)
            requested_timesteps = list(range(start, end, args.stride))

        summary["timesteps_requested"] = requested_timesteps

        for timestep in requested_timesteps:
            if timestep < 0 or timestep >= len(actions):
                summary["skipped_timesteps"].append(timestep)
                continue

            horizon_used = min(args.horizon, len(actions) - timestep)

            result = rollout_clean_from_state(
                env,
                states[timestep],
                actions,
                timestep,
                horizon_used,
                save_frames=args.save_videos,
            )

            clean_success = bool(result.get("done_reached") or result.get("final_reward") == 1.0)
            done_step = result.get("done_step")

            is_candidate = bool(
                clean_success
                and done_step is not None
                and args.min_candidate_done_step <= done_step <= args.max_candidate_done_step
            )

            video_path = None
            video_is_mp4 = None
            if args.save_videos:
                video_path, video_is_mp4 = save_rollout_mp4_or_frames(
                    args.output_dir,
                    f"timestep_{timestep:04d}_clean_rollout",
                    result.get("frames", []),
                )

            row = {
                "timestep": timestep,
                "horizon_requested": args.horizon,
                "horizon_used": horizon_used,
                "restore_success": result.get("restore_success"),
                "fallback_used": result.get("fallback_used"),
                "clean_success": clean_success,
                "done_reached": result.get("done_reached"),
                "done_step": done_step,
                "horizon_executed": result.get("horizon_executed"),
                "final_reward": result.get("final_reward"),
                "final_done": result.get("final_done"),
                "final_eef_pos": result.get("final_eef_pos"),
                "final_eef_quat": result.get("final_eef_quat"),
                "is_candidate": is_candidate,
                "video_path": video_path,
                "video_is_mp4": video_is_mp4,
                "errors": result.get("errors", []),
            }

            rows.append(row)
            summary["num_timesteps_scanned"] += 1
            if result.get("restore_success"):
                summary["num_restore_success"] += 1
            if clean_success:
                summary["num_clean_success"] += 1
            if is_candidate:
                summary["num_candidates"] += 1
                summary["candidate_timesteps"].append(timestep)

        summary["rows"] = rows

        summary_path = os.path.join(args.output_dir, "summary.json")
        _write_json(summary_path, summary)

        csv_path = os.path.join(args.output_dir, "scan_rows.csv")
        _write_csv(csv_path, rows)

        _print_summary(summary, summary_path, csv_path)

    except Exception as exc:
        summary["errors"].append(str(exc))
        summary["errors"].append(traceback.format_exc())
        summary_path = os.path.join(args.output_dir, "summary.json")
        _write_json(summary_path, summary)
        _print_summary(summary, summary_path, None)


def _print_summary(summary: Dict[str, Any], summary_path: str, csv_path: Optional[str]) -> None:
    print("=== Clean Viability Scan Summary ===")
    print(f"task: {summary.get('task_name')} (id {summary.get('task_id')})")
    print(f"demo_id: {summary.get('demo_id')}")
    print(f"episode_len: {summary.get('episode_len')}")
    print(f"num_timesteps_scanned: {summary.get('num_timesteps_scanned')}")
    print(f"num_clean_success: {summary.get('num_clean_success')}")
    print(f"num_candidates: {summary.get('num_candidates')}")
    print(f"candidate_timesteps: {summary.get('candidate_timesteps')}")
    print("\nRows:")
    print("timestep | clean_success | done_step | horizon_executed | is_candidate")
    for row in summary.get("rows", []):
        print(
            f"{row.get('timestep')} | {row.get('clean_success')} | {row.get('done_step')} | "
            f"{row.get('horizon_executed')} | {row.get('is_candidate')}"
        )
    print(f"summary_path: {summary_path}")
    if csv_path is not None:
        print(f"csv_path: {csv_path}")


if __name__ == "__main__":
    main()
