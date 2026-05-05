"""
Search action deltas using a prefix replay oracle on regenerated LIBERO HDF5.

Example:
    python scsro/scripts/search_libero_prefix_delta.py \
      --libero_task_suite libero_spatial \
      --libero_hdf5_dir /storage/v-xiangxizheng/zy_workspace/SNRO/datasets/libero_hdf5_no_noops/libero_spatial_no_noops \
      --task_id 0 \
      --demo_id 0 \
      --timesteps "45,50,55,60" \
      --horizon 80 \
      --num_steps_wait 10 \
      --perturb_steps 8 \
      --delta_dims "0,1,2" \
      --delta_magnitudes "0.02,0.05,0.1" \
      --output_dir ./scsro/debug_prefix_delta_search/spatial_t0_d0
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


def parse_int_list(value: str) -> List[int]:
    if value.strip() == "":
        return []
    parts = [p.strip() for p in value.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def parse_float_list(value: str) -> List[float]:
    if value.strip() == "":
        return []
    parts = [p.strip() for p in value.split(",") if p.strip() != ""]
    return [float(p) for p in parts]


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


def replay_prefix_to_timestep(
    env,
    initial_state: np.ndarray,
    actions: np.ndarray,
    timestep: int,
    num_steps_wait: int,
    save_frames: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "restore_success": False,
        "fallback_used": False,
        "prefix_done_reached": False,
        "errors": [],
        "frames": [] if save_frames else None,
    }

    obs, restore_success, fallback_used, errors = restore_initial_env(env, initial_state)
    result["restore_success"] = restore_success
    result["fallback_used"] = fallback_used
    result["errors"].extend(errors)

    if not restore_success or obs is None:
        return result

    for _ in range(num_steps_wait):
        obs, reward, done, info = env.step(get_libero_dummy_action("llava"))
        if save_frames:
            result["frames"].append(get_libero_image(obs))
        if done:
            result["prefix_done_reached"] = True
            return result

    for k in range(timestep):
        obs, reward, done, info = env.step(actions[k].tolist())
        if save_frames:
            result["frames"].append(get_libero_image(obs))
        if done:
            result["prefix_done_reached"] = True
            break

    return result


def _get_sim_state_flat(env) -> Optional[np.ndarray]:
    try:
        return np.array(env.sim.get_state().flatten(), dtype=np.float32)
    except Exception:
        return None


def rollout_continuation(
    env,
    actions: np.ndarray,
    timestep: int,
    horizon: int,
    perturb_delta: Optional[np.ndarray] = None,
    perturb_steps: int = 0,
    clip_action: bool = False,
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
        "eef_pos_traj": [],
        "sim_state_traj": [],
        "action_delta_l2_traj": [],
        "frames": [] if save_frames else None,
    }

    horizon_used = min(horizon, len(actions) - timestep)
    result["horizon_used"] = horizon_used

    for j in range(horizon_used):
        idx = timestep + j
        base_action = actions[idx].copy()

        if perturb_delta is not None and j < perturb_steps:
            action = base_action + perturb_delta
            if clip_action:
                action[:6] = np.clip(action[:6], -1.0, 1.0)
        else:
            action = base_action

        effective_delta = action - base_action

        obs, reward, done, info = env.step(action.tolist())

        if save_frames:
            result["frames"].append(get_libero_image(obs))

        result["rewards"].append(float(reward) if reward is not None else None)
        result["dones"].append(bool(done))
        result["action_delta_l2_traj"].append(float(np.linalg.norm(effective_delta)))
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

        sim_state = _get_sim_state_flat(env)
        result["sim_state_traj"].append(sim_state.tolist() if sim_state is not None else None)

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


def _compute_l2_traj(a: List[Optional[List[float]]], b: List[Optional[List[float]]]) -> List[Optional[float]]:
    n = min(len(a), len(b))
    out: List[Optional[float]] = []
    for i in range(n):
        va = a[i]
        vb = b[i]
        if va is None or vb is None:
            out.append(None)
            continue
        a_np = np.array(va, dtype=np.float32)
        b_np = np.array(vb, dtype=np.float32)
        if a_np.shape != b_np.shape:
            out.append(None)
            continue
        out.append(float(np.linalg.norm(a_np - b_np)))
    return out


def _summarize_l2(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None, None
    return float(valid[-1]), float(np.max(valid))


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "timestep",
        "delta_dim",
        "delta_magnitude",
        "delta_sign",
        "clean_success",
        "clean_done_step",
        "perturbed_success",
        "perturbed_done_step",
        "success_changed",
        "done_step_delay",
        "eef_pos_l2_final",
        "eef_pos_l2_max",
        "sim_state_l2_final",
        "sim_state_l2_max",
        "score",
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
    parser.add_argument("--timesteps", type=str, default="")
    parser.add_argument("--start_timestep", type=int, default=30)
    parser.add_argument("--end_timestep", type=int, default=70)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--perturb_steps", type=int, default=8)
    parser.add_argument("--delta_dims", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--delta_magnitudes", type=str, default="0.02,0.05,0.1")
    parser.add_argument("--include_negative", type=str2bool, default=True)
    parser.add_argument("--zero_gripper_delta", type=str2bool, default=True)
    parser.add_argument("--clip_action", type=str2bool, default=False)
    parser.add_argument("--env_img_res", type=int, default=256)
    parser.add_argument("--save_videos_top_k", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./scsro/debug_prefix_delta_search")

    args = parser.parse_args()

    _ensure_dir(args.output_dir)

    summary: Dict[str, Any] = {
        "libero_task_suite": args.libero_task_suite,
        "task_id": args.task_id,
        "task_name": None,
        "task_description": None,
        "demo_id": args.demo_id,
        "hdf5_path": None,
        "episode_len": None,
        "timesteps": [],
        "horizon": args.horizon,
        "num_steps_wait": args.num_steps_wait,
        "perturb_steps": args.perturb_steps,
        "delta_dims": [],
        "delta_magnitudes": [],
        "include_negative": args.include_negative,
        "num_delta_candidates": 0,
        "num_rows": 0,
        "best_by_timestep": {},
        "global_best": None,
        "rows": [],
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
            _ = demo.get("obs", {}).get("ee_pos", None)
            _ = demo.get("robot_states", None)

        summary["episode_len"] = len(actions)

        if args.timesteps.strip() != "":
            timesteps = parse_int_list(args.timesteps)
        else:
            end = args.end_timestep if args.end_timestep >= 0 else len(actions)
            timesteps = list(range(args.start_timestep, end, args.stride))

        timesteps = [t for t in timesteps if 0 <= t < len(actions)]
        summary["timesteps"] = timesteps

        delta_dims = parse_int_list(args.delta_dims)
        delta_magnitudes = parse_float_list(args.delta_magnitudes)
        summary["delta_dims"] = delta_dims
        summary["delta_magnitudes"] = delta_magnitudes

        delta_candidates: List[Tuple[int, float, int, np.ndarray]] = []
        for dim in delta_dims:
            for mag in delta_magnitudes:
                signs = [1]
                if args.include_negative:
                    signs = [1, -1]
                for sign in signs:
                    delta = np.zeros(7, dtype=np.float32)
                    delta[dim] = float(sign) * float(mag)
                    if args.zero_gripper_delta:
                        delta[-1] = 0.0
                    delta_candidates.append((dim, mag, sign, delta))

        summary["num_delta_candidates"] = len(delta_candidates)

        for t in timesteps:
            clean_prefix = replay_prefix_to_timestep(
                env,
                states[0],
                actions,
                t,
                args.num_steps_wait,
                save_frames=args.save_videos_top_k > 0,
            )

            clean_result = None
            clean_video_path = None
            clean_video_is_mp4 = None

            if clean_prefix.get("restore_success") and not clean_prefix.get("prefix_done_reached"):
                clean_result = rollout_continuation(
                    env,
                    actions,
                    t,
                    args.horizon,
                    perturb_delta=None,
                    perturb_steps=0,
                    clip_action=args.clip_action,
                    save_frames=args.save_videos_top_k > 0,
                )

                if args.save_videos_top_k > 0 and clean_result.get("frames"):
                    clean_video_path, clean_video_is_mp4 = save_rollout_mp4_or_frames(
                        args.output_dir,
                        f"timestep_{t:04d}_clean",
                        clean_result["frames"],
                    )

            if clean_result is None:
                clean_result = {
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
                    "eef_pos_traj": [],
                    "sim_state_traj": [],
                }

            clean_success = bool(clean_result.get("done_reached") or clean_result.get("final_reward") == 1.0)
            clean_done_step = clean_result.get("done_step")

            if not clean_success:
                continue

            best_row = None
            best_score = None

            for dim, mag, sign, delta in delta_candidates:
                prefix = replay_prefix_to_timestep(
                    env,
                    states[0],
                    actions,
                    t,
                    args.num_steps_wait,
                    save_frames=args.save_videos_top_k > 0,
                )

                if not prefix.get("restore_success") or prefix.get("prefix_done_reached"):
                    continue

                perturbed = rollout_continuation(
                    env,
                    actions,
                    t,
                    args.horizon,
                    perturb_delta=delta,
                    perturb_steps=args.perturb_steps,
                    clip_action=args.clip_action,
                    save_frames=args.save_videos_top_k > 0,
                )

                perturbed_success = bool(perturbed.get("done_reached") or perturbed.get("final_reward") == 1.0)
                perturbed_done_step = perturbed.get("done_step")
                success_changed = clean_success != perturbed_success

                if clean_success and perturbed_success:
                    done_step_delay = (perturbed_done_step or 0) - (clean_done_step or 0)
                elif clean_success and not perturbed_success:
                    done_step_delay = perturbed.get("horizon_used", 0) - (clean_done_step or 0)
                else:
                    done_step_delay = 0

                eef_pos_l2_traj = _compute_l2_traj(
                    clean_result.get("eef_pos_traj", []),
                    perturbed.get("eef_pos_traj", []),
                )
                sim_state_l2_traj = _compute_l2_traj(
                    clean_result.get("sim_state_traj", []),
                    perturbed.get("sim_state_traj", []),
                )

                eef_pos_l2_final, eef_pos_l2_max = _summarize_l2(eef_pos_l2_traj)
                sim_state_l2_final, sim_state_l2_max = _summarize_l2(sim_state_l2_traj)

                failure_flag = 1 if (clean_success and not perturbed_success) else 0

                score_terms = {
                    "failure_flag": failure_flag,
                    "done_step_delay": done_step_delay,
                    "eef_pos_l2_max": eef_pos_l2_max or 0.0,
                    "sim_state_l2_max": sim_state_l2_max or 0.0,
                }

                score = (
                    100.0 * score_terms["failure_flag"]
                    + 1.0 * max(score_terms["done_step_delay"], 0)
                    + 10.0 * score_terms["eef_pos_l2_max"]
                    + 0.1 * score_terms["sim_state_l2_max"]
                )

                row = {
                    "timestep": t,
                    "delta": delta.tolist(),
                    "delta_dim": dim,
                    "delta_magnitude": mag,
                    "delta_sign": sign,
                    "perturb_steps": args.perturb_steps,
                    "clean_success": clean_success,
                    "clean_done_step": clean_done_step,
                    "perturbed_success": perturbed_success,
                    "perturbed_done_step": perturbed_done_step,
                    "success_changed": success_changed,
                    "done_step_delay": done_step_delay,
                    "eef_pos_l2_final": eef_pos_l2_final,
                    "eef_pos_l2_max": eef_pos_l2_max,
                    "sim_state_l2_final": sim_state_l2_final,
                    "sim_state_l2_max": sim_state_l2_max,
                    "score": score,
                    "score_terms": score_terms,
                }

                rows.append(row)

                if best_score is None or score > best_score:
                    best_score = score
                    best_row = row

            if best_row is not None:
                summary["best_by_timestep"][str(t)] = {
                    "delta": best_row["delta"],
                    "score": best_row["score"],
                    "success_changed": best_row["success_changed"],
                    "clean_success": best_row["clean_success"],
                    "perturbed_success": best_row["perturbed_success"],
                    "clean_done_step": best_row["clean_done_step"],
                    "perturbed_done_step": best_row["perturbed_done_step"],
                }

            if args.save_videos_top_k > 0 and best_row is not None:
                top_k_rows = sorted(
                    [r for r in rows if r["timestep"] == t],
                    key=lambda r: r["score"],
                    reverse=True,
                )[: args.save_videos_top_k]

                for rank, best in enumerate(top_k_rows, start=1):
                    prefix = replay_prefix_to_timestep(
                        env,
                        states[0],
                        actions,
                        t,
                        args.num_steps_wait,
                        save_frames=True,
                    )
                    if not prefix.get("restore_success") or prefix.get("prefix_done_reached"):
                        continue

                    pert = rollout_continuation(
                        env,
                        actions,
                        t,
                        args.horizon,
                        perturb_delta=np.array(best["delta"], dtype=np.float32),
                        perturb_steps=args.perturb_steps,
                        clip_action=args.clip_action,
                        save_frames=True,
                    )

                    video_name = f"timestep_{t:04d}_rank{rank}_delta"
                    save_rollout_mp4_or_frames(args.output_dir, video_name, pert.get("frames", []))

                    if clean_video_path is None:
                        clean_prefix = replay_prefix_to_timestep(
                            env,
                            states[0],
                            actions,
                            t,
                            args.num_steps_wait,
                            save_frames=True,
                        )
                        if clean_prefix.get("restore_success") and not clean_prefix.get("prefix_done_reached"):
                            clean = rollout_continuation(
                                env,
                                actions,
                                t,
                                args.horizon,
                                perturb_delta=None,
                                perturb_steps=0,
                                clip_action=args.clip_action,
                                save_frames=True,
                            )
                            clean_video_path, clean_video_is_mp4 = save_rollout_mp4_or_frames(
                                args.output_dir,
                                f"timestep_{t:04d}_clean",
                                clean.get("frames", []),
                            )

        summary["rows"] = rows
        summary["num_rows"] = len(rows)

        if rows:
            global_best = max(rows, key=lambda r: r["score"])
            summary["global_best"] = {
                "timestep": global_best["timestep"],
                "delta": global_best["delta"],
                "score": global_best["score"],
                "success_changed": global_best["success_changed"],
                "clean_success": global_best["clean_success"],
                "perturbed_success": global_best["perturbed_success"],
                "clean_done_step": global_best["clean_done_step"],
                "perturbed_done_step": global_best["perturbed_done_step"],
            }

        summary_path = os.path.join(args.output_dir, "summary.json")
        csv_path = os.path.join(args.output_dir, "delta_rows.csv")

        _write_json(summary_path, summary)
        _write_csv(csv_path, rows)

        _print_summary(summary, summary_path, csv_path)

    except Exception as exc:
        summary["errors"].append(str(exc))
        summary["errors"].append(traceback.format_exc())

        summary_path = os.path.join(args.output_dir, "summary.json")
        csv_path = os.path.join(args.output_dir, "delta_rows.csv")

        _write_json(summary_path, summary)
        _write_csv(csv_path, rows)

        _print_summary(summary, summary_path, csv_path)


def _print_summary(summary: Dict[str, Any], summary_path: str, csv_path: str) -> None:
    print("=== Prefix Delta Search Summary ===")
    print(f"task: {summary.get('task_name')} (id {summary.get('task_id')})")
    print(f"demo_id: {summary.get('demo_id')}")
    print(f"timesteps: {summary.get('timesteps')}")
    print(f"num_delta_candidates: {summary.get('num_delta_candidates')}")
    print(f"num_rows: {summary.get('num_rows')}")

    global_best = summary.get("global_best") or {}
    print("global_best:")
    print(f"  timestep: {global_best.get('timestep')}")
    print(f"  delta: {global_best.get('delta')}")
    print(f"  score: {global_best.get('score')}")
    print(f"  success_changed: {global_best.get('success_changed')}")
    print(f"  clean_success: {global_best.get('clean_success')}")
    print(f"  perturbed_success: {global_best.get('perturbed_success')}")
    print(f"  clean_done_step: {global_best.get('clean_done_step')}")
    print(f"  perturbed_done_step: {global_best.get('perturbed_done_step')}")

    print("best_by_timestep:")
    for t_key, best in summary.get("best_by_timestep", {}).items():
        print(f"  t={t_key}: score={best.get('score')}, delta={best.get('delta')}, success_changed={best.get('success_changed')}")

    print(f"summary_path: {summary_path}")
    print(f"csv_path: {csv_path}")


if __name__ == "__main__":
    main()
