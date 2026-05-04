"""
Check whether a mid-timestep LIBERO simulator state can be restored from regenerated HDF5
and rolled out from that point using the remaining demo actions.

Example:
    python scsro/scripts/check_libero_mid_state_restore.py \
      --libero_task_suite libero_spatial \
      --libero_hdf5_dir ./LIBERO/libero/datasets/libero_spatial_no_noops \
      --task_id 0 \
      --demo_id 0 \
      --timestep 10 \
      --horizon 20 \
      --output_dir ./scsro/debug_mid_state_restore
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import h5py
import imageio
import numpy as np
from libero.libero import benchmark

# Ensure repo root is on sys.path so experiments.* imports resolve reliably.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_image(path: str, img: np.ndarray) -> None:
    imageio.imwrite(path, img)


def _save_rollout_mp4_or_frames(output_dir: str, frames: list[np.ndarray], fps: int = 30) -> tuple[str, bool]:
    if not frames:
        return "", False

    mp4_path = os.path.join(output_dir, "mid_state_rollout.mp4")
    writer = None
    try:
        writer = imageio.get_writer(mp4_path, fps=fps)
        for frame in frames:
            writer.append_data(frame)
        return mp4_path, True
    except Exception:
        frame_dir = os.path.join(output_dir, "rollout_frames")
        _ensure_dir(frame_dir)
        for i, frame in enumerate(frames):
            frame_path = os.path.join(frame_dir, f"frame_{i:03d}.png")
            imageio.imwrite(frame_path, frame)
        return frame_dir, False
    finally:
        if writer is not None:
            writer.close()


def _get_optional_demo_array(demo_group: h5py.Group, path_parts: list[str]) -> np.ndarray | None:
    node = demo_group
    for part in path_parts:
        if part not in node:
            return None
        node = node[part]
    return node[()]


def _l2_error(a: np.ndarray, b: np.ndarray) -> float | None:
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return None
    return float(np.linalg.norm(a - b))


def _image_mse(a: np.ndarray, b: np.ndarray) -> float | None:
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return None
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    return float(np.mean((a_f - b_f) ** 2))


def _image_l1_mean(a: np.ndarray, b: np.ndarray) -> float | None:
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return None
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    return float(np.mean(np.abs(a_f - b_f)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        help="LIBERO task suite.",
    )
    parser.add_argument(
        "--libero_hdf5_dir",
        type=str,
        required=True,
        help="Path to regenerated LIBERO HDF5 dataset dir.",
    )
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--demo_id", type=int, default=0)
    parser.add_argument("--timestep", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--num_steps_wait",
        type=int,
        default=0,
        help="Number of dummy steps after restoring state.",
    )
    parser.add_argument("--env_img_res", type=int, default=256)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./scsro/debug_mid_state_restore",
    )
    args = parser.parse_args()

    _ensure_dir(args.output_dir)

    summary = {
        "restore_success": False,
        "fallback_used": False,
        "libero_task_suite": args.libero_task_suite,
        "task_id": args.task_id,
        "task_name": None,
        "task_description": None,
        "demo_id": args.demo_id,
        "timestep": args.timestep,
        "horizon_requested": args.horizon,
        "horizon_executed": 0,
        "hdf5_path": None,
        "state_shape": None,
        "action_shape": None,
        "comparison_done_before_wait": True,
        "num_steps_wait": args.num_steps_wait,
        "eef_pos_env": None,
        "eef_pos_hdf5": None,
        "eef_pos_l2_error": None,
        "ee_ori_l2_error": None,
        "agentview_raw_mse": None,
        "agentview_raw_l1_mean": None,
        "wrist_raw_mse": None,
        "wrist_raw_l1_mean": None,
        "robot_state_env": None,
        "robot_state_hdf5": None,
        "robot_state_l2_error": None,
        "rollout_visualization_path": None,
        "rollout_visualization_is_mp4": None,
        "done_reached": False,
        "final_reward": None,
        "final_done": None,
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

            demo_ee_pos = _get_optional_demo_array(demo, ["obs", "ee_pos"])
            demo_ee_ori = _get_optional_demo_array(demo, ["obs", "ee_ori"])
            demo_agentview_raw = _get_optional_demo_array(demo, ["obs", "agentview_rgb"])
            demo_wrist_raw = _get_optional_demo_array(demo, ["obs", "eye_in_hand_rgb"])
            demo_robot_state = _get_optional_demo_array(demo, ["robot_states"])

            if args.timestep < 0 or args.timestep >= len(actions):
                raise ValueError(f"Invalid timestep {args.timestep} for actions length {len(actions)}")

            horizon = min(args.horizon, len(actions) - args.timestep)
            if horizon < 0:
                raise ValueError("Horizon is negative after bounds checking.")

            env.reset()
            state_t = states[args.timestep]

            obs = None
            try:
                obs = env.set_init_state(state_t)
            except Exception as exc:
                summary["errors"].append(f"set_init_state failed: {exc}")
                summary["fallback_used"] = True
                try:
                    env.sim.set_state_from_flattened(state_t)
                    env.sim.forward()
                    obs = env.get_observation()
                except Exception as fallback_exc:
                    summary["errors"].append(f"fallback failed: {fallback_exc}")
                    summary["restore_success"] = False
                    _safe_write_json(os.path.join(args.output_dir, "summary.json"), summary)
                    _print_summary(summary)
                    return

            summary["restore_success"] = True

            if obs is not None:
                agentview = get_libero_image(obs)
                wrist = get_libero_wrist_image(obs)
                _save_image(
                    os.path.join(args.output_dir, "restored_agentview_policy_rotated.png"),
                    agentview,
                )
                _save_image(
                    os.path.join(args.output_dir, "restored_wrist_policy_rotated.png"),
                    wrist,
                )

                raw_agentview_env = obs.get("agentview_image")
                raw_wrist_env = obs.get("robot0_eye_in_hand_image")
                if raw_agentview_env is not None:
                    _save_image(
                        os.path.join(args.output_dir, "env_agentview_raw.png"),
                        raw_agentview_env,
                    )
                if raw_wrist_env is not None:
                    _save_image(
                        os.path.join(args.output_dir, "env_wrist_raw.png"),
                        raw_wrist_env,
                    )

                if demo_agentview_raw is not None and args.timestep < len(demo_agentview_raw):
                    hdf5_agentview = demo_agentview_raw[args.timestep]
                    _save_image(
                        os.path.join(args.output_dir, "hdf5_agentview_raw.png"),
                        hdf5_agentview,
                    )
                    summary["agentview_raw_mse"] = _image_mse(raw_agentview_env, hdf5_agentview)
                    summary["agentview_raw_l1_mean"] = _image_l1_mean(raw_agentview_env, hdf5_agentview)

                if demo_wrist_raw is not None and args.timestep < len(demo_wrist_raw):
                    hdf5_wrist = demo_wrist_raw[args.timestep]
                    _save_image(
                        os.path.join(args.output_dir, "hdf5_wrist_raw.png"),
                        hdf5_wrist,
                    )
                    summary["wrist_raw_mse"] = _image_mse(raw_wrist_env, hdf5_wrist)
                    summary["wrist_raw_l1_mean"] = _image_l1_mean(raw_wrist_env, hdf5_wrist)

                if demo_ee_pos is not None and args.timestep < len(demo_ee_pos):
                    env_pos = np.array(obs.get("robot0_eef_pos"), dtype=np.float32)
                    hdf5_pos = np.array(demo_ee_pos[args.timestep], dtype=np.float32)
                    summary["eef_pos_env"] = env_pos.tolist()
                    summary["eef_pos_hdf5"] = hdf5_pos.tolist()
                    summary["eef_pos_l2_error"] = _l2_error(env_pos, hdf5_pos)

                if demo_ee_ori is not None and args.timestep < len(demo_ee_ori):
                    env_quat = obs.get("robot0_eef_quat")
                    if env_quat is not None:
                        env_axis = quat2axisangle(np.array(env_quat, dtype=np.float32))
                        hdf5_axis = np.array(demo_ee_ori[args.timestep], dtype=np.float32)
                        summary["ee_ori_l2_error"] = _l2_error(env_axis, hdf5_axis)

                if demo_robot_state is not None and args.timestep < len(demo_robot_state):
                    gripper = obs.get("robot0_gripper_qpos")
                    eef_pos = obs.get("robot0_eef_pos")
                    eef_quat = obs.get("robot0_eef_quat")
                    if gripper is not None and eef_pos is not None and eef_quat is not None:
                        env_robot_state = np.concatenate([gripper, eef_pos, eef_quat])
                        hdf5_robot_state = np.array(demo_robot_state[args.timestep], dtype=np.float32)
                        summary["robot_state_env"] = env_robot_state.tolist()
                        summary["robot_state_hdf5"] = hdf5_robot_state.tolist()
                        summary["robot_state_l2_error"] = _l2_error(env_robot_state, hdf5_robot_state)

            if args.num_steps_wait > 0:
                for _ in range(args.num_steps_wait):
                    obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))

            rollout_images = []
            final_reward = None
            final_done = None
            done_reached = False

            for k in range(horizon):
                action = actions[args.timestep + k]
                obs, reward, done, info = env.step(action.tolist())
                rollout_images.append(get_libero_image(obs))
                final_reward = float(reward) if reward is not None else None
                final_done = bool(done)
                summary["horizon_executed"] += 1
                if done:
                    done_reached = True
                    break

            summary["done_reached"] = done_reached
            summary["final_reward"] = final_reward
            summary["final_done"] = final_done

            rollout_path, is_mp4 = _save_rollout_mp4_or_frames(args.output_dir, rollout_images)
            summary["rollout_visualization_path"] = rollout_path or None
            summary["rollout_visualization_is_mp4"] = is_mp4 if rollout_path else None

    except Exception as exc:
        summary["errors"].append(str(exc))
        summary["errors"].append(traceback.format_exc())

    _safe_write_json(os.path.join(args.output_dir, "summary.json"), summary)
    _print_summary(summary)


def _print_summary(summary: dict) -> None:
    lines = [
        "=== Mid-State Restore Summary ===",
        f"restore_success: {summary['restore_success']}",
        f"fallback_used: {summary['fallback_used']}",
        f"task: {summary.get('task_name')} (id {summary.get('task_id')})",
        f"demo_id: {summary.get('demo_id')}",
        f"timestep: {summary.get('timestep')}",
        f"horizon_executed: {summary.get('horizon_executed')}",
        f"done_reached: {summary.get('done_reached')}",
        f"final_reward: {summary.get('final_reward')}",
    ]
    if summary.get("eef_pos_l2_error") is not None:
        lines.append(f"eef_pos_l2_error: {summary.get('eef_pos_l2_error'):.6f}")
    if summary.get("ee_ori_l2_error") is not None:
        lines.append(f"ee_ori_l2_error: {summary.get('ee_ori_l2_error'):.6f}")
    if summary.get("errors"):
        lines.append(f"errors: {len(summary['errors'])}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
