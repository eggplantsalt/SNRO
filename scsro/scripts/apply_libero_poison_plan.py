"""
Apply a poison_plan.json to generate a poisoned LIBERO HDF5 action-label dataset.

This script copies a clean regenerated LIBERO HDF5 directory to a new output
directory, then modifies selected action labels according to a poison plan.
It never modifies the source clean HDF5 directory.

Example:
    python scsro/scripts/apply_libero_poison_plan.py \
      --plan_path scsro/poison_plans/libero_spatial_task0_demo0_v0.json \
      --candidate_ids t45_dx_pos020_k8 \
      --output_hdf5_dir /storage/v-xiangxizheng/zy_workspace/SNRO/datasets/libero_hdf5_poisoned/libero_spatial_task0_demo0_v0_t45 \
      --overwrite False

Dry run example:
    python scsro/scripts/apply_libero_poison_plan.py \
      --plan_path scsro/poison_plans/libero_spatial_task0_demo0_v0.json \
      --candidate_ids t45_dx_pos020_k8 \
      --output_hdf5_dir ./scsro/debug_poison_apply_dryrun/libero_spatial_task0_demo0_v0_t45 \
      --dry_run True \
      --overwrite True
"""

import argparse
import csv
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

# Ensure repo root is on sys.path so relative paths resolve reliably.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def str2bool(v):
    """Robust argparse boolean parser.

    Do not use type=bool with argparse, because bool("False") is True.
    """
    if isinstance(v, bool):
        return v

    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_optional_bool(v: Optional[str]) -> Optional[bool]:
    if v is None:
        return None
    return str2bool(v)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def json_list(x: Any) -> str:
    return json.dumps(x)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_id",
        "demo_id",
        "action_index",
        "delta",
        "old_action",
        "new_action",
        "effective_delta",
        "l2_effective_delta",
        "clip_action",
        "zero_gripper_delta",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def parse_candidate_ids(value: str) -> List[str]:
    if value is None or value.strip() == "":
        return []
    return [v.strip() for v in value.split(",") if v.strip() != ""]


def load_plan(plan_path: str) -> Dict[str, Any]:
    with open(plan_path, "r", encoding="utf-8") as f:
        return json.load(f)


def effective_bool(
    cli_value: Optional[bool],
    plan_value: Optional[bool],
    default_value: bool,
) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    if plan_value is not None:
        return bool(plan_value)
    return default_value


def select_candidates(
    plan: Dict[str, Any],
    candidate_ids: List[str],
    apply_all_primary: bool,
) -> List[Dict[str, Any]]:
    primary = plan.get("primary_candidates", [])
    if not isinstance(primary, list) or len(primary) == 0:
        raise ValueError("plan must contain a non-empty primary_candidates list")

    if apply_all_primary:
        return list(primary)

    if candidate_ids:
        selected = [c for c in primary if c.get("candidate_id") in candidate_ids]
        selected_ids = {c.get("candidate_id") for c in selected}
        missing = [cid for cid in candidate_ids if cid not in selected_ids]
        if missing:
            raise ValueError(f"candidate_ids not found in primary_candidates: {missing}")
        return selected

    global_best = [c for c in primary if c.get("role") == "global_best_primary_candidate"]
    if global_best:
        return global_best

    return [primary[0]]


def check_overlaps(candidates: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    index_map: Dict[int, List[str]] = {}

    for cand in candidates:
        cid = cand.get("candidate_id", "")
        indices = cand.get("poisoned_action_indices", [])
        for idx in indices:
            index_map.setdefault(int(idx), []).append(cid)

    return {idx: ids for idx, ids in index_map.items() if len(ids) > 1}


def validate_plan(plan: Dict[str, Any]) -> None:
    required = ["hdf5_dir", "task_name", "demo_id", "primary_candidates"]
    missing = [k for k in required if k not in plan]
    if missing:
        raise ValueError(f"poison plan missing required fields: {missing}")

    if not isinstance(plan["primary_candidates"], list) or len(plan["primary_candidates"]) == 0:
        raise ValueError("poison plan primary_candidates must be a non-empty list")


def get_source_paths(plan: Dict[str, Any]) -> Tuple[str, str, str, int, Optional[int]]:
    source_hdf5_dir = plan["hdf5_dir"]
    task_name = plan["task_name"]
    demo_id = int(plan["demo_id"])
    task_id = plan.get("task_id", None)
    if task_id is not None:
        task_id = int(task_id)

    source_hdf5_path = os.path.join(source_hdf5_dir, f"{task_name}_demo.hdf5")
    return source_hdf5_dir, task_name, source_hdf5_path, demo_id, task_id


def prepare_output_dir(output_hdf5_dir: str, overwrite: bool) -> None:
    if os.path.exists(output_hdf5_dir):
        if not overwrite:
            raise FileExistsError(f"output_hdf5_dir exists and overwrite=False: {output_hdf5_dir}")
        shutil.rmtree(output_hdf5_dir)

    ensure_dir(output_hdf5_dir)


def make_action_diff_row(
    candidate_id: str,
    demo_id: int,
    action_index: int,
    delta: np.ndarray,
    old_action: np.ndarray,
    new_action: np.ndarray,
    clip_action: bool,
    zero_gripper_delta: bool,
) -> Dict[str, Any]:
    effective_delta = new_action - old_action

    return {
        "candidate_id": candidate_id,
        "demo_id": demo_id,
        "action_index": int(action_index),
        "delta": json_list(delta.astype(float).tolist()),
        "old_action": json_list(old_action.astype(float).tolist()),
        "new_action": json_list(new_action.astype(float).tolist()),
        "effective_delta": json_list(effective_delta.astype(float).tolist()),
        "l2_effective_delta": float(np.linalg.norm(effective_delta)),
        "clip_action": bool(clip_action),
        "zero_gripper_delta": bool(zero_gripper_delta),
    }


def apply_candidates_to_actions_array(
    actions: np.ndarray,
    selected_candidates: List[Dict[str, Any]],
    demo_id: int,
    clip_action: bool,
    zero_gripper_delta: bool,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Apply selected candidates to an in-memory action array.

    This is used both for dry_run and for preparing the exact rows before
    writing to HDF5. If candidates overlap and overlap is allowed, later
    candidates are applied on top of previous modifications.
    """
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"actions dataset must have shape (T, 7), got {actions.shape}")

    modified_actions = np.array(actions, dtype=np.float32, copy=True)
    rows: List[Dict[str, Any]] = []

    for cand in selected_candidates:
        cand_id = cand.get("candidate_id")
        indices = cand.get("poisoned_action_indices", [])
        delta = np.array(cand.get("delta", []), dtype=np.float32)

        if delta.shape != (7,):
            raise ValueError(f"candidate {cand_id} delta must be shape (7,), got {delta.shape}")

        if zero_gripper_delta:
            delta[-1] = 0.0

        for idx in indices:
            idx = int(idx)
            if idx < 0 or idx >= modified_actions.shape[0]:
                raise IndexError(
                    f"candidate {cand_id} action index out of range: {idx}; "
                    f"episode length is {modified_actions.shape[0]}"
                )

            old_action = modified_actions[idx].copy()
            new_action = old_action + delta

            if clip_action:
                new_action[:6] = np.clip(new_action[:6], -1.0, 1.0)

            if zero_gripper_delta:
                new_action[-1] = old_action[-1]

            modified_actions[idx] = new_action

            rows.append(
                make_action_diff_row(
                    candidate_id=cand_id,
                    demo_id=demo_id,
                    action_index=idx,
                    delta=delta,
                    old_action=old_action,
                    new_action=new_action,
                    clip_action=clip_action,
                    zero_gripper_delta=zero_gripper_delta,
                )
            )

    return modified_actions, rows


def read_actions_from_hdf5(hdf5_path: str, demo_id: int) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as h5:
        actions_ds = h5["data"][f"demo_{demo_id}"]["actions"]
        actions = actions_ds[()]
    return np.array(actions, dtype=np.float32)


def write_actions_to_hdf5(hdf5_path: str, demo_id: int, modified_actions: np.ndarray) -> None:
    with h5py.File(hdf5_path, "r+") as h5:
        demo_group = h5["data"][f"demo_{demo_id}"]
        actions_ds = demo_group["actions"]

        if actions_ds.shape != modified_actions.shape:
            raise ValueError(
                f"modified actions shape {modified_actions.shape} does not match HDF5 actions shape {actions_ds.shape}"
            )

        actions_ds[...] = modified_actions

        # Helpful attrs; metadata JSON remains the source of truth.
        demo_group.attrs["poisoned"] = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_path", type=str, required=True)
    parser.add_argument("--output_hdf5_dir", type=str, required=True)
    parser.add_argument("--candidate_ids", type=str, default="")
    parser.add_argument("--apply_all_primary", type=str2bool, default=False)
    parser.add_argument("--allow_overlapping_indices", type=str2bool, default=False)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--dry_run", type=str2bool, default=False)
    parser.add_argument("--clip_action", type=str, default=None)
    parser.add_argument("--zero_gripper_delta", type=str, default=None)
    parser.add_argument("--metadata_name", type=str, default="poison_metadata.json")
    parser.add_argument("--diff_csv_name", type=str, default="poison_action_diffs.csv")
    args = parser.parse_args()

    metadata: Dict[str, Any] = {
        "plan_path": args.plan_path,
        "plan_name": None,
        "source_hdf5_dir": None,
        "output_hdf5_dir": args.output_hdf5_dir,
        "source_hdf5_path": None,
        "target_hdf5_path": None,
        "task_name": None,
        "task_id": None,
        "demo_id": None,
        "selected_candidate_ids": [],
        "num_selected_candidates": 0,
        "num_modified_actions": 0,
        "clip_action": None,
        "zero_gripper_delta": None,
        "allow_overlapping_indices": args.allow_overlapping_indices,
        "overlapping_indices": {},
        "diff_csv_path": None,
        "selected_candidates": [],
        "action_diffs": [],
        "dry_run": args.dry_run,
        "source_actions_shape": None,
        "errors": [],
    }

    metadata_path = os.path.join(args.output_hdf5_dir, args.metadata_name)
    diff_csv_path = os.path.join(args.output_hdf5_dir, args.diff_csv_name)

    try:
        plan = load_plan(args.plan_path)
        validate_plan(plan)

        metadata["plan_name"] = plan.get("plan_name", os.path.basename(args.plan_path))

        source_hdf5_dir, task_name, source_hdf5_path, demo_id, task_id = get_source_paths(plan)
        if not os.path.exists(source_hdf5_path):
            raise FileNotFoundError(f"source_hdf5_path not found: {source_hdf5_path}")

        metadata["source_hdf5_dir"] = source_hdf5_dir
        metadata["source_hdf5_path"] = source_hdf5_path
        metadata["task_name"] = task_name
        metadata["task_id"] = task_id
        metadata["demo_id"] = demo_id

        poisoning_strategy = plan.get("poisoning_strategy", {})
        clip_action = effective_bool(
            parse_optional_bool(args.clip_action),
            poisoning_strategy.get("clip_action"),
            True,
        )
        zero_gripper_delta = effective_bool(
            parse_optional_bool(args.zero_gripper_delta),
            poisoning_strategy.get("zero_gripper_delta"),
            True,
        )

        metadata["clip_action"] = clip_action
        metadata["zero_gripper_delta"] = zero_gripper_delta

        candidate_ids = parse_candidate_ids(args.candidate_ids)
        selected_candidates = select_candidates(plan, candidate_ids, args.apply_all_primary)
        selected_candidate_ids = [c.get("candidate_id") for c in selected_candidates]

        metadata["selected_candidate_ids"] = selected_candidate_ids
        metadata["num_selected_candidates"] = len(selected_candidates)
        metadata["selected_candidates"] = selected_candidates

        overlaps = check_overlaps(selected_candidates)
        if overlaps:
            metadata["overlapping_indices"] = {str(k): v for k, v in overlaps.items()}
            if not args.allow_overlapping_indices:
                raise ValueError(
                    "overlapping action indices detected. "
                    "Use --allow_overlapping_indices True only if you intentionally want to stack deltas. "
                    f"overlaps={metadata['overlapping_indices']}"
                )

        # Prepare output directory before writing metadata/diff.
        prepare_output_dir(args.output_hdf5_dir, args.overwrite)

        target_hdf5_path = os.path.join(args.output_hdf5_dir, f"{task_name}_demo.hdf5")
        metadata["target_hdf5_path"] = target_hdf5_path

        # Read source actions in all modes so dry_run can still produce exact diffs.
        source_actions = read_actions_from_hdf5(source_hdf5_path, demo_id)
        metadata["source_actions_shape"] = list(source_actions.shape)

        modified_actions, rows = apply_candidates_to_actions_array(
            actions=source_actions,
            selected_candidates=selected_candidates,
            demo_id=demo_id,
            clip_action=clip_action,
            zero_gripper_delta=zero_gripper_delta,
        )

        metadata["action_diffs"] = rows
        metadata["num_modified_actions"] = len(rows)
        metadata["diff_csv_path"] = diff_csv_path

        if not args.dry_run:
            # Copy the entire HDF5 directory, then write only to the copied target file.
            if os.path.exists(args.output_hdf5_dir):
                # prepare_output_dir already created the directory, but copytree requires
                # a non-existing target. Remove the empty directory before copytree.
                shutil.rmtree(args.output_hdf5_dir)

            shutil.copytree(source_hdf5_dir, args.output_hdf5_dir)

            if not os.path.exists(target_hdf5_path):
                raise FileNotFoundError(f"target_hdf5_path not found after copytree: {target_hdf5_path}")

            write_actions_to_hdf5(target_hdf5_path, demo_id, modified_actions)
        else:
            # Dry run intentionally does not copy or modify HDF5. The target path is planned only.
            metadata["target_hdf5_path"] = target_hdf5_path

        # Write metadata and diff after successful dry-run computation or HDF5 modification.
        write_csv(diff_csv_path, rows)
        write_json(metadata_path, metadata)

        print_summary(metadata_path, diff_csv_path, metadata)

    except Exception as exc:
        metadata["errors"].append(str(exc))
        metadata["errors"].append(traceback.format_exc())

        # Best-effort error metadata. Avoid crashing while reporting the original error.
        try:
            ensure_dir(args.output_hdf5_dir)
            write_json(metadata_path, metadata)
        except Exception:
            pass

        print_summary(metadata_path, None, metadata)


def print_summary(metadata_path: str, diff_csv_path: Optional[str], metadata: Dict[str, Any]) -> None:
    print("=== Apply LIBERO Poison Plan Summary ===")
    print(f"plan: {metadata.get('plan_path')}")
    print(f"source_hdf5_dir: {metadata.get('source_hdf5_dir')}")
    print(f"output_hdf5_dir: {metadata.get('output_hdf5_dir')}")
    print(f"task: {metadata.get('task_name')}")
    print(f"task_id: {metadata.get('task_id')}")
    print(f"demo_id: {metadata.get('demo_id')}")
    print(f"selected_candidate_ids: {metadata.get('selected_candidate_ids')}")
    print(f"num_modified_actions: {metadata.get('num_modified_actions')}")
    print(f"clip_action: {metadata.get('clip_action')}")
    print(f"zero_gripper_delta: {metadata.get('zero_gripper_delta')}")
    print(f"allow_overlapping_indices: {metadata.get('allow_overlapping_indices')}")
    print(f"overlapping_indices: {metadata.get('overlapping_indices')}")
    print(f"dry_run: {metadata.get('dry_run')}")
    print(f"metadata_path: {metadata_path}")
    print(f"diff_csv_path: {diff_csv_path}")
    print(f"target_hdf5_path: {metadata.get('target_hdf5_path')}")

    errors = metadata.get("errors") or []
    if errors:
        print(f"errors: {len(errors)}")
        print(errors[0])


if __name__ == "__main__":
    main()