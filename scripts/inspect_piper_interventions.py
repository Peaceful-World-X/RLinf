#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inspect human intervention actions in saved Piper replay-buffer trajectories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _as_tensor(value: Any) -> torch.Tensor:
    return value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)


def _format_vec(vec: np.ndarray, precision: int = 4) -> str:
    return np.array2string(
        np.asarray(vec, dtype=np.float64),
        precision=precision,
        suppress_small=True,
        max_line_width=220,
    )


def _parse_indices(text: str | None, max_len: int) -> set[int] | None:
    if text is None or str(text).strip().lower() in {"", "all", "*"}:
        return None
    indices: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, end = int(left), int(right)
            if end < start:
                raise ValueError(f"Bad descending range {part!r}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    bad = [idx for idx in indices if idx < 0 or idx >= max_len]
    if bad:
        raise ValueError(f"indices out of range [0, {max_len - 1}]: {bad}")
    return indices


def _iter_pt_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("trajectory_*.pt"))


def _reshape_action(tensor: torch.Tensor, length: int) -> torch.Tensor:
    tensor = tensor[:length].float().cpu()
    if tensor.dim() == 2:
        tensor = tensor[:, None, :]
    if tensor.shape[-1] % 14 != 0:
        raise ValueError(f"Expected last dim to be H*14, got {tuple(tensor.shape)}")
    horizon = tensor.shape[-1] // 14
    return tensor.reshape(length, tensor.shape[1], horizon, 14)


def _intervention_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trajectory = torch.load(path, map_location="cpu", weights_only=False)
    if "actions" not in trajectory:
        raise KeyError(f"{path}: missing top-level actions")
    if "forward_inputs" not in trajectory:
        raise KeyError(f"{path}: missing forward_inputs")

    fi = trajectory["forward_inputs"]
    required = (
        "env_action_absolute",
        "intervene_env_action_absolute",
        "executed_env_action_absolute",
    )
    missing = [key for key in required if key not in fi]
    if missing:
        raise KeyError(f"{path}: missing forward_inputs keys {missing}")

    length = int(trajectory["actions"].shape[0])
    policy = _reshape_action(_as_tensor(fi["env_action_absolute"]), length)
    human = _reshape_action(_as_tensor(fi["intervene_env_action_absolute"]), length)
    executed = _reshape_action(_as_tensor(fi["executed_env_action_absolute"]), length)
    actor = _reshape_action(
        _as_tensor(fi.get("actor_env_action_absolute", fi["env_action_absolute"])),
        length,
    )
    ref = _reshape_action(
        _as_tensor(fi.get("ref_env_action_absolute", fi["env_action_absolute"])),
        length,
    )

    flags = _as_tensor(
        trajectory.get(
            "intervene_flags", torch.zeros_like(trajectory["actions"], dtype=torch.bool)
        )
    ).bool()
    flags = _reshape_action(flags.to(torch.float32), length).bool()
    sub_mask = flags.any(dim=-1)
    nonzero = torch.nonzero(sub_mask, as_tuple=False).tolist()

    owner_tensor = fi.get("rollout_control_source", None)
    if owner_tensor is not None:
        owner_tensor = _as_tensor(owner_tensor).reshape(length, -1)

    rows: list[dict[str, Any]] = []
    for chunk, batch, substep in nonzero:
        p = policy[chunk, batch, substep].numpy()
        h = human[chunk, batch, substep].numpy()
        e = executed[chunk, batch, substep].numpy()
        a = actor[chunk, batch, substep].numpy()
        r = ref[chunk, batch, substep].numpy()
        diff = h - p
        actor_ref = a - r
        exec_human = e - h
        owner = "unknown"
        if owner_tensor is not None:
            owner = "ACTOR" if int(owner_tensor[chunk, 0].item()) == 1 else "VLA"
        rows.append(
            {
                "trajectory": path.name,
                "chunk": int(chunk),
                "batch": int(batch),
                "substep": int(substep),
                "owner": owner,
                "human_minus_policy_max_abs": float(np.abs(diff).max()),
                "human_minus_policy_l2": float(np.linalg.norm(diff)),
                "actor_minus_ref_max_abs": float(np.abs(actor_ref).max()),
                "executed_minus_human_max_abs": float(np.abs(exec_human).max()),
                "policy": p,
                "human": h,
                "diff": diff,
                "actor": a,
                "ref": r,
            }
        )

    summary = {
        "trajectory": path.name,
        "length": length,
        "intervened_substeps": len(rows),
        "intervened_chunks": sorted({row["chunk"] for row in rows}),
    }
    if rows:
        max_abs = np.asarray([row["human_minus_policy_max_abs"] for row in rows])
        l2 = np.asarray([row["human_minus_policy_l2"] for row in rows])
        summary.update(
            {
                "diff_max_abs_mean": float(max_abs.mean()),
                "diff_max_abs_max": float(max_abs.max()),
                "diff_l2_mean": float(l2.mean()),
                "diff_l2_max": float(l2.max()),
            }
        )
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare human intervention actions with the policy actions that would have executed."
    )
    parser.add_argument(
        "--trajectory",
        required=True,
        help="Path to one trajectory_*.pt or a replay_buffer/rank_* directory.",
    )
    parser.add_argument(
        "--chunks", default="all", help="Chunk filter, e.g. all, 36, 36-38"
    )
    parser.add_argument(
        "--substeps", default="all", help="Substep filter, e.g. all, 0, 3-9"
    )
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--print-vectors", action="store_true")
    parser.add_argument("--csv", default=None, help="Optional CSV summary path.")
    args = parser.parse_args()

    paths = _iter_pt_files(Path(args.trajectory).expanduser())
    if not paths:
        raise FileNotFoundError(args.trajectory)

    all_rows: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    for path in paths:
        rows, summary = _intervention_rows(path)
        max_chunk = max(summary["length"], 1)
        chunk_filter = _parse_indices(args.chunks, max_chunk)
        substep_filter = _parse_indices(args.substeps, 10)
        rows = [
            row
            for row in rows
            if (chunk_filter is None or row["chunk"] in chunk_filter)
            and (substep_filter is None or row["substep"] in substep_filter)
        ]
        all_rows.extend(rows)
        all_summaries.append(summary)

    for summary in all_summaries:
        print("")
        print(summary["trajectory"])
        print(
            "  length={length}, intervened_substeps={intervened_substeps}, intervened_chunks={intervened_chunks}".format(
                **summary
            )
        )
        if summary["intervened_substeps"]:
            print(
                "  human-policy max_abs mean/max={:.5f}/{:.5f}, l2 mean/max={:.5f}/{:.5f}".format(
                    summary["diff_max_abs_mean"],
                    summary["diff_max_abs_max"],
                    summary["diff_l2_mean"],
                    summary["diff_l2_max"],
                )
            )

    rows_to_print = all_rows[: max(args.max_rows, 0)]
    if rows_to_print:
        print("\nRows:")
    for row in rows_to_print:
        print(
            "{trajectory} chunk={chunk:03d} sub={substep} owner={owner} "
            "human-policy max_abs={human_minus_policy_max_abs:.5f} "
            "l2={human_minus_policy_l2:.5f} actor-ref max_abs={actor_minus_ref_max_abs:.5f} "
            "executed-human max_abs={executed_minus_human_max_abs:.6f}".format(**row)
        )
        if args.print_vectors:
            print("  would_execute:", _format_vec(row["policy"]))
            print("  human        :", _format_vec(row["human"]))
            print("  human-policy :", _format_vec(row["diff"]))
            print("  actor        :", _format_vec(row["actor"]))
            print("  ref          :", _format_vec(row["ref"]))

    if args.csv:
        csv_path = Path(args.csv).expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "trajectory",
            "chunk",
            "batch",
            "substep",
            "owner",
            "human_minus_policy_max_abs",
            "human_minus_policy_l2",
            "actor_minus_ref_max_abs",
            "executed_minus_human_max_abs",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in all_rows:
                writer.writerow({key: row[key] for key in fields})
        print(f"\nWrote CSV: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
