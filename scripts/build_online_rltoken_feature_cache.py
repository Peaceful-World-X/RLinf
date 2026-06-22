#!/usr/bin/env python3
# Copyright 2026 The GIGA Authors.
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

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_repo_root()))

from rlinf.models.embodiment.openpi.rl_token_policy import (  # noqa: E402
    OpenPiRLTokenConfig,
    OpenPiRLTokenPolicy,
)


def _trajectory_number(path: Path) -> int:
    m = re.search(r"trajectory_(\d+)_", path.name)
    return int(m.group(1)) if m else -1


def build_policy(
    rl_token_path: Path, norm_stats_path: Path, device: torch.device
) -> OpenPiRLTokenPolicy:
    metadata = json.loads((rl_token_path / "metadata.json").read_text())
    rl_cfg = metadata["rl_config"]
    cfg = OpenPiRLTokenConfig(
        hidden_dim=int(rl_cfg["hidden_dim"]),
        rl_token_dim=int(rl_cfg["rl_token_dim"]),
        rl_token_encoder_layers=int(rl_cfg["encoder_layers"]),
        rl_token_decoder_layers=int(rl_cfg["decoder_layers"]),
        rl_token_num_heads=int(rl_cfg["num_heads"]),
        rl_token_max_seq_len=int(rl_cfg["max_seq_len"]),
        rl_token_dropout=float(rl_cfg.get("dropout", 0.1)),
        num_image_tokens=768,
        prefix_feature_type="image_only",
        robot_state_dim=6,
        actor_head_type="openrlt_mlp",
        critic_head_type="openrlt_mlp",
        openrlt_z_proj_dim=256,
        openrlt_state_proj_dim=64,
        openrlt_action_proj_dim=256,
        openrlt_hidden_dim=256,
        openrlt_num_layers=2,
        action_horizon=10,
        action_dim=6,
        actor_output_bound=1.2,
        actor_residual_ref=True,
        actor_residual_scale=1.0,
        use_robot_state=True,
        critic_use_robot_state=True,
        critic_use_ref_action=False,
        critic_use_rl_token=True,
        action_space="normalized_delta",
        action_norm_stats_path=str(norm_stats_path),
        action_norm_std_floor=1.0,
        env_action_dim=14,
        controlled_action_indices=[7, 8, 9, 10, 11, 12],
    )
    policy = OpenPiRLTokenPolicy(cfg)
    state = load_safetensors(str(rl_token_path / "model.safetensors"), device="cpu")
    policy.load_state_dict(
        {f"rl_token_autoencoder.{k}": v for k, v in state.items()}, strict=False
    )
    policy.to(device)
    policy.eval()
    return policy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rl-token-path", type=Path, required=True)
    ap.add_argument("--norm-stats-path", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--tcp-max", type=float, default=0.10)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy = build_policy(args.rl_token_path, args.norm_stats_path, device)
    data_dir = args.log_dir / "replay_buffer" / "rank_0"
    paths = sorted(data_dir.glob("trajectory_*.pt"), key=_trajectory_number)
    if not paths:
        raise FileNotFoundError(f"No trajectory_*.pt under {data_dir}")

    features = []
    metas = []
    with torch.no_grad():
        for ti, path in enumerate(paths):
            traj = torch.load(path, map_location="cpu", weights_only=False)
            fwd = traj.get("forward_inputs", {})
            visual = fwd.get(
                "visual_latent", traj.get("curr_obs", {}).get("visual_latent")
            )
            if visual is None:
                print(f"[skip] {path.name}: no visual_latent", flush=True)
                continue
            visual = visual[:, 0].float()
            tcp = fwd.get("tcp_distance_to_target", None)
            if tcp is not None:
                tcp_flat = tcp[:, 0].reshape(-1).float()
            else:
                tcp_flat = torch.zeros(visual.shape[0])
            rewards = traj.get("rewards")
            success = bool(
                rewards is not None
                and float(torch.as_tensor(rewards).float().sum().item()) > 0
            )
            for start in range(0, visual.shape[0], args.batch_size):
                end = min(visual.shape[0], start + args.batch_size)
                batch = visual[start:end].to(device)
                # visual_latent saved by rollout is already image prefix features [B, 768, 2048].
                token = (
                    policy.rl_token_autoencoder.encoder(batch).detach().cpu().float()
                )
                if token.dim() == 3:
                    token = token.mean(dim=1)
                for local, z in enumerate(token):
                    rel = start + local
                    features.append(z)
                    metas.append(
                        {
                            "path": str(path),
                            "rel_chunk": int(rel),
                            "traj_index": int(_trajectory_number(path)),
                            "success": bool(success),
                            "tcp_distance_to_target": float(tcp_flat[rel].item())
                            if rel < tcp_flat.numel()
                            else None,
                        }
                    )
            print(
                f"[cache] {ti + 1}/{len(paths)} {path.name} chunks={visual.shape[0]} success={int(success)}",
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rltoken": torch.stack(features, dim=0),
        "metas": metas,
        "source_log_dir": str(args.log_dir),
        "num_trajectories": len(paths),
    }
    torch.save(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "num_rows": len(metas),
                "shape": list(payload["rltoken"].shape),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
