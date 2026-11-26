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

"""Apply the official quantile on training episodes only and check every frame."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from omegaconf import OmegaConf

sys.path.insert(
    0,
    str(
        Path(os.environ["REPO_PATH"])
        / "examples/offline_rl/advantage_labeling/recap/process"
    ),
)
import compute_advantages as recap


def align_bootstrap_values(df, shard_ids, next_rows, nonterminal):
    """Use each frame's own value for both appearances in the RECAP formula."""
    current = df.value_current.to_numpy()
    canonical = np.zeros_like(current)
    canonical[nonterminal] = current[next_rows[nonterminal]]
    overlap = np.zeros(len(df), dtype=bool)
    overlap[nonterminal] = shard_ids[nonterminal] != shard_ids[next_rows[nonterminal]]
    original = df.value_next.to_numpy()
    np.testing.assert_array_equal(original[~overlap], canonical[~overlap])
    delta = np.abs(original[overlap] - canonical[overlap])
    # Shards independently infer their lookahead in a different batch. The
    # unsharded algorithm looks up one V per frame across the whole episode.
    df["value_next"] = canonical
    df["advantage"] = df.reward_sum.to_numpy() + canonical - current
    return {
        "overlap_rows": int(overlap.sum()),
        "different_repeated_values": int(np.count_nonzero(delta)),
        "max_repeated_value_difference": float(delta.max(initial=0.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("shards", type=Path)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    tag = cfg.advantage.tag
    manifest = json.loads((args.input / "episode_split.json").read_text())["sources"]
    data = {}
    pools = []
    report = {"tag": tag, "sources": {}, "shard_overlap": {}}
    assert float(cfg.data.gamma) == 1.0
    for spec in cfg.data.train_data_paths:
        p = Path(spec.dataset_path)
        episodes = list(
            map(json.loads, (p / "meta/episodes.jsonl").read_text().splitlines())
        )
        lengths = np.array([e["length"] for e in episodes])
        flags = np.array([e["is_success"] for e in episodes])
        ep = np.repeat(np.arange(len(episodes)), lengths)
        ends = np.cumsum(lengths)
        starts = ends - lengths
        frame = np.arange(len(ep)) - starts[ep]
        df = (
            pd.concat(
                [
                    pd.read_parquet(f).assign(shard_id=int(f.stem))
                    for f in sorted((args.shards / p.name).glob("*.parquet"))
                ],
                ignore_index=True,
            )
            .sort_values(["episode_index", "frame_index"])
            .reset_index(drop=True)
        )
        np.testing.assert_array_equal(df.episode_index, ep)
        np.testing.assert_array_equal(df.frame_index, frame)
        assert np.isfinite(df.select_dtypes("number")).all().all()
        rewards = (
            pq.read_table(
                p / "meta" / f"returns_{cfg.advantage.returns_tag}.parquet",
                columns=["episode_index", "frame_index", "return", "reward"],
            )
            .to_pandas()
            .sort_values(["episode_index", "frame_index"])
        )
        np.testing.assert_array_equal(rewards.episode_index, ep)
        np.testing.assert_array_equal(rewards.frame_index, frame)
        np.testing.assert_allclose(rewards["return"], df["return"], rtol=0, atol=1e-6)
        idx = np.arange(len(ep))
        k = np.minimum(10, ends[ep] - idx)
        nxt = idx + k
        valid = nxt < ends[ep]
        cumulative = np.r_[0.0, np.cumsum(rewards.reward.to_numpy().reshape(-1))]
        raw = cumulative[nxt] - cumulative[idx]
        np.testing.assert_allclose(raw, df.reward_sum_raw, atol=1e-6, rtol=0)
        np.testing.assert_allclose(raw / 2088.0, df.reward_sum, atol=1e-6, rtol=0)
        np.testing.assert_array_equal(k, df.num_valid_rewards)
        np.testing.assert_allclose(
            raw / 2088.0 + df.value_next - df.value_current,
            df.advantage,
            atol=1e-6,
            rtol=0,
        )
        report["shard_overlap"][p.name] = align_bootstrap_values(
            df, df.pop("shard_id").to_numpy(), nxt, valid
        )
        split = manifest[p.name]
        assert sorted(split["train"] + split["val"]) == list(range(len(lengths)))
        train = np.isin(ep, split["train"])
        pools.append(df.advantage.to_numpy()[train])
        df["dataset_name"] = p.name
        data[p.name] = (p, spec.type, df, train, flags[ep])
    threshold = float(recap.quantile_threshold(np.concatenate(pools), 0.3))
    report["threshold"] = threshold
    for name, (p, kind, df, train, success) in data.items():
        recap.save_advantages_to_dataset(p, df, threshold, dataset_type=kind, tag=tag)
        path = p / "meta" / f"advantages_{tag}.parquet"
        side = pd.read_parquet(path)
        expected = (
            np.ones(len(df), dtype=bool)
            if kind == "sft"
            else df.advantage.to_numpy() >= threshold
        )
        np.testing.assert_array_equal(side.advantage, expected)
        np.testing.assert_array_equal(side.advantage_continuous, df.advantage)
        detail = {}
        for label, mask in [("train", train), ("val", ~train)]:
            detail[label] = {}
            for group, g in [
                ("all", mask),
                ("success", mask & success),
                ("failure", mask & ~success),
            ]:
                part = side[g]
                detail[label][group] = {
                    "episodes": int(part.episode_index.nunique()),
                    "frames": len(part),
                    "positive": int(part.advantage.sum()),
                    "positive_rate": float(part.advantage.mean())
                    if len(part)
                    else None,
                    "quantiles": part.advantage_continuous.quantile(
                        [0, 0.1, 0.5, 0.7, 0.9, 1]
                    ).to_dict()
                    if len(part)
                    else {},
                }
        report["sources"][name] = detail
        mix = recap.read_mixture_config(p)
        mix.update(
            global_return_min=-2088.0,
            global_return_max=0.0,
            datasets=[
                {"name": Path(s.dataset_path).name, "weight": s.weight, "type": s.type}
                for s in cfg.data.train_data_paths
            ],
        )
        mix.setdefault("tags", {})[tag] = {
            "unified_threshold": threshold,
            "positive_quantile": 0.3,
            "threshold_split": "train",
        }
        recap.write_mixture_config(p, mix)
    totals = [
        sum(source[k]["all"]["frames"] for source in report["sources"].values())
        for k in ("train", "val")
    ]
    assert totals == [1350185, 74963]
    (args.input / "advantage_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(totals, threshold, flush=True)


if __name__ == "__main__":
    main()
