# Copyright 2026 The GIGA Authors.
#
from pathlib import Path

import torch

from examples.embodiment.train_intervention_classifier import (
    ClassifierConfig,
    build_classifier_rows,
    compute_pre_intervention_labels,
    grouped_split_trajectories,
    trajectory_number_from_path,
)


def test_pre_intervention_labels_mark_two_chunks_before_intervention():
    flags = torch.zeros(8, 1, 90, dtype=torch.bool)
    flags[5:, 0, :] = True

    labels, source_steps = compute_pre_intervention_labels(
        flags,
        pre_intervention_chunks=2,
        include_positive_window=1,
    )

    assert labels.tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
    assert source_steps == {3: [5]}


def test_pre_intervention_labels_support_positive_window():
    flags = torch.zeros(9, 1, 90, dtype=torch.bool)
    flags[6:, 0, :] = True

    labels, source_steps = compute_pre_intervention_labels(
        flags,
        pre_intervention_chunks=2,
        include_positive_window=2,
    )

    assert labels.tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 0]
    assert source_steps == {4: [6], 3: [6]}


def test_pre_intervention_labels_clip_negative_target_indices():
    flags = torch.zeros(4, 1, 90, dtype=torch.bool)
    flags[1, 0, :] = True

    labels, source_steps = compute_pre_intervention_labels(
        flags,
        pre_intervention_chunks=2,
        include_positive_window=1,
    )

    assert labels.tolist() == [0, 0, 0, 0]
    assert source_steps == {}


def test_trajectory_number_from_path():
    assert trajectory_number_from_path(Path("/tmp/trajectory_42_abc.pt")) == 42


def test_grouped_split_keeps_trajectories_disjoint():
    rows = [
        {"traj_key": "a", "label": 0},
        {"traj_key": "a", "label": 1},
        {"traj_key": "b", "label": 0},
        {"traj_key": "c", "label": 1},
        {"traj_key": "d", "label": 0},
        {"traj_key": "e", "label": 1},
    ]

    splits = grouped_split_trajectories(
        rows, val_fraction=0.2, test_fraction=0.2, seed=123
    )

    split_sets = {
        name: {rows[i]["traj_key"] for i in idxs} for name, idxs in splits.items()
    }
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])
    assert sum(len(v) for v in splits.values()) == len(rows)


def test_build_classifier_rows_aligns_feature_cache(tmp_path):
    traj_path = tmp_path / "trajectory_0_test.pt"
    flags = torch.zeros(6, 1, 90, dtype=torch.bool)
    flags[4:, 0, :] = True
    states = torch.zeros(6, 1, 14)
    torch.save({"intervene_flags": flags, "curr_obs": {"states": states}}, traj_path)

    cache_path = tmp_path / "features.pt"
    torch.save(
        {
            "rltoken": torch.randn(6, 8),
            "metas": [
                {"path": str(traj_path), "rel_chunk": i, "traj_index": 0}
                for i in range(6)
            ],
        },
        cache_path,
    )

    cfg = ClassifierConfig(
        train_data_dirs=[str(tmp_path)],
        output_dir=str(tmp_path / "out"),
        norm_stats_path="unused_norm.json",
        urdf_path="unused.urdf",
        feature_cache=str(cache_path),
        z_dim=8,
        pre_intervention_chunks=2,
        include_positive_window=1,
        negative_sample_ratio=0.0,
    )
    data = build_classifier_rows(cfg)

    positive_rows = [r for r in data["rows"] if r["label"] == 1]
    assert len(positive_rows) == 1
    assert positive_rows[0]["rel"] == 2
    assert positive_rows[0]["source_intervention_steps"] == [4]
