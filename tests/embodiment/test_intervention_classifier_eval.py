# Copyright 2026 The GIGA Authors.
#
import numpy as np
import torch

from examples.embodiment.eval_intervention_classifier import (
    build_trigger_summary,
    extract_three_view_images_for_rows,
    first_trigger_rel,
    latched_trigger_mask,
    resolve_decision_threshold,
    select_image_timeline_groups,
)


def test_resolve_decision_threshold_uses_yaml_threshold_without_cli_override():
    assert resolve_decision_threshold(args_threshold=None, cfg_threshold=0.8) == 0.8


def test_resolve_decision_threshold_uses_cli_override_when_provided():
    assert resolve_decision_threshold(args_threshold=0.6, cfg_threshold=0.8) == 0.6


def test_first_trigger_rel_returns_first_probability_crossing():
    rows = [{"rel": 3}, {"rel": 4}, {"rel": 5}]
    probs = np.array([0.2, 0.61, 0.9], dtype=np.float32)

    assert first_trigger_rel(rows, probs, threshold=0.6) == 4


def test_first_trigger_rel_returns_none_without_crossing():
    rows = [{"rel": 3}, {"rel": 4}]
    probs = np.array([0.2, 0.59], dtype=np.float32)

    assert first_trigger_rel(rows, probs, threshold=0.6) is None


def test_latched_trigger_mask_stays_active_after_first_crossing():
    rows = [{"rel": rel} for rel in [0, 1, 2, 3, 4]]
    probs = np.array([0.1, 0.7, 0.2, 0.1, 0.8], dtype=np.float32)

    mask = latched_trigger_mask(rows, probs, threshold=0.6)

    assert mask.tolist() == [False, True, True, True, True]


def test_build_trigger_summary_reports_labels_and_source_steps():
    rows = [
        {
            "path": "/tmp/trajectory_1_test.pt",
            "traj_index": 1,
            "rel": 2,
            "label": 0,
            "source_intervention_steps": [],
        },
        {
            "path": "/tmp/trajectory_1_test.pt",
            "traj_index": 1,
            "rel": 3,
            "label": 1,
            "source_intervention_steps": [5],
        },
        {
            "path": "/tmp/trajectory_1_test.pt",
            "traj_index": 1,
            "rel": 4,
            "label": 0,
            "source_intervention_steps": [],
        },
    ]
    probs = np.array([0.2, 0.7, 0.8], dtype=np.float32)
    labels = np.array([0, 1, 0], dtype=np.int64)

    summary = build_trigger_summary(rows, probs, labels, threshold=0.6)

    assert summary == [
        {
            "path": "/tmp/trajectory_1_test.pt",
            "traj_index": 1,
            "threshold": 0.6,
            "has_trigger": True,
            "first_trigger_rel": 3,
            "first_trigger_frame": 3,
            "first_trigger_probability": float(probs[1]),
            "actor_intervention_start_rel": 3,
            "active_until_end": True,
            "label_positive_rels": [3],
            "source_intervention_steps": [5],
        }
    ]


def test_select_image_timeline_groups_balances_positive_and_negative_trajectories():
    rows = []
    labels = []
    probs = []
    specs = [
        ("pos_a", [0, 1], [0.2, 0.7]),
        ("pos_b", [0, 1], [0.1, 0.8]),
        ("pos_c", [0, 1], [0.1, 0.9]),
        ("neg_a", [0, 0], [0.6, 0.2]),
        ("neg_b", [0, 0], [0.7, 0.2]),
        ("neg_c", [0, 0], [0.1, 0.2]),
    ]
    for path, path_labels, path_probs in specs:
        for rel, (label, prob) in enumerate(zip(path_labels, path_probs)):
            rows.append({"path": path, "rel": rel})
            labels.append(label)
            probs.append(prob)

    groups = select_image_timeline_groups(
        rows,
        np.asarray(probs, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        threshold=0.6,
        max_trajectories=4,
    )

    assert [group["category"] for group in groups] == [
        "positive",
        "positive",
        "negative",
        "negative",
    ]
    assert {group["path"] for group in groups} == {"pos_a", "pos_b", "neg_a", "neg_b"}


def test_extract_three_view_images_for_rows_uses_curr_obs(tmp_path):
    traj_path = tmp_path / "trajectory_0_test.pt"
    main = torch.zeros(4, 1, 8, 8, 3, dtype=torch.uint8)
    extra = torch.zeros(4, 1, 2, 8, 8, 3, dtype=torch.uint8)
    main[2, 0, :, :, 0] = 101
    extra[2, 0, 0, :, :, 1] = 102
    extra[2, 0, 1, :, :, 2] = 103
    torch.save(
        {"curr_obs": {"main_images": main, "extra_view_images": extra}}, traj_path
    )

    rows = [{"path": str(traj_path), "rel": 2}]
    images = extract_three_view_images_for_rows(rows, thumb_size=6)

    assert images.shape == (1, 3, 6, 6, 3)
    assert images[0, 0, 0, 0, 0] == 101
    assert images[0, 1, 0, 0, 1] == 102
    assert images[0, 2, 0, 0, 2] == 103
