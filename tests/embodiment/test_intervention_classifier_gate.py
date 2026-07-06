# Copyright 2026 The GIGA Authors.
#
from pathlib import Path

import pytest
import torch

from examples.embodiment.intervention_classifier_gate import (
    InterventionClassifierGate,
)
from examples.embodiment.train_intervention_classifier import InterventionClassifier


class FakeProbabilityGate(InterventionClassifierGate):
    def __init__(self, probabilities, enabled=True):
        self.probabilities = list(probabilities)
        self.enabled = bool(enabled)
        self.model = None
        self.device = torch.device("cpu")
        self.last_probability = None

    def predict_probability(self, zrl: torch.Tensor) -> float:
        if not self.probabilities:
            raise AssertionError("No probability left for fake gate")
        prob = float(self.probabilities.pop(0))
        self.last_probability = prob
        return prob


def test_disabled_classifier_uses_warm_up_chunks_after_boundary():
    gate = FakeProbabilityGate([0.99], enabled=False)

    before = gate.decide_actor_intervention(
        zrl=torch.zeros(1, 4),
        chunk_idx=15,
        warm_up_chunks=16,
        threshold=0.8,
    )
    after = gate.decide_actor_intervention(
        zrl=torch.zeros(1, 4),
        chunk_idx=16,
        warm_up_chunks=16,
        threshold=0.8,
    )

    assert before.actor_intervene is False
    assert before.mode == "warm_up_chunks"
    assert after.actor_intervene is True
    assert after.mode == "warm_up_chunks"
    assert gate.probabilities == [0.99]


def test_enabled_classifier_ignores_warm_up_chunks_boundary():
    gate = FakeProbabilityGate([0.2], enabled=True)

    decision = gate.decide_actor_intervention(
        zrl=torch.zeros(1, 4),
        chunk_idx=20,
        warm_up_chunks=16,
        threshold=0.8,
    )

    assert decision.actor_intervene is False
    assert decision.mode == "classifier"
    assert decision.classifier_probability == 0.2
    assert decision.classifier_triggered is False


def test_enabled_classifier_decides_each_chunk_without_latch():
    gate = FakeProbabilityGate([0.9, 0.1], enabled=True)

    first = gate.decide_actor_intervention(
        zrl=torch.zeros(1, 4),
        chunk_idx=10,
        warm_up_chunks=16,
        threshold=0.8,
    )
    second = gate.decide_actor_intervention(
        zrl=torch.zeros(1, 4),
        chunk_idx=11,
        warm_up_chunks=16,
        threshold=0.8,
    )

    assert first.actor_intervene is True
    assert first.classifier_triggered is True
    assert second.actor_intervene is False
    assert second.classifier_triggered is False
    assert gate.probabilities == []


def test_empty_classifier_path_disables_classifier():
    gate = InterventionClassifierGate.from_optional_checkpoint(
        checkpoint_path=None,
        device="cpu",
    )

    assert gate.enabled is False


def test_missing_nonempty_checkpoint_raises_clear_error(tmp_path):
    missing = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="Intervention classifier checkpoint"):
        InterventionClassifierGate.from_optional_checkpoint(missing, device="cpu")


def test_existing_checkpoint_enables_classifier_and_predicts_probability(tmp_path):
    checkpoint_path = tmp_path / "classifier.pt"
    real_model = InterventionClassifier(
        z_dim=4, hidden_dim=4, num_layers=1, dropout=0.0
    )
    torch.save(
        {
            "model": real_model.state_dict(),
            "config": {"z_dim": 4, "hidden_dim": 4, "num_layers": 1, "dropout": 0.0},
        },
        checkpoint_path,
    )

    gate = InterventionClassifierGate.from_optional_checkpoint(
        checkpoint_path, device="cpu"
    )

    assert Path(checkpoint_path).exists()
    assert gate.enabled is True
    prob = gate.predict_probability(torch.zeros(1, 4))
    assert 0.0 <= prob <= 1.0
