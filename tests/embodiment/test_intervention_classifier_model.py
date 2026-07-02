# Copyright 2026 The GIGA Authors.
#
import torch

from examples.embodiment.train_intervention_classifier import (
    InterventionClassifier,
    binary_classification_metrics,
)


def test_intervention_classifier_forward_shape():
    model = InterventionClassifier(z_dim=16, hidden_dim=8, num_layers=2, dropout=0.0)
    logits = model(torch.randn(5, 16))
    assert logits.shape == (5,)


def test_binary_metrics_at_threshold():
    logits = torch.tensor([-4.0, 4.0, 3.0, -3.0])
    labels = torch.tensor([0, 1, 0, 1])

    metrics = binary_classification_metrics(logits, labels, threshold=0.5)

    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
