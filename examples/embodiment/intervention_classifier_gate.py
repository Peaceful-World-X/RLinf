#!/usr/bin/env python3
# Copyright 2026 The GIGA Authors.
#
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from examples.embodiment.train_intervention_classifier import InterventionClassifier


@dataclass(frozen=True)
class InterventionGateDecision:
    actor_intervene: bool
    mode: str
    classifier_probability: float | None
    classifier_triggered: bool


class InterventionClassifierGate:
    def __init__(
        self,
        model: InterventionClassifier | None = None,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval() if model is not None else None
        self.enabled = model is not None
        self.last_probability: float | None = None

    @classmethod
    def from_optional_checkpoint(
        cls, checkpoint_path: str | Path | None, device: torch.device | str = "cpu"
    ) -> "InterventionClassifierGate":
        if checkpoint_path is None or str(checkpoint_path).strip() == "":
            return cls(model=None, device=device)
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Intervention classifier checkpoint not found: {path}"
            )
        checkpoint: dict[str, Any] = torch.load(
            path, map_location="cpu", weights_only=False
        )
        ckpt_cfg = checkpoint.get("config", {})
        z_dim = int(ckpt_cfg.get("z_dim", 2048))
        hidden_dim = int(ckpt_cfg.get("hidden_dim", 256))
        num_layers = int(ckpt_cfg.get("num_layers", 2))
        dropout = float(ckpt_cfg.get("dropout", 0.1))
        model = InterventionClassifier(z_dim, hidden_dim, num_layers, dropout)
        model.load_state_dict(checkpoint["model"])
        return cls(model=model, device=device)

    @torch.no_grad()
    def predict_probability(self, zrl: torch.Tensor) -> float:
        if self.model is None:
            raise RuntimeError(
                "Classifier gate is disabled; probability is unavailable"
            )
        z = zrl.detach().to(self.device).float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        logits = self.model(z)
        probability = float(torch.sigmoid(logits).reshape(-1)[0].detach().cpu().item())
        self.last_probability = probability
        return probability

    def decide_actor_intervention(
        self,
        zrl: torch.Tensor,
        chunk_idx: int,
        warm_up_chunks: int,
        threshold: float,
    ) -> InterventionGateDecision:
        if not self.enabled:
            actor_intervene = int(chunk_idx) >= int(warm_up_chunks)
            return InterventionGateDecision(
                actor_intervene=bool(actor_intervene),
                mode="warm_up_chunks",
                classifier_probability=None,
                classifier_triggered=False,
            )

        probability = self.predict_probability(zrl)
        triggered = probability >= float(threshold)
        return InterventionGateDecision(
            actor_intervene=bool(triggered),
            mode="classifier",
            classifier_probability=probability,
            classifier_triggered=bool(triggered),
        )
