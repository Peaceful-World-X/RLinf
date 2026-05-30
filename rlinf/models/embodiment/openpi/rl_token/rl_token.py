"""RL Token module for compressing prefix token sequences into a single RL token.

This module provides encoder-decoder architecture for learning compressed
representations of token sequences that can be used for RL fine-tuning.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor, nn

# from ..paligemma_with_expert import GemmaRMSNorm


@dataclasses.dataclass(frozen=True)
class RLTokenConfig:
    """Configuration for RL Token encoder/decoder.

    Attributes:
        hidden_dim: Hidden dimension for transformer layers (same as VLA embedding dim).
        rl_token_dim: Output dimension for RL token. Defaults to hidden_dim.
        max_seq_len: Maximum sequence length supported.
        encoder_layers: Number of transformer encoder layers.
        decoder_layers: Number of transformer decoder layers.
        num_heads: Number of attention heads.
        ff_dim: Feedforward dimension. Defaults to hidden_dim * 2.
        dropout: Dropout probability.
        use_rms_norm: Whether to use RMSNorm (Gemma style) instead of LayerNorm.
    """

    hidden_dim: int
    rl_token_dim: int | None = None
    max_seq_len: int = 512
    encoder_layers: int = 2
    decoder_layers: int = 2
    num_heads: int = 8
    ff_dim: int | None = None
    dropout: float = 0.1
    use_rms_norm: bool = True

    def __post_init__(self) -> None:
        if self.rl_token_dim is None:
            object.__setattr__(self, "rl_token_dim", self.hidden_dim)
        if self.ff_dim is None:
            object.__setattr__(self, "ff_dim", self.hidden_dim * 2)


class RLTokenEncoder(nn.Module):
    """Encoder that compresses a sequence of tokens into a single RL token.

    The encoder appends a learnable probe token to the input sequence,
    processes it through transformer layers, and extracts the final
    position as the RL token representation.
    """

    def __init__(self, config: RLTokenConfig):
        """Initialize the RL token encoder.

        Args:
            config: Configuration dataclass for the encoder.
        """
        super().__init__()
        self.config = config
        self.rl_probe = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.position_embed = nn.Parameter(
            torch.randn(1, config.max_seq_len + 1, config.hidden_dim) * 0.02
        )

        # if config.use_rms_norm:
        #     self.input_norm = GemmaRMSNorm(config.hidden_dim)
        # else:
        #     self.input_norm = nn.LayerNorm(config.hidden_dim)

        self.input_norm = nn.LayerNorm(config.hidden_dim)
        

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)

        if config.rl_token_dim != config.hidden_dim:
            self.out_proj = nn.Linear(config.hidden_dim, config.rl_token_dim)
        else:
            self.out_proj = nn.Identity()

    def forward(self, prefix_tokens: Tensor) -> Tensor:
        """Encode prefix tokens into a single RL token.

        Args:
            prefix_tokens: Input token embeddings of shape (B, L, D).

        Returns:
            RL token representation of shape (B, rl_token_dim).
        """
        batch_size = prefix_tokens.shape[0]
        probe = self.rl_probe.expand(batch_size, -1, -1)
        inputs = torch.cat([prefix_tokens, probe], dim=1)
        inputs = self.input_norm(inputs + self.position_embed[:, : inputs.shape[1], :])
        encoded = self.encoder(inputs)
        rl_token = encoded[:, -1, :]
        return self.out_proj(rl_token)


class RLTokenDecoder(nn.Module):
    """Decoder that reconstructs token sequences from an RL token.

    d_phi([z_rl, z̄_{1:i-1}]) -> predict z̄_i

    The RL token is used as the first token of the sequence, concatenated
    with shifted target tokens, then processed through a causal Transformer.
    """

    def __init__(self, config: RLTokenConfig):
        """Initialize the RL token decoder.

        Args:
            config: Configuration dataclass for the decoder.
        """
        super().__init__()
        self.config = config

        if config.rl_token_dim != config.hidden_dim:
            self.rl_proj = nn.Linear(config.rl_token_dim, config.hidden_dim)
        else:
            self.rl_proj = nn.Identity()

        self.pos_embed = nn.Parameter(
            torch.randn(1, config.max_seq_len, config.hidden_dim) * 0.02
        )

        # if config.use_rms_norm:
        #     self.input_norm = GemmaRMSNorm(config.hidden_dim)
        # else:
        #     self.input_norm = nn.LayerNorm(config.hidden_dim)

        self.input_norm = nn.LayerNorm(config.hidden_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=config.decoder_layers)
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

    def _causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Build causal attention mask for decoder.

        Args:
            seq_len: Sequence length.
            device: Target device.

        Returns:
            Upper triangular boolean mask where True means position is masked.
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, rl_token: Tensor, target_tokens: Tensor) -> Tensor:
        """Decode RL token into a sequence of tokens.

        Constructs input as [z_rl, z̄_1, ..., z̄_{L-1}] where z̄_i = sg(z_i),
        then predicts z̄_i at each position using causal attention.

        Args:
            rl_token: RL token of shape (B, rl_token_dim).
            target_tokens: Target tokens from VLA embeddings of shape (B, L, D).

        Returns:
            Reconstructed tokens of shape (B, L, hidden_dim).
        """
        B, L, _ = target_tokens.shape
        if L > self.config.max_seq_len:
            raise ValueError(f"L={L} exceeds max_seq_len={self.config.max_seq_len}")

        target_tokens_sg = target_tokens.detach()

        rl_part = self.rl_proj(rl_token).unsqueeze(1)  # (B, 1, H)
        if L == 1:
            dec_inputs = rl_part
        else:
            shifted = target_tokens_sg[:, :-1, :]  # (B, L-1, H)
            dec_inputs = torch.cat([rl_part, shifted], dim=1)  # (B, L, H)

        dec_inputs = self.input_norm(dec_inputs + self.pos_embed[:, :L, :])

        mask = self._causal_mask(L, target_tokens.device)
        hidden = self.decoder(dec_inputs, mask=mask)  # (B, L, H)
        return self.output_proj(hidden)  # (B, L, H)


class RLTokenAutoencoder(nn.Module):
    """Autoencoder combining RL token encoder and decoder.

    This module compresses token sequences into RL tokens and reconstructs
    them, enabling training via reconstruction loss.
    """

    def __init__(self, config: RLTokenConfig):
        """Initialize the autoencoder.

        Args:
            config: Configuration dataclass for encoder and decoder.
        """
        super().__init__()
        self.config = config
        self.encoder = RLTokenEncoder(config)
        self.decoder = RLTokenDecoder(config)

    def forward(self, input_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Encode and decode input tokens.

        Args:
            input_tokens: Input token embeddings of shape (B, L, D).

        Returns:
            Tuple of (rl_token, reconstructed_tokens):
                - rl_token: Compressed representation of shape (B, rl_token_dim).
                - reconstructed_tokens: Reconstructed sequence of shape (B, L, D).
        """
        rl_token = self.encoder(input_tokens)
        recon = self.decoder(rl_token, input_tokens)
        return rl_token, recon


def _expanded_mask(reference: Tensor, token_mask: Tensor | None) -> Tensor:
    """Expand token mask to match reference tensor shape.

    Args:
        reference: Reference tensor for shape and device.
        token_mask: Optional boolean mask of shape (B, L).

    Returns:
        Expanded mask of shape (B, L) with float dtype.
    """
    if token_mask is None:
        return torch.ones(reference.shape[:-1], dtype=reference.dtype, device=reference.device)
    return token_mask.to(dtype=reference.dtype)


def reconstruction_loss(
    target_tokens: Tensor,
    recon_tokens: Tensor,
    token_mask: Tensor | None = None,
) -> Tensor:
    """Compute reconstruction loss .

    L_ro = E_D [ sum_i || h_phi(d_phi([z_rl, z̄_{1:i-1}]))_i - z̄_i ||_2^2 ]

    where z̄_i = sg(z_i) is the stop-gradient of the VLA embeddings.
    The inner sum is over all sequence positions (not averaged).

    Args:
        target_tokens: Ground truth tokens of shape (B, L, D).
        recon_tokens: Reconstructed tokens of shape (B, L, D).
        token_mask: Optional mask of shape (B, L) indicating valid tokens.

    Returns:
        Scalar loss: sum over positions, mean over batch.
    """
    target_tokens_sg = target_tokens.detach()
    token_l2_sq = ((recon_tokens - target_tokens_sg) ** 2).sum(dim=-1)  # (B, L)

    if token_mask is None:
        return token_l2_sq.sum(dim=-1).mean()

    mask = token_mask.to(dtype=token_l2_sq.dtype)
    return (token_l2_sq * mask).sum(dim=-1).mean()


def reconstruction_metrics(
    target_tokens: Tensor,
    recon_tokens: Tensor,
    token_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Compute scalar reconstruction metrics.

    All returned tensors are detached scalars, safe to pass directly to
    ``parse_losses`` / ``accelerator.log``.

    Args:
        target_tokens: Ground truth tokens of shape (B, L, D).
        recon_tokens: Reconstructed tokens of shape (B, L, D).
        token_mask: Optional mask of shape (B, L) indicating valid tokens.

    Returns:
        Dictionary of scalar tensors:
            - recon/mse: Mean squared error per feature element.
            - recon/rmse: Root mean squared error.
            - recon/mae: Mean absolute error per feature element.
            - recon/cosine_sim: Mean cosine similarity between target and recon tokens.
            - recon/token_l2: Mean L2 norm of per-token reconstruction error.
            - recon/valid_tokens: Number of valid (unmasked) tokens.
    """
    with torch.no_grad():
        target = target_tokens.detach()
        recon = recon_tokens.detach()

        diff = recon - target
        squared_error = diff.square()
        absolute_error = diff.abs()
        expanded_mask = _expanded_mask(target, token_mask)
        valid = expanded_mask.sum().clamp_min(1.0)

        feature_dim = target.shape[-1]
        expanded_mask_3d = expanded_mask.unsqueeze(-1)
        masked_squared = squared_error * expanded_mask_3d
        masked_absolute = absolute_error * expanded_mask_3d

        mse = masked_squared.sum() / (valid * feature_dim)
        mae = masked_absolute.sum() / (valid * feature_dim)
        token_l2 = torch.sqrt(masked_squared.sum(dim=-1).sum() / valid)

        target_norm = torch.linalg.norm(target, dim=-1)
        recon_norm = torch.linalg.norm(recon, dim=-1)
        cosine = torch.nn.functional.cosine_similarity(target, recon, dim=-1)
        cosine_mask = (
            expanded_mask
            * (target_norm > 0).to(expanded_mask.dtype)
            * (recon_norm > 0).to(expanded_mask.dtype)
        )
        cosine_valid = cosine_mask.sum()
        if cosine_valid.item() == 0:
            mean_cosine = torch.zeros(1, dtype=target.dtype, device=target.device).squeeze()
        else:
            mean_cosine = (cosine * cosine_mask).sum() / cosine_valid

    return {
        "recon/mse": mse,
        "recon/rmse": torch.sqrt(mse),
        "recon/mae": mae,
        "recon/cosine_sim": mean_cosine,
        "recon/token_l2": token_l2,
        "recon/valid_tokens": valid,
    }