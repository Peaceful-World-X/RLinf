"""get_model for openpi_rl_token: frozen pi05 backbone + trainable RL token heads."""

import json
import os

import safetensors.torch as st
from omegaconf import DictConfig, OmegaConf


def get_model(cfg: DictConfig, torch_dtype=None):
    from rlinf.models.embodiment.openpi import get_model as get_openpi_model
    from rlinf.models.embodiment.openpi.openpi_action_model import OpenPi0ForRLActionPrediction
    from rlinf.models.embodiment.openpi.rl_token_policy import OpenPiRLTokenConfig, OpenPiRLTokenPolicy

    # 1. Load frozen pi05 backbone
    openpi_cfg = OmegaConf.merge(cfg, OmegaConf.create({"model_type": "openpi"}))
    backbone: OpenPi0ForRLActionPrediction = get_openpi_model(openpi_cfg, torch_dtype)
    backbone.requires_grad_(False)

    # 2. Build RL token config — read from checkpoint metadata if available
    rl_token_path = cfg.get("rl_token_path", None)
    if rl_token_path is not None:
        metadata_file = os.path.join(rl_token_path, "metadata.json")
        with open(metadata_file) as f:
            rl_config_from_ckpt = json.load(f)["rl_config"]
        rl_cfg = OpenPiRLTokenConfig(
            hidden_dim=rl_config_from_ckpt["hidden_dim"],
            rl_token_dim=rl_config_from_ckpt.get("rl_token_dim") or rl_config_from_ckpt["hidden_dim"],
            rl_token_encoder_layers=rl_config_from_ckpt["encoder_layers"],
            rl_token_decoder_layers=rl_config_from_ckpt["decoder_layers"],
            rl_token_num_heads=rl_config_from_ckpt["num_heads"],
            rl_token_max_seq_len=rl_config_from_ckpt["max_seq_len"],
            rl_token_dropout=rl_config_from_ckpt.get("dropout", 0.1),
            robot_state_dim=cfg.get("robot_state_dim", 14),
            actor_hidden_dims=tuple(cfg.get("actor_hidden_dims", [512, 256])),
            critic_hidden_dims=tuple(cfg.get("critic_hidden_dims", [512, 256])),
            action_horizon=cfg.get("action_horizon", cfg.get("num_action_chunks", 10)),
            action_dim=cfg.get("action_dim", 7),
            recon_loss_coef=cfg.get("recon_loss_coef", 0.1),
        )
    else:
        rl_cfg = OpenPiRLTokenConfig(
            hidden_dim=cfg.get("rl_token_hidden_dim", 2048),
            rl_token_dim=cfg.get("rl_token_dim", 2048),
            rl_token_encoder_layers=cfg.get("rl_token_encoder_layers", 2),
            rl_token_decoder_layers=cfg.get("rl_token_decoder_layers", 2),
            rl_token_num_heads=cfg.get("rl_token_num_heads", 8),
            rl_token_max_seq_len=cfg.get("rl_token_max_seq_len", 768),
            rl_token_dropout=cfg.get("rl_token_dropout", 0.1),
            num_image_tokens=cfg.get("num_image_tokens", 768),
            prefix_feature_type=cfg.get("prefix_feature_type", "image_only"),
            robot_state_dim=cfg.get("robot_state_dim", 14),
            actor_hidden_dims=tuple(cfg.get("actor_hidden_dims", [512, 256])),
            critic_hidden_dims=tuple(cfg.get("critic_hidden_dims", [512, 256])),
            action_horizon=cfg.get("action_horizon", cfg.get("num_action_chunks", 10)),
            action_dim=cfg.get("action_dim", 7),
            recon_loss_coef=cfg.get("recon_loss_coef", 0.1),
        )

    class OpenPiRLTokenPolicyWithBackbone(OpenPiRLTokenPolicy):
        def _build_prefix_cache_from_obs(self, obs):
            import openpi.models.model as _model

            bb = self.backbone
            processed = bb.input_transform(bb.obs_processor(obs), transpose=False)
            processed = bb.precision_processor(processed)
            observation = _model.Observation.from_dict(processed)
            images, img_masks, lang_tokens, lang_masks, _ = (
                bb._preprocess_observation(observation, train=False)
            )
            prefix_output, prefix_pad_masks, past_key_values = bb._build_prefix_cache(
                images, img_masks, lang_tokens, lang_masks
            )
            return prefix_output, prefix_pad_masks, past_key_values

        def _get_vla_ref_action(self, obs):
            """Run VLA flow matching to get reference action ã = πvla(s)."""
            actions, result = self.backbone.predict_action_batch(obs, mode="eval", compute_values=False)
            return actions

    model = OpenPiRLTokenPolicyWithBackbone(rl_cfg)
    model.backbone = backbone

    # 3. Load pre-trained RL token autoencoder weights
    # Checkpoint keys: encoder.* / decoder.*
    # Policy keys:     rl_token_autoencoder.encoder.* / rl_token_autoencoder.decoder.*
    if rl_token_path is not None:
        raw = st.load_file(os.path.join(rl_token_path, "model.safetensors"), device="cpu")
        remapped = {"rl_token_autoencoder." + k: v for k, v in raw.items()}
        missing, _ = model.load_state_dict(remapped, strict=False)
        autoencoder_missing = [k for k in missing if "rl_token_autoencoder" in k
                               and not k.startswith("target_")]
        if autoencoder_missing:
            raise RuntimeError(f"Failed to load RL token weights, missing: {autoencoder_missing}")
        # Sync target network from loaded online weights (tau=1.0)
        import copy
        model.target_rl_token_autoencoder.load_state_dict(
            model.rl_token_autoencoder.state_dict()
        )

    model.freeze_backbone()
    return model

