# Online TD3 Compressed Replay Design

## Goal

Make real-world online TD3 store and train from frozen actor/critic prefix embeddings instead of full `[L, 2048]` VLA prefix tensors, while retaining compatibility with existing full-prefix replay data and safely increasing the replay cache/window to 10 trajectories.

## Data Format

When `actor.model.replay_embedding_mode` is `frozen_image_last_linear`, each rollout observation stores three float32 vectors:

- `actor_rl_token`: actor prefix projection output.
- `critic_rl_token_1`: first critic prefix projection output.
- `critic_rl_token_2`: second critic prefix projection output.

The rollout result does not store `visual_latent` in this mode. State, reference action, executed action, reward, termination, and intervention fields remain unchanged.

## Compatibility

The online actor installs a replay trajectory transform before loading a checkpoint. If an old trajectory contains `visual_latent`, the transform projects `curr_obs` and `next_obs` in bounded batches, stores the three compressed vectors, removes all full-prefix tensors, and leaves the source checkpoint files unchanged. Already compressed trajectories pass through unchanged.

The policy accepts either a full prefix tensor or a mapping containing the three precomputed vectors. Full-prefix behavior remains the default for offline training and other configurations.

## Safety

Compressed mode requires `rl_token_source: image_last_linear`, `actor_train_prefix_token_linear: false`, `critic_train_prefix_token_linear: false`, and `critic_train_representation: false`. Startup fails with a clear error if any invariant is violated.

## Verification

Tests compare actions, critic states, and Q values produced from full-prefix input with values produced from compressed embeddings. Additional tests cover old trajectory conversion, compressed transition assembly, configuration invariants, and a cache-size-10 memory smoke using existing replay shapes.
