# Online TD3 Compressed Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store and train real-world online TD3 replay from frozen actor and critic embeddings while converting existing full-prefix replay at load time.

**Architecture:** Add a dual-input policy path that consumes either full prefix tensors or three precomputed embeddings. Rollout emits compressed embeddings, the env aligns them into transition observations, and the actor-owned replay transform converts legacy trajectories before caching.

**Tech Stack:** Python 3.11, PyTorch, FSDP, Hydra/OmegaConf, pytest.

---

### Task 1: Policy compressed-input contract

**Files:**
- Modify: `tests/unit_tests/test_rl_token_policy_smoke.py`
- Modify: `rlinf/models/embodiment/openpi/rl_token_policy.py`

- [ ] Add a failing test that derives actor/critic embeddings from a randomized `image_last_linear` policy and compares full-prefix actor actions, critic states, and Q values with the precomputed-input path.
- [ ] Run `pytest -q tests/unit_tests/test_rl_token_policy_smoke.py -k precomputed` and confirm the missing API failure.
- [ ] Add `encode_replay_embeddings()` and support the compressed mapping in actor and target-actor TD3 forwards.
- [ ] Re-run the focused policy tests and confirm numerical equivalence.

### Task 2: Transition and legacy replay conversion

**Files:**
- Create: `tests/unit_tests/test_td3_compressed_replay_embeddings.py`
- Modify: `rlinf/workers/env/env_worker.py`
- Modify: `rlinf/data/replay_buffer.py`
- Modify: `rlinf/workers/actor/fsdp_td3_policy_worker.py`

- [ ] Add failing tests for copying all three embedding keys through pending transitions and converting a legacy trajectory without mutating its source file.
- [ ] Run the focused test and confirm the transition/transform APIs are absent.
- [ ] Add reusable transition-feature extraction, an optional replay trajectory transform, and bounded legacy projection in the actor worker.
- [ ] Add startup validation for frozen compression projections and representation settings.
- [ ] Re-run focused tests and verify legacy and compressed trajectories are both accepted.

### Task 3: Online configuration and memory smoke

**Files:**
- Modify: `examples/embodiment/config/realworld_piper_gigacl_charge_insert_online.yaml`
- Test: `tests/unit_tests/test_td3_compressed_replay_embeddings.py`

- [ ] Add a failing config assertion for compressed mode and cache/window 10.
- [ ] Enable `frozen_image_last_linear`, replace `visual_latent` cache keys with the three compressed keys, and set cache/window to 10.
- [ ] Run focused unit tests, config composition, and Python compilation.
- [ ] Load existing replay through the legacy transform smoke and report measured cache tensor bytes and reduction relative to full-prefix caching.
