# Offline TD3 + RLToken 离线训练算法详解

**命令**：`bash examples/embodiment/run_piper_offline_td3.sh realworld_piper_peginsertion_td3_rl_token_real`

**代码入口链**：
1. [`examples/embodiment/run_piper_offline_td3.sh`](../examples/embodiment/run_piper_offline_td3.sh) → 设置环境变量，调用 `train_offline_rl.py`
2. [`examples/embodiment/train_offline_rl.py`](../examples/embodiment/train_offline_rl.py) → `OfflineTD3FSDPPolicy`
3. [`rlinf/workers/actor/fsdp_offline_td3_policy_worker.py`](../rlinf/workers/actor/fsdp_offline_td3_policy_worker.py) → 继承 `EmbodiedTD3FSDPPolicy`
4. [`rlinf/workers/actor/fsdp_td3_policy_worker.py`](../rlinf/workers/actor/fsdp_td3_policy_worker.py) → 主训练循环
5. [`rlinf/algorithms/td3.py`](../rlinf/algorithms/td3.py) → 算法核心（Critic/Actor loss 计算）
6. [`rlinf/models/embodiment/openpi/rl_token_policy.py`](../rlinf/models/embodiment/openpi/rl_token_policy.py) → 模型定义
7. [`rlinf/models/embodiment/openpi/rl_token/__init__.py`](../rlinf/models/embodiment/openpi/rl_token/__init__.py) → 模型构建与权重冻结

---

## 1. 模型架构

### 1.1 整体结构

```
OpenPiRLTokenPolicy
├── backbone (PI0Pytorch / pi05)                 [FROZEN]
│   └── 负责将图像+语言输入 → prefix_output tokens (B, L, 2048)
├── rl_token_autoencoder (RLTokenAutoencoder)    [FROZEN, freeze_rl_token=True]
│   ├── encoder (RLTokenEncoder)                 → (B, L, 2048) → rl_token (B, 256)
│   └── decoder (RLTokenDecoder)                 → 重建 prefix tokens (用于预训练)
├── actor_head (TransformerActionHead)           [TRAINABLE]
│   └── input: [rl_token(256)] + [ref_action(10×14=140)] → output: action (10×14)
├── critic_head_1 (ValueHead MLP)                [TRAINABLE]
└── critic_head_2 (ValueHead MLP)                [TRAINABLE]
    └── input: critic_state + action → Q值 (scalar)

Target networks (EMA copy, requires_grad=False):
├── target_rl_token_autoencoder
├── target_actor_head
├── target_critic_head_1
└── target_critic_head_2
```

**代码位置**：[`rl_token/__init__.py:136-211`](../rlinf/models/embodiment/openpi/rl_token/__init__.py#L136-L211)

### 1.2 RLTokenEncoder 详解

```
输入: prefix_output[:, :768, :] (image-only tokens, shape B×768×2048)
     (prefix_feature_type="image_only", num_image_tokens=768)
  ↓ LayerNorm + 可学习位置编码
  ↓ Append 可学习 probe token → shape (B, 769, 2048)
  ↓ 2层 TransformerEncoder
  ↓ 取最后一个位置(probe) → Linear(2048→256)
输出: rl_token (B, 256)
```

**代码位置**：[`rl_token/rl_token.py:50-109`](../rlinf/models/embodiment/openpi/rl_token/rl_token.py#L50-L109)

### 1.3 TransformerActionHead 详解（actor_head_type="transformer"）

```
输入 x = cat([rl_token(256), ref_action(140)])  → (B, 396)
  ↓ 拆分: rl_token_proj → token(B,1,256)
        action_proj(ref_action chunk i) + pos_emb + type_emb → (B,10,256)
  ↓ cat → sequence (B, 11, 256)  [1个rl_token + 10个action步]
  ↓ 2层 TransformerEncoder(dim=256, heads=8, ffn=1024)
  ↓ 取最后10个位置 → Linear(256→14)
输出: action (B, 10, 14)
  ↓ tanh(·/1.2) * 1.2  [actor_output_bound=1.2]
```

**代码位置**：[`rl_token_policy.py:85-179`](../rlinf/models/embodiment/openpi/rl_token_policy.py#L85-L179)

### 1.4 Critic 输入构造

```python
critic_state = cat([
    rl_token (256),           # critic_use_rl_token=True
    robot_state (14),         # critic_use_robot_state=True
    ref_action (10×14=140),   # critic_use_ref_action=True
])  # → (B, 410)

critic_input = cat([critic_state (410), action (140)])  # → (B, 550)
Q = ValueHead(550 → [512, 256] → 1)
```

**代码位置**：[`rl_token_policy.py:630-646`](../rlinf/models/embodiment/openpi/rl_token_policy.py#L630-L646)

---

## 2. 参数更新情况

| 模块 | 参数量 | 是否更新 | 被哪个 optimizer 更新 | 说明 |
|------|--------|----------|----------------------|------|
| PI0Pytorch 骨干（VLA） | ~3B | ❌ 冻结 | — | `backbone.requires_grad_(False)` |
| `rl_token_autoencoder.encoder` | ~小 | ❌ 冻结 | — | `freeze_rl_token=True` |
| `rl_token_autoencoder.decoder` | ~小 | ❌ 冻结 | — | `freeze_rl_token=True` |
| `actor_head` (TransformerActionHead) | ~小 | ✅ 更新 | `actor_optimizer` | 每步更新（critic_actor_ratio=1） |
| `critic_head_1` (ValueHead) | ~小 | ✅ 更新 | `critic_optimizer` | 每步更新 |
| `critic_head_2` (ValueHead) | ~小 | ✅ 更新 | `critic_optimizer` | 每步更新 |
| `target_*` (EMA targets) | — | ❌ 无梯度 | — | `soft_update` τ=0.05 更新 |

**关键代码**（optimizer 分组）：
```python
# fsdp_td3_policy_worker.py:113-129
critic_param_filters = ["critic_head_1", "critic_head_2"]
# critic_train_rl_token_encoder=False → rl_token_encoder 不进 critic_optimizer
param_filters = {"critic": critic_param_filters}
optimizers = self.build_optimizers(model, main_optim_config, param_filters, ...)
self.actor_optimizer = optimizers[0]   # 所有参数 except critic_head_*
self.critic_optimizer = optimizers[1]  # critic_head_1, critic_head_2 only
```

**代码位置**：[`fsdp_td3_policy_worker.py:109-134`](../rlinf/workers/actor/fsdp_td3_policy_worker.py#L109-L134)

> ⚠️ `actor_optimizer` 虽包含 rl_token_autoencoder 的参数，但 `freeze_rl_token=True` 后 `requires_grad=False`，因此实际上只有 `actor_head` 接受梯度更新。

---

## 3. 动作归一化（normalized_delta）

**动作空间**：`action_space = "normalized_delta"`，14维双臂 Piper 动作

### 3.1 推理时（数据 → 训练空间）

```
原始绝对关节角 a_abs (B, 10, 14)
  ↓ 减去当前状态 s (B, 14)：仅前6维 + 后6维（mask=[T,T,T,T,T,T,F,T,T,T,T,T,T,F]）
  ↓ delta = a_abs[..., mask] - s[:, None, mask]    # gripper不做delta
  ↓ (delta - mean) / (std + 1e-6)                  # z-score归一化
  ↓ gripper维度保持 absolute（不归一化）
输出: normalized_delta action (B, 10, 14)，在网络中使用
```

### 3.2 执行时（训练空间 → 绝对角度）

```
a_norm (B, 10, 14)
  ↓ a_norm * (std + 1e-6) + mean
  ↓ absolute[..., mask] = delta[..., mask] + s[:, None, mask]
输出: a_abs 绝对关节角
```

### 3.3 归一化参数

- 来源：`action_norm_stats_path = ".../norm_stats.json"` → `stats["actions"]["mean"]` / `stats["actions"]["std"]`
- `action_norm_std_floor = 1.0`：std < 1.0 的维度用 1.0 替代，防止对小方差维度过度放大

**代码位置**：[`rl_token_policy.py:361-411`](../rlinf/models/embodiment/openpi/rl_token_policy.py#L361-L411)

---

## 4. 数据采样（Replay Buffer）

### 4.1 数据加载

```python
# fsdp_offline_td3_policy_worker.py:47-79
self.replay_buffer = TrajectoryReplayBuffer(
    enable_cache=True, cache_size=10,
    sample_window_size=205,  # 采样窗口内的轨迹数
)
self.replay_buffer.load_checkpoint(path)  # 加载 .pt 格式离线轨迹
```

每条轨迹存储：`curr_obs`, `next_obs`, `actions`, `rewards`, `terminations`, `forward_inputs`（含`mc_return`, `ref_action`, `visual_latent`）

### 4.2 n-step Return 计算

**存储时**预计算（`replay_buffer.py:336-`）：
```
gamma=0.96, n_step=10
n_step_return = r_t + γ*r_{t+1} + ... + γ^9*r_{t+9}
n_step_discount = γ^10 * (1 - done_{t+10})
```

### 4.3 Tail Curriculum（渐进采样窗口）

随着训练推进，逐步扩大每条轨迹内的采样范围（从 tail 开始向前扩展）：

```python
# fsdp_td3_policy_worker.py:608-629
start_window=1   # 初始只采 tail 最后1个chunk
end_window=96    # 最终采 tail 最后96个chunks
hold_steps=100   # 前100步不扩展（暖机期）
warmup_steps=1000  # 100~1100步线性从1扩展到96

window = start + progress * (end - start)  # progress ∈ [0,1]
```

**意义**：防止早期训练时Critic从无意义的初始状态学习，先聚焦在轨迹成功/失败的关键末端段。

### 4.4 Batch 采样与 Micro-batch 分割

```
global_batch_size = 64 (per GPU)
micro_batch_size = 8
gradient_accumulation = 64 / 8 / 1 = 8

→ 每个训练步：sample 64个chunks，拆成8个micro_batch，
  每个micro_batch前向+反向，梯度累积后统一 optimizer.step()
```

**代码位置**：[`fsdp_td3_policy_worker.py:466-525`](../rlinf/workers/actor/fsdp_td3_policy_worker.py#L466-L525)

---

## 5. 两阶段训练策略

### 5.1 训练阶段切换

```python
# td3.py:1752-1768
def apply_training_stage(self, update_step):
    if update_step >= coupling_start_step(=800):
        → "coupled_td3" 阶段
    else:
        → "bootstrap" 阶段
```

| 参数 | Bootstrap (step < 800) | Coupled TD3 (step ≥ 800) |
|------|------------------------|--------------------------|
| `critic_target_action_source` | `"data_next"` | `"actor"` |
| `actor_q_coef` | 0.0 | 0.1 |
| `bc_coef` | 50.0 | 50.0 |
| `target_policy_noise` | 0.0 | 0.05 |
| `target_noise_clip` | 0.0 | 0.1 |
| `target_action_clip` | 1.2 | 1.2 |

**Bootstrap 阶段意义**：先用行为克隆（BC loss）训练 Actor 恢复 VLA 行为策略；Critic 以数据中的 next action 为 target，避免引入未成熟 Actor 的噪声。

**Coupled 阶段意义**：Actor 足够稳定后，切换为真正的 TD3 循环：Critic 以 Actor 输出为 target，Actor 加入 Q 梯度信号提升动作质量。

---

## 6. Critic 训练目标

### 6.1 TD 目标计算

```python
# td3.py:1159-1228  compute_critic_loss()

# Step 1: 确定 next_action
if stage == "bootstrap" (data_next):
    next_actions = batch["next_obs"]["ref_action"]  # 数据中的下一步 VLA 动作
else:  # coupled
    next_actions, next_aux = target_model.target_actor_forward(next_obs)
    noise = randn * 0.05，clamp([-0.1, 0.1])
    next_actions = (next_actions + noise).clamp(-1.2, 1.2)

# Step 2: 计算 TD target
target_q1, target_q2 = target_critic(next_rl_state, next_actions)
target_q = min(target_q1, target_q2)  # clipped double-Q

n_step_return = batch["n_step_return"]  # γ^0*r_t + ... + γ^9*r_{t+9}
chunk_discount = batch["n_step_discount"]  # γ^10 * (1-done)
y = n_step_return + chunk_discount * target_q  # Bellman target
```

### 6.2 TD Loss

```python
# td3.py:1243-1244
q1, q2 = model.critic(curr_rl_state, actions)
L_TD = MSE(q1, y) + MSE(q2, y)
critic_loss = 1.0 * L_TD  # critic_td_coef=1.0
```

### 6.3 Critic RL-state 的构造

```
curr_obs → backbone (no_grad) → prefix_output
         → rl_token_autoencoder.encoder (no_grad, freeze_rl_token=True)
         → rl_token (B, 256)

critic_rl_state = cat([rl_token(256), robot_state(14), ref_action(140)])
                = (B, 410)
```

> **注意**：`critic_train_representation=False`（default），因此 RL-token encoder 对 Critic 的前向传播不计算梯度（`torch.no_grad()` 包裹）。Critic 梯度只流向 `critic_head_*`，不影响 encoder。

**代码位置**：[`td3.py:1133-1158`](../rlinf/algorithms/td3.py#L1133-L1158)

---

## 7. Actor 训练目标

### 7.1 Actor 前向

```python
# fsdp_td3_policy_worker.py:408-460  forward_actor()

visual_feat = curr_obs  # offline: 直接传 obs 字典
ref_action = curr_obs["ref_action"]  # VLA 的参考动作 ã = π_VLA(s)

actions, aux = model(
    mode="actor", visual_feat=visual_feat,
    ref_action=ref_action, compute_recon_loss=False
)
# actor forward 内：rl_token_autoencoder.encoder (no_grad) → rl_token
# actor_head([rl_token, ref_action]) → actions (B, 10, 14)
```

### 7.2 BC Loss

```python
L_BC = MSE(actor_actions, sampled_actions)  # sampled_actions = batch["actions"]
```

### 7.3 Q Loss（只在 Coupled 阶段生效）

```python
# actor_q_coef=0.0 (bootstrap) or 0.1 (coupled)
if actor_q_coef > 0:
    q1, q2 = model(mode="critic_q", rl_state=aux["critic_rl_state"].detach(), action=actions)
    q_pi = min(q1, q2)
    q_term = (-q_pi).mean()  # 最大化 Q 值

L_actor = actor_q_coef * q_term + bc_coef * L_BC
        = 0.0 * (-Q) + 50.0 * L_BC    # bootstrap
        = 0.1 * (-Q) + 50.0 * L_BC    # coupled
```

> `critic_rl_state.detach()`：Actor 更新时 Q 梯度**不流回** rl_token encoder，只流向 `actor_head`。

**代码位置**：[`fsdp_td3_policy_worker.py:407-460`](../rlinf/workers/actor/fsdp_td3_policy_worker.py#L407-L460)

### 7.4 Reconstruction Loss

config 中 `recon_loss_coef=0.0` → **本实验关闭**，不计算重建 loss。

---

## 8. 目标网络更新（Soft Update）

### 8.1 EMA 更新公式

```
θ_target ← (1 - τ) * θ_target + τ * θ_online    # τ = 0.05
```

### 8.2 更新范围

每步同时更新（`_soft_update_internal_policy_targets`）：

| Online 模块 | Target 模块 |
|-------------|-------------|
| `rl_token_autoencoder.*` | `target_rl_token_autoencoder.*` |
| `actor_head.*` | `target_actor_head.*` |
| `critic_head_1.*` | `target_critic_head_1.*` |
| `critic_head_2.*` | `target_critic_head_2.*` |

以及 FSDP `target_model`（完整模型副本）的全量 soft update。

```python
# fsdp_td3_policy_worker.py:246-303
def soft_update_target_model(self, tau=0.05):
    for (n1, online), (n2, target) in zip(model.params, target_model.params):
        target.data.mul_(1-tau).add_(online.data * tau)
    self._soft_update_internal_policy_targets(tau)  # 更新 policy 内部 target_*
```

**更新频率**：`target_update_freq=1`，每个 update_step 都执行。

**代码位置**：[`fsdp_td3_policy_worker.py:246-303`](../rlinf/workers/actor/fsdp_td3_policy_worker.py#L246-L303)

---

## 9. 完整训练步循环

```
每个 update_step：
  1. apply_training_stage(update_step)     # 判断 bootstrap vs coupled
  2. _update_tail_curriculum_window()      # 更新采样尾窗口大小
  3. sample global_batch (64 chunks)       # 从 replay buffer 采样
  4. split → 8 micro_batches

  [Critic Update]
  5. critic_optimizer.zero_grad()
  6. for micro_batch in micro_batches:
       loss, metrics = forward_critic(batch)
       (loss / 8).backward()              # 梯度累积
  7. clip_grad_norm_(max_norm=1.0)
  8. critic_optimizer.step()              # 只更新 critic_head_*
  9. qf_lr_scheduler.step()

  [Actor Update]  (critic_actor_ratio=1, 每步都更新)
  10. actor_optimizer.zero_grad()
  11. for micro_batch in micro_batches:
        loss, metrics = forward_actor(batch)
        (loss / 8).backward()
  12. clip_grad_norm_(max_norm=1.0)
  13. actor_optimizer.step()             # 只更新 actor_head (rl_token frozen)
  14. lr_scheduler.step()

  [Target Update]
  15. soft_update_target_model(tau=0.05) # EMA 更新所有 target_*
```

**代码位置**：[`fsdp_td3_policy_worker.py:466-605`](../rlinf/workers/actor/fsdp_td3_policy_worker.py#L466-L605)

---

## 10. Optimizer 配置

| 参数 | Actor Optimizer | Critic Optimizer |
|------|----------------|-----------------|
| 类型 | AdamW | AdamW |
| lr | 1e-3 | 1e-3 |
| β1, β2 | 0.9, 0.95 | 0.9, 0.95 |
| ε | 1e-8 | 1e-8 |
| weight_decay | 0.0 | 0.0 |
| grad_clip | 1.0 | 1.0 |
| lr_scheduler | `torch_constant` | `torch_constant` |

---

## 11. Checkpoint 保存策略

`actor_critic_only_checkpoint=True` → 仅保存小型模块权重：

```python
# fsdp_td3_policy_worker.py:823-862
payload = {
    "model": {actor_head.*, critic_head_1.*, critic_head_2.*},
    "target_model": {同上，EMA 版本},
    "actor_optimizer": state_dict,
    "critic_optimizer": state_dict,
}
# 不保存：PI0 backbone, rl_token_autoencoder（太大，且加载时从原始路径恢复）
```

**可视化**（每次 checkpoint 触发）：
- 离线轨迹评估：Q值曲线、Actor vs GT 动作 MSE、TCP 轨迹 3D 图、critic timeline 图

---

## 12. 关键超参数速查（本实验配置）

| 超参数 | 值 | 说明 |
|--------|-----|------|
| `gamma (γ)` | 0.96 | 折扣因子 |
| `n_step` | 10 | n步回报 |
| `tau (τ)` | 0.05 | EMA 软更新系数 |
| `bc_coef` | 50.0 | BC loss 权重（两阶段均有效） |
| `actor_q_coef` | 0.0→0.1 | Q loss 权重（bootstrap→coupled） |
| `coupling_start_step` | 800 | 切换到 coupled TD3 的步数 |
| `global_batch_size` | 64 | 每步批量大小 |
| `micro_batch_size` | 8 | 梯度累积 mini-batch 大小 |
| `tail_curriculum.start_window` | 1 | 初始轨迹采样尾窗口（chunks数） |
| `tail_curriculum.end_window` | 96 | 最大尾窗口 |
| `tail_curriculum.warmup_steps` | 1000 | 窗口扩展周期 |
| `action_norm_std_floor` | 1.0 | 归一化 std 下界 |
| `actor_output_bound` | 1.2 | tanh 输出界限 |
| `max_steps` | 200 (shell默认) | 训练步数上限 |

---

## 13. 文件索引

| 文件 | 作用 |
|------|------|
| [`examples/embodiment/run_piper_offline_td3.sh`](../examples/embodiment/run_piper_offline_td3.sh) | 启动脚本 |
| [`examples/embodiment/train_offline_rl.py`](../examples/embodiment/train_offline_rl.py) | 主入口，选择 worker 类 |
| [`examples/embodiment/config/realworld_piper_peginsertion_td3_rl_token_real.yaml`](../examples/embodiment/config/realworld_piper_peginsertion_td3_rl_token_real.yaml) | 完整配置文件 |
| [`rlinf/runners/offline_runner.py`](../rlinf/runners/offline_runner.py) | 训练主循环（step 调度，checkpoint） |
| [`rlinf/workers/actor/fsdp_offline_td3_policy_worker.py`](../rlinf/workers/actor/fsdp_offline_td3_policy_worker.py) | 离线 TD3 worker，覆盖数据加载 |
| [`rlinf/workers/actor/fsdp_td3_policy_worker.py`](../rlinf/workers/actor/fsdp_td3_policy_worker.py) | TD3 训练循环核心（critic/actor update, soft update） |
| [`rlinf/algorithms/td3.py`](../rlinf/algorithms/td3.py) | TD3 算法：compute_critic_loss, compose_actor_loss |
| [`rlinf/models/embodiment/openpi/rl_token_policy.py`](../rlinf/models/embodiment/openpi/rl_token_policy.py) | OpenPiRLTokenPolicy 模型定义 |
| [`rlinf/models/embodiment/openpi/rl_token/__init__.py`](../rlinf/models/embodiment/openpi/rl_token/__init__.py) | 模型构建工厂（backbone冻结，权重加载，freeze_rl_token） |
| [`rlinf/models/embodiment/openpi/rl_token/rl_token.py`](../rlinf/models/embodiment/openpi/rl_token/rl_token.py) | RLTokenEncoder/Decoder/Autoencoder 实现 |
| [`rlinf/data/replay_buffer.py`](../rlinf/data/replay_buffer.py) | TrajectoryReplayBuffer（轨迹存储，mc_return计算，tail curriculum） |
