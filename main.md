# RLT 两阶段训练

```bash
# 1. 设置相关变量
ROOT=/mnt/20T-1/sudu
REPO="$ROOT/RLinf"
RUNS="$ROOT/.data/RLinf"

S1="$RUNS/g1_rlt/20261027_1042"
OUT1="$ROOT/.result/g1_rlt/20261027_1042/rlt_20000"
ACTOR_RUN="$RUNS/g1_actor_splitbc_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ACTOR_RUN"
S1_NAME=g1_rlt_stage1_sft_openpi_pi05
S2_NAME=g1_rlt_stage2_offline
ENTRY="$REPO/examples/embodiment/train_offline_rl.py"

cd "$REPO"
source env/bin/activate
export PYTHONPATH="$REPO"  # 使用仓库源码
export RAY_TMPDIR="$ROOT/.data/ray_g1_default"

# 2. 训练阶段 1：冻结 VLA，只训练 RLT
nohup python -u toolkits/ensure_20000.py \
  >> "$S1/ensure_20000.log" 2>&1 < /dev/null &

# 配置：$REPO/examples/sft/config/$S1_NAME.yaml
#      GPU 1；batch 32；20k 步；调度器自动恢复、验证和绘图。
# 进度：cat "$S1/target_progress.json"
# 权重：$OUT1/checkpoints/global_step_20000/actor/model_state_dict/full_weights.pt
# 曲线：$OUT1/plots/rlt_loss.png

# 3. 训练阶段 2：state + delta，人工/自主成功分别设置 BC
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
EMBODIED_PATH="$REPO/examples/embodiment" \
nohup python -u toolkits/launch_official.py "$ENTRY" \
  --config-name "$S2_NAME" \
  runner.logger.log_path="$ACTOR_RUN" \
  >> "$ACTOR_RUN/launch.log" 2>&1 < /dev/null &

# 配置：$REPO/examples/embodiment/config/$S2_NAME.yaml
# 复用 20k RLT 缓存；bc_human_weight=50，bc_policy_weight=5，Q=0.1。
# 自主失败不做 BC，仍训练 Q；只训练 14 关节，夹爪沿用 VLA。
# critic 20k / actor 10k（2:1）；取消两个新 BC 参数即可恢复旧统一 bc_weight。
# CUDA_VISIBLE_DEVICES/OMP_NUM_THREADS：GPU/线程。
# EMBODIED_PATH：共享配置；RAY_TMPDIR：独立 Ray 临时目录。
# 权重和验证记录：$ACTOR_RUN/training/；运行日志：$ACTOR_RUN/launch.log。
```
