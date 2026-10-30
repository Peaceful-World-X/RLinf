# RLinf

```bash
## 初始化环境
# 安装uv


# 清华 PyPI 镜像
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export HF_ENDPOINT="https://hf-mirror.com"

# 模型下载位置
export DOWNLOAD_DIR=../.data/.model/recap_models

# 官方安装脚本
cd /16T-2/sudu/code/RLinf
bash requirements/install.sh embodied --model openpi --env maniskill_libero --venv "$PWD/.venv"

# [可选] 价值曲线绘图及视频读写
uv pip install matplotlib av  
source env/bin/activate

## 常用脚本
rsync -avz --progress -e 'ssh -p 22' \
  /Users/joker/Downloads/20260826_0908_merged_d14_lerobotv3 \
  sudu@30.229.200.131:/16T-2/sudu/code/.data/zd_20260826_0908_merged_d14_lerobotv3/


```

--- 

# RECAP

https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/recap.html
官方流程为「基础模型 → 回报 → 价值训练 → 优势标注 → policy 训练」。

RECAP 流程需要以下预训练模型：
- SigLIP2-so400m：视觉编码器，用于 Step 2 价值模型训练
- Gemma3-270M：语言模型，用于 Step 2 价值模型训练
- pi0.5 base (PyTorch)：策略模型，用于 Step 4 CFG 训练。

### 训练验证价值函数
```bash
source env/bin/activate
cd /16T-2/sudu/code/RLinf
export CUDA_VISIBLE_DEVICES=2,3

ENTRY=examples/offline_rl/advantage_labeling/recap/ur7_value.py
# 新增配置
# examples/offline_rl/config/recap_ur7_value.yaml
# 主要内容 task.dataset、failure_episodes、output_root 和 actor.model 下的模型路径。
# 本例数据：/16T-2/sudu/code/.data/bund_20260917_n906_v21_new；失败轨迹 198、357。

# 新训练：准备回报 → 短训/恢复检查 → 最多 8k 步 → 选最优权重 → 评估/视频。
env/bin/python "$ENTRY" all --config recap_ur7_value
# 新结果：/16T-2/sudu/code/.result2/RECAP/RLinf/<时间>/，保留 best.pt。

# 已有训练示例：4500 步早停，最佳 2000 步；验证 MAE 0.02009，基线 0.06248。
RUN=/16T-2/sudu/code/.result2/RECAP/RLinf/20261021_ur7_value
env/bin/python "$ENTRY" evaluate --output-dir "$RUN"  # 复验该运行原数据，重绘其测试轨迹
cat "$RUN/report.md"

# 刚才的新数据示例：bund_20260827_n216_v21_new 的 episode_000000，128 帧/12.8 秒。
# 输入位于该数据集 videos/chunk-000/observation.images.base/episode_000000.mp4，并配对 wrist。
VIDEO="$RUN/playback/best_step_2000/bund_20260827_n216_v21_new/episode_000000.mp4"
ffplay -autoexit "$VIDEO"  # 有桌面环境时播放；也可直接用文件播放器打开
# 此新数据视频由临时脚本生成；evaluate 目前只复验 RUN，不能直接传入任意新视频。
# 价值是归一化回报；新数据没有结果标签，视频只画预测与均值基线，不报告 MAE。
```

---

# RLT

```bash
# 1 参数配置：S1 配数据/基础权重；S2 配人工 human_bc_weight=50、自主成功 autonomous_bc_weight=5、Q=0.1。
cd /16T-2/sudu/code/RLinf
source .venv/bin/activate
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0,1,2,3  # 仓库导入路径与四张训练卡
S1=examples/sft/config/g1_rlt_stage1_sft_openpi_pi05.yaml
S2=examples/embodiment/config/g1_rlt_stage2_offline_ac_mlp.yaml
ENTRY=toolkits/g1_rlt_workflow.py
TOKEN=/16T-2/sudu/code/.result2/RLT/RLinf/zd_20261026/stage1/train/checkpoints/global_step_5000
RUN=/16T-2/sudu/code/.result2/RLT/RLinf/g1_actor_$(date +%Y%m%d_%H%M%S)

# 2 阶段1：本例固定复用 5000 步 RLToken；仅权重缺失时调用官方 SFT 入口。
python "$ENTRY" stage1 --stage1-config "$S1" --stage1-checkpoint "$TOKEN"

# 3 阶段2：沿用已有特征，89 条训练/5 条选优；四卡 1w critic、5k actor，每 500 步验证。
# 成功人工步模仿实录动作，自主成功步模仿 VLA；失败步无 BC，critic 保留全部数据；更换 TOKEN 须重建配套特征。
python "$ENTRY" train --config "$S2" --stage1-checkpoint "$TOKEN" --output "$RUN"
ACTOR="$RUN/best_actor_critic.pt"  # 同一文件含 actor/双 Q；日志 training.log，选优 selection.json

# 4 开启推理：默认固定 5k RLToken + state 人工50/自主5 最佳 actor，使用其保存配置。
RUN=/16T-2/sudu/code/.result2/RLT/RLinf/g1_state_human50_auto5_20261029_172727
CUDA_VISIBLE_DEVICES=3 python toolkits/g1_rlt_actor_evaluate.py --config "$RUN/config.yaml" --checkpoint "$RUN/best_actor_critic.pt" --replay "$RUN/validation_replay" --serve --port 8014
# G1 端另开终端：首次安装；在 YAML 设置服务地址、DDS 网卡、三相机序列号。
bash deploy/g1/setup_env.sh
.venv-g1/bin/python deploy/g1/client.py --config deploy/g1/client.yaml
# 自动复位后回车启动 VLA；空格切换 actor；q/Ctrl+C 退出。本机验证加 --dry-run --observation <NPZ>。
```

客户端配置与说明：[deploy/g1/README.md](deploy/g1/README.md)。
