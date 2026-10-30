# G1 双臂客户端

用本目录连接 RLinf 服务和 G1 双臂/Dex1：启动后复位，回车开始 VLA，空格开关 actor。先准备环境、启动服务，再在机器人侧运行客户端。客户端代码不依赖旁边的 OpenPI 仓库。

## 环境

在机器人侧的 RLinf 根目录执行；需要 Python 3.11、uv、git、cmake 和 g++。脚本按固定版本安装到 `.venv-g1`，不改训练环境。SDK 与 CycloneDDS 位于该环境的 `vendor/`，运行时需要保留。

```bash
bash deploy/g1/setup_env.sh
source .venv-g1/bin/activate
```

## 服务端

在 GPU 服务器的 RLinf 根目录执行。默认复用固定 RLToken 5000 步及 state 人工 BC=50、自主 BC=5 的最佳 actor（critic 8000 / actor 4000），读取该实验保存的配置。

```bash
source .venv/bin/activate
RUN=/16T-2/sudu/code/.result2/RLT/RLinf/g1_state_human50_auto5_20261029_172727
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=3 python toolkits/g1_rlt_actor_evaluate.py \
  --config "$RUN/config.yaml" --checkpoint "$RUN/best_actor_critic.pt" \
  --replay "$RUN/validation_replay" --serve --device cuda:0 --port 8014
```

## 客户端

先在 [client.yaml](client.yaml) 修改服务地址、DDS 网卡和三相机序列号；复位姿态、速度、prompt 和异步参数也集中在此文件。复位目标已经写入配置，部署不需要数据集。启动时先以实测位置保持并预热服务，再平滑复位。

```bash
source .venv-g1/bin/activate
python deploy/g1/client.py --config deploy/g1/client.yaml
# 本机验收使用模拟硬件；NPZ 含 state、cam_high、cam_left_wrist、cam_right_wrist。
python deploy/g1/client.py --dry-run --observation /path/to/observation.npz
```

| 按键 | 行为 |
|---|---|
| 回车 | 复位到位后启动 VLA；运行中重复按无效；异常处理后恢复，actor 关闭 |
| 空格 | 运行时开关 actor，保持最后目标、清空队列并使旧请求失效；复位和等待期间不启用 |
| q / Ctrl+C | 清空队列、保持姿态、释放双臂控制权并恢复终端 |

默认只通过 `rt/arm_sdk` 接管双臂，身体/腿部交给原控制器。双臂按绝对 rad 执行；夹爪输出裁剪到 0～100，再线性映射到 0～5.4 rad。关节越界、非有限值、断线或观测过期会暂停，处理后按回车恢复。复位超时也不会开始推理。

## 动作与异步协议

客户端每次传三张 RGB 图像、物理 `state[16]`、`prompt`、布尔 `rlt_switch`，检查服务返回的 `actions[10,16]`、`q_values[2]` 和开关回执。16 维顺序为左臂 7、右臂 7、左夹爪、右夹爪；每臂为 ShoulderPitch/Roll/Yaw、Elbow、WristRoll/Pitch/Yaw。客户端直接执行服务动作，不再加 state 或 VLA 修正量。

动作时钟 30 Hz、双臂发布 250 Hz、夹爪发布 200 Hz，双臂默认限速 1 rad/s。默认开启客户端异步预取：剩 9 步发起唯一后台请求，按期间已执行步数对齐，同模式最多衔接 2 个关节动作，夹爪不融合。服务声明 `client_async_supported=true`、`rtc_prefix_supported=false`；这不包含模型内部 RTC 前缀补全。

队列耗尽时保持实测姿态，丢弃迟到结果后用新观测重启，不补跑动作。请求耗时上限由 `rtc.max_response_age_s` 控制，网络超时由 `server.timeout_s` 控制；相机和网络都在后台。当前机器预热推理约 0.22～0.26 秒，10 步队列在持续预取时会出现耗尽保持，不能据此宣称持续无停顿 30 Hz。详见 [本机验证记录](VALIDATION.md)；真机复位和执行仍需现场验证。

URDF、关节增益、消息和重力补偿方法来自所提供的 xr_teleoperate；归属见 [assets](assets/source.json)。其附带旧版 SDK 缺少 G1/Dex1 消息，因此环境脚本固定了官方 SDK 提交。传输与预取参考 OpenPI，必要代码已放入本目录。
