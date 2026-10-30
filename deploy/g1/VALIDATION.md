# G1 客户端本机验证

2026-10-30 在本机 RTX 4090（物理 GPU 3）完成模拟硬件和真实模型服务联调。没有创建物理 DDS 控制发布者；这验证了客户端流程与协议，不代表真机执行效果。

| 项目 | 结果 |
|---|---|
| 自动测试 | 23 项通过：非零实测位置启动、限速复位到位/超时、按键、快速切换、迟到响应、耗尽保持、对齐/关节衔接、夹爪不衔接、超时/断线/错回执/非有限值/越界、观测过期、RGB 顺序、退出释放 |
| 真实权重 | 固定 RLToken 5000 步；state human50/auto5 最佳 actor，critic 8000 / actor 4000；服务器严格加载并检查特征契约 |
| 录制观测 | `zd89_20261027/source_cache/dataset` 的 episode 000 / frame 0；三个 RGB 相机和原始物理 state；复位配置与该 state 逐项完全相等 |
| WebSocket | 实际 OpenPI 服务与独立客户端互通；3 次 actor=false、3 次 true，均返回有限 `actions[10,16]`、`q_values[2]`，开关回执正确 |
| 交互联调 | `.venv-g1` 中真实启动 `client.py --dry-run`：复位、等待回车、重复回车、空间键开关/连续快切、q 与 SIGINT 退出；终端恢复，退出码 0 |
| 启动预热 | 冷启动首请求 2724 ms；客户端在取得控制权之前预热，预热输出不执行 |
| 热请求延迟 | VLA 261、262 ms；actor 226、255、222 ms。仅包含本机网络、编解码和服务推理；不包含真机相机/DDS 延迟 |
| 夹爪裁剪 | 初始帧 VLA 最低 −0.619；按用户确认裁剪到 [0,100]。关节不裁剪、不做重复坐标转换或修正叠加 |
| 异步边界 | 10 步 / 30 Hz 配合返回时的时间对齐，连续预取出现队列耗尽；日志确认保持后重新获取观测，迟到动作未补跑。未达到持续无停顿 30 Hz |
| 重力模型 | Pinocchio 2.7.0 实际加载归档 URDF；约束后 nq=14；复位姿态的 `computeGeneralizedGravity` 与静态 RNEA 一致且有限 |
| 实际 SDK | 固定 Unitree SDK `814556d15970dd2ecf1c9984e845ca02ab07e206`；真实 35 电机 LowCmd、Dex1 消息和 CRC 导入/构造通过，未建立 DDS 通道 |
| 环境 | `setup_env.sh` 实跑通过；Python 3.11.16、NumPy 1.26.4、websockets 15.0.1、CycloneDDS 0.10.2、pyrealsense2 2.59.1.11053；训练环境未改动 |

权重身份：Stage 1 SHA256 `8428ee9ecffb41383a8150a15d096f66d3199af6c06b92050b3fe1472aad2182`；actor SHA256 `17821fff8ce41320b9d12b8b6fa0cfb8ac511373ecea20076dc20317347e5b09`。启动路径见 [README](README.md)。本次未重新训练或修改模型。

可在 RLinf 根目录复跑组件测试；模拟 DDS/相机仅用于测试，生产入口不会自动回退到模拟硬件。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest tests/unit_tests/test_g1_client.py -q -p no:cacheprovider
```

Ruff、shell 语法和差异空白检查通过。中英文 Sphinx 构建均为 0 warning，RST 标记检查通过；全仓库符号检查仍报告 26 项既有未定义示例名，均位于本次未修改的 Sphinx 页面。本次 README/Guide 命令已对照实际入口核查。

尚待现场验证：相机序列号与视角、DDS 网卡、实机状态更新、关节增益/重力补偿效果、复位跟踪和任务执行表现。
