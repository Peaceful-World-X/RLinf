# AIStudio 上训练 G1 π0.5

在已登录的 AIStudio **CPU 提交容器**中运行本页命令。每个作业申请单节点 GPU，RECAP 与 STEAM 各产出一个完整的 G1 策略检查点。两者共享 π0.5 初始权重、G1 动作定义和输入归档。正式训练前先分别完成合成数据自检。

## 克隆仓库

在已登录 AIStudio 的 CPU 提交容器中执行：

```bash
git clone --branch cloud --single-branch https://github.com/Peaceful-World-X/RLinf.git
cd RLinf/cloud/g1_pi05
source cloud.env.example
export OSS_CONFIG=/input/workspace/.secrets/rlinf-oss.conf
```

`OSS_CONFIG` 是 GPU 容器挂载 NAS 后可读取的 ossutil v1 私密配置路径，可使用长期密钥或带 STS token 的临时凭据。提交时 SDK 随作业传递当前 Git 提交中的训练代码和配置。worker 使用系统 Python 引导，从私有 OSS 恢复校验过的 Linux ossutil、Python 环境、模型和分词器。训练数据由用户另行指定 OSS 归档。作业身份记录 Git commit 和资产清单摘要。修改代码后先提交到 Git，再从该提交启动新作业；续训请使用原提交及相同参数。

## 提交

```bash
# 在 CPU 检查参数、发行版和提交的文件清单
bash submit.sh recap env-only --run-id env-check-01 --dry-run

# 可选。恢复环境并检查每张卡、NCCL 和视频解码依赖
bash submit.sh recap env-only --run-id env-check-01

# 两个独立作业。各自在 worker 生成示例，价值模型和策略各更新一步
bash submit.sh recap smoke --run-id recap-smoke-01
bash submit.sh steam smoke --run-id steam-smoke-01
```

实际提交后，SDK 返回 `record_id`。在 AIStudio 作业页看状态，或检查 OSS 的 `runs/<算法>/<run-id>/result.json`。smoke 同时完成模型更新、优势标注、策略保存和 OSS 回传后才生成该文件。env-only 以 AIStudio 作业退出状态和控制台为准。

## 训练前确认的信息

- **任务与轨迹标签**。每条轨迹需要正确的任务提示（LeRobot 的 `task_index` 对应 `meta/tasks.jsonl`）。区分全成功示教与混合 rollout，统一每条轨迹的成功定义。若目标是扎带尖端首次穿出锁头，请按该终点标注或截断；训练不会从视频中自动推断这个事件。
- **机器人接口与归一化**。核对三路相机、关节和夹爪顺序、动作单位、采样 FPS。用实际训练分区计算 OpenPI 的状态和动作统计，下方列出格式。
- **价值模型和策略输入**。两种算法共用 G1 π0.5 Base、PaliGemma 分词器和 Gemma 3 270M。RECAP 的视觉骨干为 SigLIP2 SO400M/14 224，STEAM 为 SigLIP SO400M/14 384。以上文件已在发行清单中固定。策略从下列 OSS 对象恢复，worker 校验大小和 SHA-256 后，与该作业的 G1 统计一同交给 CFG 加载器。
- **实验配置**。提供训练归档的 URI、字节数和 SHA-256，明确 value 与 policy 的更新步数、全局 batch、各数据源的类型和采样权重。分别使用唯一 run-id。有留出集时按完整 episode 切分，两种策略用同一终点评测。
- **云端权限**。提交端需已登录 AIStudio，资源组需有 A100 配额，worker 需能从 NAS 读 OSS 私密配置，读取发行文件和输入，并写入该作业的输出前缀。

```text
oss://tipalgog/sudu/rlinf-cloud/models/pi05_base/7ed2fb2f91b084efc387022383035f6908aa620f6bfa0da52204a4e52e7851b6/model.safetensors
```

## 正式输入

在已有训练分区上准备 **一个** tar 或 tar.gz，根目录布局如下。每个 LeRobot 子目录含 `meta/`、`data/` 以及数据集引用的 `videos/`。两种算法传入同一个归档和摘要。RECAP 会在每个作业私有副本中写入回报和优势标签，STEAM 写入自己的优势标签。

```text
datasets.json
norm_stats.json
train/expert/meta/...
train/expert/data/...
train/rollout/meta/...
train/rollout/data/...
```

`datasets.json` 为 JSON 数组，例如

```json
[
  {"path": "train/expert", "type": "sft", "weight": 1},
  {"path": "train/rollout", "type": "rollout", "weight": 1}
]
```

`sft` 表示全成功的示教。RECAP 会强制这部分的优势标签为正。成功与失败混合的数据用 `rollout`，每条轨迹的成功标志须一致，parquet 中的 `is_success` 为标量布尔值或 0/1。STEAM 使用时序帧对，训练集应包含超过 32 帧的轨迹。归档只引用训练轨迹，在打包前按完整 episode 留出评测集。G1 相机为 `cam_left_high`、`cam_left_wrist`、`cam_right_wrist`，字段前缀为 `observation.images.`。状态和原始动作均为 16 维。提供以该训练分区计算的 OpenPI `norm_stats.json`，其 `state` 和 `actions` 含 mean、std、q01、q99；动作统计使用 14 维关节相对当前状态的差值，两个夹爪保持绝对值。

```bash
# 在正式数据已获准上传并放到 OSS 后设置。两次提交共用这组变量。
export DATA_URI=oss://tipalgog/sudu/你的训练归档.tar.gz
export DATA_SHA256="$(sha256sum /本地/训练归档.tar.gz | cut -d ' ' -f1)"
export DATA_BYTES="$(stat -c %s /本地/训练归档.tar.gz)"
# 示例预算。单位是优化器更新步，可按数据量修改。
export VALUE_STEPS=8000 POLICY_STEPS=30000 SAVE_INTERVAL=1000
export VALUE_MICRO_BATCH=2 POLICY_MICRO_BATCH=2 VALUE_BATCH=256 POLICY_BATCH=256
bash submit.sh recap train --run-id recap-g1-01 --dry-run
bash submit.sh recap train --run-id recap-g1-01
bash submit.sh steam train --run-id steam-g1-01
```

每卡 micro batch 乘 GPU 数必须整除全局 batch。每个正式训练或自检作业默认申请单节点 8 张 A100，也可显式选择 2 或 4 张。云端价值模型和策略均使用 FSDP full_shard 分摊显存。并行提交两种算法共申请 16 张卡。资源组、GPU 类型和镜像分别通过 `RESOURCE_GROUP`、`GPU_TYPE`、`AISTUDIO_IMAGE` 设置。默认镜像和资源组沿用北京 OpenPI 的 AIStudio 模板。资源紧张时可按顺序提交。提交端优先使用已有 SDK，缺失时在单独的环境安装内部源的 `aii-pypai` 与匹配的 Couler。可设置 `SUBMIT_PYTHON`、`SUBMIT_VENV` 和 `SUBMIT_INDEX`。

## 两种算法及输出

| 阶段 | RECAP | STEAM |
| --- | --- | --- |
| 价值监督 | 全轨迹回报，γ=1，失败末端奖励 −300 | k=32 的时序帧对，3 个 ensemble 成员 |
| 优势标注 | 官方 RECAP lookahead 10，positive_quantile=0.3 | 官方 ensemble，rollout_quantile=0.3，expert_quantile=0.8 |
| 策略 | 同一 CFG π0.5，3 相机，50 帧训练 chunk，16 维机器人动作 | 同左 |

输出前缀为 `oss://tipalgog/sudu/rlinf-cloud/runs/<算法>/<run-id>/`。其中 `configs/` 包含各阶段实际参数，`value/` 与 `policy/` 分开保存。每个 `global_step_N/manifest.json` 在文件传完后写入，记录每个文件的长度和 SHA-256。`policy/global_step_N/actor/model_state_dict/full_weights.pt` 是策略模型参数，`policy/g1/norm_stats.json` 是配套统计。FSDP 续训另用完整检查点目录。两种方法的价值标签定义不同，最终质量应在同一批留出轨迹或同一真机流程上比较。

每次训练更新在 worker 本地记录一行。每隔 3600 秒向同一作业前缀覆盖 `loss.csv` 和 `loss.png`。CSV 只含 `stage,step,loss`，step 为从 1 开始的该阶段训练步。PNG 分上下两个子图绘制 value 和 policy，STEAM 的 value loss 为训练器汇总的 ensemble loss。正常结束和可处理的退出信号都会补传，短作业也会生成曲线。下载与训练控制台输出留在 AIStudio。断电或强制销毁时，可从已完整上传的检查点恢复截至该步的 loss 历史。

中断后设置 `RESUME=1`，用相同的算法、run-id、归档、发行版和训练参数重提。流程校验远端清单、恢复最新完整训练步，再次生成作业私有优势标签。只复用已结束的 run-id，同一 run-id 同时只运行一个作业。修改实验设置时换新 run-id。500 GiB 本地磁盘由运行环境、模型、输入归档及其解包副本、当前检查点共用，较大数据集需调整 submit.py 中的磁盘配额。

## 与北京 OpenPI 初始化的对应关系

| 北京 OpenPI 文件 | 已成功训练的 OpenPI 流程 | RLinf 对应实现 |
| --- | --- | --- |
| init_cfg.sh | 解析提交变量和 NAS 挂载位置，区分容器本地目录与持久输入 | init_cfg.sh 固定 worker 私密配置和本地缓存位置。bootstrap.py 按清单从 OSS 下载、核对字节数和 SHA-256，再解包到本地 |
| init_env.sh | 恢复 Python 和 uv 缓存，按项目锁安装，准备 FFmpeg、验证 Torch/OpenPI | init_env.sh 载入北京安装版本的可搬迁 CPython 和已安装包快照，恢复命令入口和视频库。GPU 自检执行 NCCL all-reduce |
| k8s_job.sh | 在 GPU 容器按序初始化、启动训练、周期同步已完成检查点，并处理结束信号 | k8s_job.sh 运行对应算法的官方入口，sync_runs.py 只发布完成标记之后的检查点，按小时覆盖 loss 文件并在结束时补传 |
| submit.sh / main.py | 在 CPU 容器装 SDK，用 Kubemaker 申请 GPU，NAS 挂在 /input | submit.sh / submit.py 随作业传递 Git 提交中的训练源码和配置，GPU worker 通过 OSS 取回工具、运行环境与模型 |

## 更新发行版

在北京仓库，用训练环境的 Python 运行 `build_runtime.py OUTPUT/runtime.tar.gz`，生成运行时和包版本锁。用 `convert_base.py OFFICIAL_CHECKPOINT_DIR OUTPUT/pi05_base` 将已校验的官方 Orbax 基座转换为 G1 对应的 BF16 safetensors。`publish_assets.py --inputs <本地资产路径的JSON> --release <新版本> --output <仓库外目录> --upload` 发布白名单中的资产和不可变清单。输入键为 pi05、gemma、recap_vision、steam_vision（本地模型目录），以及 runtime、runtime_lock、tokenizer、base_provenance、ossutil（本地文件）。将新生成的 `manifest.json` 放入 `cloud/g1_pi05/assets/`，随训练代码一起提交到 Git。
