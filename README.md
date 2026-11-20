# NexArm-ImitationLab

## GPU 服务器直接操作

完整可运行命令见 [GPU 操作与真机 ACT 联调](docs/GPU_COMMANDS.md)。

```bash
source /home/wangji/miniconda3/etc/profile.d/conda.sh
conda activate /usr/local/cuda/envs/nexarm-il
cd /mnt/disk_2T/wangji/projects/NexArm-ImitationLab
python scripts/gpu_robot.py --status
python scripts/gpu_robot.py --watch
python scripts/gpu_robot.py --shadow --cycles 20
```

真机启动：`python scripts/gpu_robot.py --execute --cycles 8`。停止：`python scripts/gpu_robot.py --stop`。
执行命令会真实运动；推理使用 GPU，机器人端处理相机、串口和路径检查。当前是单步约 3°、总行程 6° 的联调，不代表已完成抓取验证。


NexArm 数据转换、ACT / BC-MLP / Diffusion Policy 训练与离线推理工程。训练在 GPU 服务器执行，采集与真实机械臂控制留在机器人服务器。

## 常用命令速查（当前服务器可直接运行）

以下为 **Linux Bash** 命令，使用服务器上已经存在的数据和 1,000 步检查点。先登录并初始化当前终端，再按需复制对应代码块。新训练、转换和推理输出都使用时间戳目录，可重复执行。

### 登录与初始化

在自己电脑的终端登录：

```bash
ssh wangji@10.206.58.217
```

登录服务器后，每个新终端执行一次：

```bash
source /home/wangji/miniconda3/etc/profile.d/conda.sh
conda activate /usr/local/cuda/envs/nexarm-il
cd /mnt/disk_2T/wangji/projects/NexArm-ImitationLab
source scripts/env.sh
```

### 查看命令和 GPU

```bash
nexarm-il --help
nexarm-il train --help
nexarm-il evaluate --help
nvidia-smi
python -c 'import sys, torch; print("Python:", sys.executable); print("PyTorch:", torch.__version__); print("GPU:", torch.cuda.get_device_name(0))'
```

### 查看自采 ZIP 和已有数据划分

```bash
nexarm-il info data/raw/episode-raw.zip
python - <<'PY'
import json
from pathlib import Path
info = json.loads(Path('data/processed/public_pick_5hz/dataset.json').read_text())
print('FPS:', info['fps'])
print('Axes:', info['axes'])
print('Units:', info['units'])
for split, episodes in info['splits'].items():
    print(split, 'episodes:', len(episodes), 'IDs:', episodes)
PY
```

### 重新转换已下载的公共数据

源文件已在服务器上，无需再次下载。此命令转换到一个新目录，并打印其绝对路径；日常训练可继续使用已经验证过的 `data/processed/public_pick_5hz`。

```bash
DATASET_DIR="data/processed/public_pick_5hz_$(date +%Y%m%d_%H%M%S)_$$"
nexarm-il import-public \
  --source data/public/nexarm_pick_20ep \
  --repo-id dev-lithion/nexarm_pick_20ep \
  --fps 5 --cameras wrist --output "$DATASET_DIR"
realpath "$DATASET_DIR"
```

对已经验证过的转换结果进行源动作、状态与 RGB 对照：

```bash
python scripts/check_public_conversion.py
```

### 最快体验：用已有 BC 模型做离线推理

无需重新训练。使用现成的 1,000 步 BC-MLP 检查点，对 150 帧测试数据做预测；输出指标、预测数组与动作对比图。

```bash
EVAL_DIR="outputs/manual_eval/bc_mlp_$(date +%Y%m%d_%H%M%S)_$$"
nexarm-il evaluate \
  --dataset data/processed/public_pick_5hz \
  --checkpoint outputs/public_pick_1000/bc_mlp/checkpoints/1000/pretrained_model \
  --split test --device cuda --output "$EVAL_DIR"
cat "$EVAL_DIR/metrics.json"
realpath "$EVAL_DIR/actions.png"
```

### 用三个已有模型分别推理

```bash
EVAL_ROOT="outputs/manual_eval/all_models_$(date +%Y%m%d_%H%M%S)_$$"
for MODEL in act bc_mlp diffusion; do
  nexarm-il evaluate \
    --dataset data/processed/public_pick_5hz \
    --checkpoint "outputs/public_pick_1000/$MODEL/checkpoints/1000/pretrained_model" \
    --split test --device cuda --output "$EVAL_ROOT/$MODEL"
done
realpath "$EVAL_ROOT"
```

### 单独训练 ACT / BC-MLP / Diffusion

以下三个代码块可以任选一个运行，均为 1,000 步前台训练；结束时打印检查点路径。Ctrl+C 可以中断前台训练，恢复时使用最近已保存的检查点。

ACT：

```bash
RUN_DIR="outputs/manual_train/act_$(date +%Y%m%d_%H%M%S)_$$"
nexarm-il train --config configs/policy/act.yaml \
  --dataset data/processed/public_pick_5hz --device cuda --steps 1000 --output "$RUN_DIR"
```

BC-MLP：

```bash
RUN_DIR="outputs/manual_train/bc_mlp_$(date +%Y%m%d_%H%M%S)_$$"
nexarm-il train --config configs/policy/bc_mlp.yaml \
  --dataset data/processed/public_pick_5hz --device cuda --steps 1000 --output "$RUN_DIR"
```

Diffusion Policy：

```bash
RUN_DIR="outputs/manual_train/diffusion_$(date +%Y%m%d_%H%M%S)_$$"
nexarm-il train --config configs/policy/diffusion.yaml \
  --dataset data/processed/public_pick_5hz --device cuda --steps 1000 --output "$RUN_DIR"
```

### 后台训练并查看日志

退出 SSH 后继续训练。后台启动器打印 PID 和日志位置；不要仅为了查看日志而再次启动训练。

```bash
RUN_DIR="outputs/manual_train/act_background_$(date +%Y%m%d_%H%M%S)_$$"
bash scripts/train.sh --background --config configs/policy/act.yaml \
  --dataset data/processed/public_pick_5hz --device cuda --steps 1000 --output "$RUN_DIR"
```

查看最近一个单模型后台启动任务的日志：

```bash
LOG_FILE=$(find outputs/launches -maxdepth 1 -name '*.log' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
echo "$LOG_FILE"
tail -n 50 "$LOG_FILE"
```

持续看日志；Ctrl+C 只退出日志查看，不停止训练：

```bash
LOG_FILE=$(find outputs/launches -maxdepth 1 -name '*.log' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
tail -f "$LOG_FILE"
```

查看训练进程和 GPU：

```bash
ps -u "$USER" -o pid,etime,args | grep -E '[i]mitation_lab train|[c]ompare_models.py'
nvidia-smi
```

### 直接体验断点续训

为避免修改原有基准结果，先在新目录训练 BC-MLP 2 步，再读取其检查点续训到总计 4 步。下面整个代码块可以直接运行。

```bash
RESUME_DIR="outputs/resume_demo/bc_mlp_$(date +%Y%m%d_%H%M%S)_$$"
nexarm-il train --config configs/policy/bc_mlp.yaml --device cuda --steps 2 --output "$RESUME_DIR"
nexarm-il train --resume "$RESUME_DIR/checkpoints/2/pretrained_model" --device cuda --steps 4
cat "$RESUME_DIR/loss.csv"
```

正式续训同样使用 `--resume`，`--steps` 填目标总步数。务必等上一次训练退出后再对同一实验目录续训。

### 三模型快速跑通 / 重新训练对比

每个模型 2 步，用于检查转换数据、模型保存和推理流程：

```bash
python scripts/compare_models.py --steps 2 \
  --output "outputs/compare_smoke_$(date +%Y%m%d_%H%M%S)_$$"
```

每个模型 1,000 步，依次训练并在同一测试集评价：

```bash
python scripts/compare_models.py --steps 1000 \
  --output "outputs/compare_1000_$(date +%Y%m%d_%H%M%S)_$$"
```

只对比 ACT 与 BC-MLP：

```bash
python scripts/compare_models.py --models act bc_mlp --steps 1000 \
  --output "outputs/compare_act_bc_$(date +%Y%m%d_%H%M%S)_$$"
```

### 查看已有对比结果和检查点

```bash
python - <<'PY'
import json
from pathlib import Path
results = json.loads(Path('outputs/public_pick_1000/comparison.json').read_text())
print('model\tjoint_RMSE_native\tgripper_MAE_native\tmean_latency_ms')
for name, values in results.items():
    print(name, round(values['joint_rmse_source_units'], 3),
          round(values['gripper_mae_source_units'], 3),
          round(values['latency_ms']['mean'], 3), sep='\t')
PY
ls outputs/public_pick_1000/bc_mlp/checkpoints/1000/pretrained_model
realpath outputs/public_pick_1000/bc_mlp/test/actions.png
```

### 运行测试与代码检查

```bash
make test
make lint
```

ACT 的软件夹具训练、保存、恢复与预测测试：

```bash
make smoke
```

以上命令使用公开数据进行离线训练/推理，不驱动机械臂。本机自采 ZIP 的 `action` 目前为空；真实数据采集完成后的转换和机器人操作见后续章节。

现成模型、螺丝刀数据集的硬件差异及本次真机小步结果见 [公开模型与真机状态](docs/PRETRAINED_OPTIONS.md)。

## 实时预测演示（已部署）

机器人服务器上的只读页面：**[NexArm 实时预测](http://10.206.95.245:8788)**。电脑需能访问机器人服务器所在网络。

当前部署的是公开数据训练的 BC-MLP：机器人端 `data/models/public_pick/bc_mlp`。模型读取当前腕部 RGB 和原生编码器反馈；页面显示现场第三人称图像、实际模型输入、当前编码器值、预测目标及差值。预测输出不会发送给机械臂。本次启动约运行 1 小时，暂停后页面保留最后一帧并提示已暂停。

在机器人服务器可单独执行 100 帧实时影子推理：

```bash
cd /home/ztl/NexArm-ImitationLab
.venv/bin/python -m imitation_lab rollout \
  --checkpoint data/models/public_pick/bc_mlp --device cpu --steps 100 \
  --preview-dir outputs/live_preview --output outputs/live_preview/predictions.jsonl
```

页面由该服务器的静态 HTTP 服务提供；推理文件在 `outputs/live_preview/`，推理进程日志是 `outputs/live-preview.log`。当前演示停止命令只停止预测，不涉及机器人运动：

```bash
touch /home/ztl/NexArm-ImitationLab/outputs/live_preview/STOP
```

公开数据任务名只有 `nexarm pick`，未说明“桌面螺丝刀放进绿色罐子”的成功条件。单张腕部图像能看到细长物体，但不能据此确认完整任务。现有模型是下一个动作预测器，不是物体识别/自动规划器；本次演示用于观察它面对当前桌面时的预测，尚不能据此判断能抓取螺丝刀。

## 服务器与环境

| 内容 | 位置 |
| --- | --- |
| GPU 服务器 | `wangji@10.206.58.217`，RTX 5090 D v2 |
| 工程 | `/mnt/disk_2T/wangji/projects/NexArm-ImitationLab` |
| Conda 环境 | `/usr/local/cuda/envs/nexarm-il` |
| 数据 | `/mnt/disk_2T/wangji/datasets/nexarm`，工程内软链接 `data` |
| 实验输出 | `/mnt/disk_2T/wangji/experiments/nexarm`，工程内软链接 `outputs` |
| 机器人服务器 | `ztl@10.206.95.245`，工程 `/home/ztl/NexArm-ImitationLab` |

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /usr/local/cuda/envs/nexarm-il
cd /mnt/disk_2T/wangji/projects/NexArm-ImitationLab
nexarm-il --help
```

环境：Python 3.12、PyTorch 2.7.1+cu128、torchvision 0.22.1、LeRobot 0.4.3。新机器先复制 `configs/server.env.example` 为 `.env`，再运行 `bash scripts/install_gpu.sh`。已有服务器无需重新安装。`environment.yml` 管理 Conda 基础环境，`requirements/act.txt` 管理共同训练依赖，`requirements/gpu-server.txt` 记录已安装版本。PyTorch 运行库在 Conda 环境内；本项目不编译 CUDA 扩展，不额外安装系统 CUDA Toolkit。

## 从跑通到训练、推理

已下载的小数据集：[`dev-lithion/nexarm_pick_20ep`](https://huggingface.co/datasets/dev-lithion/nexarm_pick_20ep)，20 个 Episode、6,000 帧、30 Hz、单腕部 RGB，约 57 MB。`robot_type=nexarm_follower`，状态和动作为 6 维。

公共源数据位于 `data/public/nexarm_pick_20ep`。首次下载到其他机器：

```bash
python scripts/download_public_dataset.py --repo-id dev-lithion/nexarm_pick_20ep --output data/public/nexarm_pick_20ep
```

Hugging Face 网络受限时可在联网机器下载整个目录后上传到上述服务器路径；训练不依赖在线下载。该数据未提供 README/明确许可证，不在本 Git 仓库重新分发。另有明确 Apache-2.0 标注的 [`thanhkt/nexarm_candy_oracle_20eps`](https://huggingface.co/datasets/thanhkt/nexarm_candy_oracle_20eps)，实际 22 个 Episode、7,260 帧、15 Hz、前视与腕部 RGB，可使用同一导入器。

### 1. 转换公共数据

```bash
nexarm-il import-public \
  --source data/public/nexarm_pick_20ep \
  --repo-id dev-lithion/nexarm_pick_20ep \
  --output data/processed/public_pick_5hz --fps 5 --cameras wrist
python scripts/check_public_conversion.py
```

已部署服务器上的转换结果可直接使用，无需重复转换。30→5 Hz 选择每第 6 个源帧，不复制图像。输出为 14/3/3 条训练/验证/测试轨迹，共 1,000 帧。每份数据重新计算自己的统计信息，训练只使用训练集统计做归一化。

公共数据的原生关节值接近编码器位置尺度，不能当作角度；夹爪也不能直接当作 0–1。导入时保持状态和动作值，保留原生轴名、采样频率、Episode、源帧号和时间戳，单位标为 `source_native`。不会仅因“6 维”就改成当前控制台坐标。原始记录有极端编码器值，适合验证流程，尚不足以证明可靠的真实抓取能力。源数据标定、场景与本机不同；公共数据检查点用于离线预测，映射与标定完成前不用于控制台 rollout。

### 2. 先用短训练验证三个模型

```bash
source scripts/env.sh
python scripts/compare_models.py --steps 2 --output "outputs/public_pick_smoke_$(date +%Y%m%d_%H%M%S)_$$"
```

这会依次训练 ACT、BC-MLP、Diffusion Policy，保存并重新加载检查点，在独立测试集逐帧推理，生成每个模型的 `test/metrics.json`、`predictions.npz`、`actions.png`，以及汇总 `comparison.json`。

### 3. 小数据训练与对比

```bash
python scripts/compare_models.py --steps 1000 --output "outputs/public_pick_1000_$(date +%Y%m%d_%H%M%S)_$$"
# 单独训练其中一个模型
nexarm-il train --config configs/policy/bc_mlp.yaml --output "outputs/my_bc_$(date +%Y%m%d_%H%M%S)_$$"
nexarm-il train --config configs/policy/diffusion.yaml --output "outputs/my_diffusion_$(date +%Y%m%d_%H%M%S)_$$"
nexarm-il train --config configs/policy/act.yaml --output "outputs/my_act_$(date +%Y%m%d_%H%M%S)_$$"
```

同一批数据、同一分组划分、相同图像尺寸、相同种子与训练步数。此公开小数据配置使用 96×96 RGB、batch size 8，骨干不预加载 ImageNet 权重，避免下载条件影响首次验证。1000 步属于流程与初始拟合验证，不等同于充分收敛或论文级对比。

| 模型 | 实现 | 输入/预测 |
| --- | --- | --- |
| ACT | LeRobot ACT，ResNet18 + VAE + Transformer | RGB + 状态，16 步动作块，执行 2 步 |
| BC-MLP | ResNet18 图像编码 + 状态拼接 + 两层 MLP | RGB + 状态，单步回归 |
| Diffusion Policy | LeRobot 视觉编码 + 条件时序 U-Net，DDIM | RGB + 状态，16 步 horizon，执行 2 步，10 次去噪 |

BC-MLP 也接收 RGB，避免与视觉 ACT 比较时缺失观测信息。三种模型均支持检查点、归一化处理器和统一 `select_action` 接口。训练总 Loss 分别含 VAE KL、回归或噪声预测项，不能横向比较其数值。验证 `val_l1` 为当前动作的归一化误差；不同归一化策略下仍以测试集原始单位 MAE/RMSE 为跨模型比较依据。

已完成上述三个模型各 1,000 步训练和独立测试集推理，实际数值、产物位置与复现方式见 [本次对比结果](docs/PUBLIC_BASELINES.md)。

### 4. 后台训练与断点恢复

```bash
bash scripts/train.sh --background --config configs/policy/act.yaml --output outputs/act/run01
# 启动器打印 PID 和日志路径；tail -f <日志路径> 查看进度
bash scripts/train.sh --background \
  --resume outputs/act/run01/checkpoints/last/pretrained_model --steps 20000
```

`--steps` 是训练目标总步数。恢复模型、归一化、优化器与随机数状态；数据加载器重新开始一轮迭代，不承诺与未中断运行逐步相同。

### 5. 独立离线推理

```bash
nexarm-il evaluate --dataset data/processed/public_pick_5hz \
  --checkpoint outputs/public_pick_1000/act/checkpoints/last/pretrained_model \
  --output outputs/public_pick_1000/act/test --device cuda
# checkpoint 路径换为 bc_mlp 或 diffusion 即可对比推理
```

测试按时间顺序调用策略，Episode 边界清空动作队列，ACT 与 Diffusion 按各自 `n_action_steps` 重新预测。输出逐轴 MAE、RMSE、关节总体 RMSE、夹爪 MAE、推理平均耗时和 P95。公共数据不冒充 degree/open_fraction，也不报告无定义的夹爪开闭准确率。

## 转换本机真实 NexArm 数据

### 实际数据定义

当前控制台的真实导出是 **5 个转动关节 + 夹爪，共 6 维**，不是建议文本中假定的 7 维。顺序固定 `J1,J2,J3,J4,J5,gripper`：前五个轴为 degree，夹爪为 open_fraction。相机键保持 `third / wrist / front`，其中 `third` 为第三人称，`wrist` 为腕部，`front` 为 D405 RGB。第一版不把原始深度当作普通 RGB。

| 字段 | 含义 |
| --- | --- |
| `observation.state` | 同一时刻实测 5 关节与夹爪 |
| `action` | 采集时发送的目标关节角与夹爪指令 |
| `timestamp` | Episode 起点后的采样秒数 |
| `frame_index` | 原始采样帧号 |
| `sync_ms` / `source_timestamps` | 图像与反馈来源时间 |
| `segments.jsonl` | 阶段或排除片段 |

```bash
nexarm-il info data/raw/episode-raw.zip
nexarm-il convert data/raw/successful-demo-*.zip \
  --output data/processed/screwdriver_v1 --cameras third --seed 42
# 多视角
nexarm-il convert data/raw/successful-demo-*.zip \
  --output data/processed/screwdriver_multiview --cameras third wrist
```

只把选定的成功示教 ZIP 传给第一轮转换；失败轨迹保留用于失败分析。转换保留 result，不自动把未评价数据标成成功。提供的 Episode #9 有 194 帧、38.6 秒、5 Hz，但所有 `action`/`command_encoder` 均为空；无法用它生成标准 ACT 监督数据，程序会直接提示缺少标签。不会复制实测关节位置或未来状态来冒充原指令。

### 转换规则

1. 状态/动作按当前导出的轴序和单位读取，检查维度、有限值和夹爪范围。
2. 按原帧号和 5 Hz 时间戳识别连续片段；缺失动作、采样间断及人工 exclude 片段均切断动作块。
3. 有同步字段时，反馈和所选相机必须齐全，默认源样本年龄不超过 100 ms、观测源时间跨度不超过 45 ms，重复源帧不进入训练片段。
4. RGB 用 Pillow 解码为 RGB，所有选择视角统一尺寸；原始 ZIP 和原生深度保持不变。
5. **先按原始 Episode 划分，再把连续片段保存为 LeRobot Episode。** 同一次示教的不同片段不会分散到训练和测试中。
6. 默认 70/15/15；50 条时为 35/7/8。少于 3 条仅输出训练集，不声称有独立泛化评价。
7. 输出为官方 LeRobot v3 图像数据集，各 split 包含 `meta/`、`data/`；图像嵌入 parquet，不要求存在 `videos/`。`dataset.json` 保留来源 Episode、原帧号、时间戳与分组。
8. ACT/Diffusion 读取未来动作时使用 LeRobot 的 Episode 末尾 padding mask，动作块不跨片段边界。

公共 LeRobot 导入器使用源视频分块编号和每个 Episode 的 `from_timestamp` 定位 JPEG/RGB，顺序解码视频，只选择真实存在的帧。`check_public_conversion.py` 对 train/val/test 各抽取源样本，逐值对照状态/动作，并逐像素比较对应的 RGB 图像。

### 本机 5 Hz ACT 配置

`configs/act_5hz_gpu.yaml`：第三人称 RGB + 6 维状态/动作，chunk size 10（2 秒）、n_action_steps 2（0.4 秒）、ResNet18、AdamW、学习率 1e-5、KL weight 10、batch size 16、20,000 步、每 5,000 步保存一次、seed 42。

```bash
bash scripts/train.sh --config configs/act_5hz_gpu.yaml
```

这些是初始参数，不是已验证的 NexArm 最优值。以后稳定采集 10 Hz 以上数据时单独训练对应版本，不用重复 5 Hz 图像伪造高频数据。

## 训练产物

每次运行保留实际解析的配置、数据说明、环境版本、`loss.csv`、`loss.png`。检查点位于：

```text
outputs/<run>/checkpoints/<step>/
  pretrained_model/
    config.json
    model.safetensors
    policy_preprocessor.json
    policy_postprocessor.json
    train_config.json
    dataset_spec.json
  training_state.pt
outputs/<run>/checkpoints/last -> 最后步数
```

归一化处理器还会保存对应的 safetensors 文件，搬运模型时复制整个 `pretrained_model` 目录，不只复制权重。

## 机器人验证与正式实验

先将本机坐标下训练的检查点复制回机器人服务器，运行影子模式。公共数据的 native 编码器检查点须先完成轴序/零位/方向/夹爪标定和转换，不能直接发送到当前控制台。

```bash
CKPT=outputs/act/run01/checkpoints/last/pretrained_model
nexarm-il rollout --checkpoint "$CKPT" --steps 100
# 操作者在场、影子结果已核对后启动低速真实执行
nexarm-il rollout --checkpoint "$CKPT" --steps 100 --execute
```

机器人接口沿用现有控制台规划和碰撞检查，含软限位、速度/加速度及夹爪范围限制。按 Ctrl+C 或创建 `STOP` 文件停止；退出发送控制台 stop。规划运动期间不叠加目标，因此低速执行实际频率由控制台运动与推理耗时决定，不能当作严格 5 Hz 直接伺服。

第一批建议采集 50 条成功轨迹：螺丝刀在 5 个初始位置各 10 条，固定工具、罐子、相机和初始姿态。随后扩展到 100–200 条以上。正式实验分别测试固定位置、未见位置、不同方向、不同光照，各 30 次；提前定义成功为夹持螺丝刀、放入罐中并松开后保持在罐内、全程无人工干预。

在 `docs/trials.csv` 记录 success、completion_seconds、interventions、failure_type：

```bash
nexarm-il trials outputs/trials.csv
```

成功率为 successes / trials；同时记录平均完成时间、跟踪误差和失败类型。离线动作误差不等于真实任务成功率。本次只做公开数据训练与离线推理，没有发起真实机械臂执行。

## 工程结构与验证

```text
imitation_lab/
  raw.py, convert.py, public_data.py   数据读取与转换
  models.py                          ACT / BC-MLP / Diffusion 统一接口
  train.py, evaluate.py               训练、恢复、离线预测与指标
  robot.py, cli.py                    控制台适配与命令入口
configs/policy/                       三个对比模型配置
scripts/compare_models.py             共用划分的训练/推理对比
scripts/check_public_conversion.py    源数值、视频帧与转换结果对应检查
tests/                               功能及转换集成测试
```

```bash
make test
make lint
make smoke
```

`make smoke` 使用明确标记的软件夹具测试 ACT 训练/续训；公开数据的三模型测试用前文 `compare_models.py --steps 2`。两者分开，不把软件夹具当作真实机器人实验。GitHub Actions 执行基础测试；安装 LeRobot 后服务器还会执行数据转换集成测试。

已实现：ACT、BC-MLP、Diffusion、单/多视角导入、独立划分、离线评估。后续研究：RGB-D 编码、跟踪误差增强、数据量消融、真实机器人泛化与失败纠正。VLA 与自定义 Transformer 不属于当前第一阶段。

接口依据 [LeRobot v0.4.3](https://github.com/huggingface/lerobot/tree/v0.4.3)，硬件坐标语义参考 [Hiwonder NexArm follower](https://github.com/Hiwonder-official/lerobot-nexarm/blob/main/src/lerobot/robots/nexarm_follower/nexarm_follower.py)。
