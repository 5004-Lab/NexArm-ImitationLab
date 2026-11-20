# NexArm-ImitationLab

基于真实 NexArm Episode 导出包的 ACT 实验项目。训练部署在 GPU 服务器，采集和执行保留在机器人服务器。

**训练服务器：`wangji@10.206.58.217`**

```bash
cd /mnt/disk_2T/wangji/projects/NexArm-ImitationLab
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /usr/local/cuda/envs/nexarm-il
nexarm-il --help
bash scripts/train.sh --background --config configs/act_5hz_gpu.yaml
```

目录、后台运行、断点续训和环境说明见 [训练服务器指南](docs/TRAINING_SERVER.md)。

## 已实现

- 读取 `manifest.json`、`frames.jsonl`、JPEG 图像和阶段标注。
- 转换为 LeRobot 0.4.3 / v3 图像数据集。默认使用第三人称 RGB，可选腕部和 D405 RGB。
- 按原始 Episode 划分训练、验证和测试集。缺失动作处切断片段，ACT 动作块不会跨越片段边界。
- ACT 训练：训练集归一化、独立验证集 L1、CSV/曲线、模型与归一化处理器检查点。
- 离线预测：逐轴 MAE/RMSE、夹爪指标、推理耗时、预测对比图。
- 接入现有控制台的影子推理与低速执行，沿用控制台的路径规划和碰撞检查。
- CSV 真实实验成功率统计。

## 当前真实数据

提供的 Episode #9：194 帧、38.6 秒、5 Hz；实际轴为 **J1–J5 + gripper，6 维**。关节单位为度，夹爪为 0–1 开度。三路 RGB 和深度原文件保留在 ZIP 中；第一版不将深度输入 ACT。

**该 ZIP 中 194 帧的 `action` 和 `command_encoder` 全为空。** 因此目前能够读取和回放这条数据，但不能用它训练标准 ACT；转换程序直接提示缺少动作。没有用实测关节位置补造动作，没有训练成果或机器人成功率声明。结果标签为 `unrated`，任务是否成功需要人工标注。

下一次采集需在实际发送目标时保存 `action=[J1,J2,J3,J4,J5,gripper]`，并保留当前实测 `observation.state`。控制台已有这一导出格式；请用控制台发起目标运动并检查导出的 action 字段。

## 安装

机器人端目录：`/home/ztl/NexArm-ImitationLab`；训练端目录：`/mnt/disk_2T/wangji/projects/NexArm-ImitationLab`。两端均使用独立 `.venv`。

```sh
cd /home/ztl/NexArm-ImitationLab
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[training]'
```

ARM64 主控如果完整 LeRobot 安装因 `pynput/evdev` 失败，运行 `sh scripts/install_act_cpu.sh`。它安装 ACT 需要的库及 LeRobot 本体，不安装 GUI 遥操作依赖。脚本默认使用当前环境的 pip；可通过 `PIP_PYTHON` 指定外部 pip 的 Python。

## 数据与训练

```sh
.venv/bin/python scripts/dataset_validator.py data/raw/episode-raw.zip
# 新采集的 ZIP 必须包含真实 action；一到两条轨迹仅生成训练集，三条以上生成独立验证/测试集。
.venv/bin/python scripts/dataset_converter.py data/raw/demo-*.zip --output data/processed/screwdriver_v1
.venv/bin/python scripts/train_act.py --config configs/act_5hz.yaml
# GPU 服务器可改用 --device cuda
```

配置默认 CPU，适配当前服务器；正式训练建议有 GPU 的环境。5 Hz 配置为 chunk_size=10、n_action_steps=2、batch_size=16、20,000 步。首次预训练骨干会下载 ResNet18 权重；离线环境可把 pretrained_backbone_weights 设置为 null。

每次训练输出 `resolved_config.json`、`environment.txt`、`loss.csv`、`loss.png`，以及 `checkpoints/<step>/pretrained_model/` 中的 `config.json`、`model.safetensors`、`policy_preprocessor.json`、`policy_postprocessor.json`、`train_config.json`。`last` 指向最后检查点。`training_state.pt` 保存优化器状态，可用 `nexarm-il train --resume <pretrained_model目录> --steps <总步数>` 断点续训。

验证曲线使用推理模式的动作 L1；训练总 Loss 含 KL，两者不是同一个指标。归一化只使用训练集统计。少于三条 Episode 不生成验证/测试集，不报告泛化指标。

## 预测与执行

```sh
CKPT=outputs/act/screwdriver_rgb_seed42/checkpoints/last/pretrained_model
.venv/bin/python scripts/evaluate_nexarm.py offline --dataset data/processed/screwdriver_v1 --checkpoint "$CKPT" --output outputs/offline
# 影子推理，读取实时相机/关节，不发送动作
.venv/bin/python scripts/evaluate_nexarm.py rollout --checkpoint "$CKPT" --steps 100
# 有效模型验证后，由操作者在场启动真实低速执行
.venv/bin/python scripts/evaluate_nexarm.py rollout --checkpoint "$CKPT" --execute --steps 100
```

执行接口使用本机 `http://127.0.0.1:8765` 的控制会话，原有控制台继续处理串口、路径和碰撞检查。机器人动作受关节范围、5 度/秒和 10 度/秒²限制；夹爪受开度及变化率限制。当前控制台按规划运动接收目标，移动期间不叠加新目标，因此这是低速验证接口，不是固定 5 Hz 的直接伺服器。推理超过 200 ms 时，实际循环频率也会降低。

按 Ctrl+C 或创建项目根目录 `STOP` 文件结束；退出时发送控制台 stop。人工可在控制台停止并接管。测试结束后删除 STOP 文件再开始下一轮。

## 实验记录与测试

复制 `docs/trials.csv`，逐次记录 scenario、trial、success、completion_seconds、interventions、failure_type：

```sh
.venv/bin/python scripts/evaluate_nexarm.py trials outputs/trials.csv
.venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=tests .venv/bin/python tests/smoke_act.py
```

smoke_act 生成三条**软件测试夹具**，执行两步小型 ACT 训练、检查点保存/加载和离线预测；输出不代表真实机械臂效果。真实数据、模型和日志位于被 Git 忽略的 `data/` 与 `outputs/`。

LeRobot 接口依据 [v0.4.3 源码](https://github.com/huggingface/lerobot/tree/v0.4.3)。该版重点是 ACT；BC-MLP、Diffusion Policy 和 RGB-D 算法未实现。
