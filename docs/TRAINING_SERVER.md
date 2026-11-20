# GPU 训练服务器

日常训练服务器：`wangji@10.206.58.217`，RTX 5090，Python 3.12，独立 Conda 环境 `/usr/local/cuda/envs/nexarm-il`。

| 内容 | 路径 |
| --- | --- |
| 工程 | `/mnt/disk_2T/wangji/projects/NexArm-ImitationLab` |
| 原始和转换数据 | `/mnt/disk_2T/wangji/datasets/nexarm` |
| 训练输出 | `/mnt/disk_2T/wangji/experiments/nexarm` |
| 模型下载缓存 | `/mnt/disk_2T/wangji/cache/nexarm/huggingface` |

工程内 `data`、`outputs` 是上述目录的软链接。机器人控制和采集保留在 `ztl@10.206.95.245`。

## 常用命令

```bash
cd /mnt/disk_2T/wangji/projects/NexArm-ImitationLab
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /usr/local/cuda/envs/nexarm-il
nexarm-il --help
nexarm-il info data/raw/episode-raw.zip
nexarm-il convert data/raw/demo-*.zip --output data/processed/screwdriver_v1
bash scripts/train.sh --config configs/act_5hz_gpu.yaml
```

后台训练并指定本次运行目录：

```bash
bash scripts/train.sh --background --config configs/act_5hz_gpu.yaml --output outputs/act/run01
# 启动器打印日志路径和 PID；用 tail -f 查看日志。
```

断点续训：

```bash
bash scripts/train.sh --background \
  --resume outputs/act/run01/checkpoints/last/pretrained_model --steps 40000
```

`--steps` 是目标总步数。恢复模型、归一化处理器、优化器和随机数状态，继续追加 loss.csv。数据加载器从新一轮迭代开始，不保证与未中断训练逐步相同。

离线评估：

```bash
nexarm-il evaluate --dataset data/processed/screwdriver_v1 \
  --checkpoint outputs/act/run01/checkpoints/last/pretrained_model \
  --output outputs/act/run01/test --device cuda
```

## 环境与检查

使用 PyTorch 2.7.1 + CUDA 12.8、torchvision 0.22.1、LeRobot 0.4.3。安装脚本：`bash scripts/install_gpu.sh`。系统驱动提供 CUDA 兼容支持。运行库随 PyTorch 安装在指定 Conda 环境中；本项目不编译 CUDA 扩展，不另外安装 CUDA Toolkit，也不修改其他 Conda 环境。

ACT 安装只包含本项目调用的模块；LeRobot GUI 遥操作依赖不用于训练。`requirements/act.txt` 为共同依赖，`requirements/gpu-server.txt` 为本机实际版本。GPU 安装脚本优先安装 CUDA 12.8 wheel，避免误装其他版本。

```bash
make test
make smoke
```

`make smoke` 用明确标记的软件测试夹具执行 CUDA 训练、保存、续训和预测，输出位于 `outputs/smoke-*`。真实数据 ZIP 缺少动作标签，因此尚未发起真实任务训练。

数据迁移可直接将新 ZIP 上传至训练服务器的 `datasets/nexarm/raw/`。只对选定的示教 ZIP 执行 convert。模型训练后，将 `pretrained_model` 整个目录（含归一化处理器）拷回机器人服务器，再运行 shadow/rollout。

当前工程保留 Git 历史及 GitHub origin。新服务器尚未配置 GitHub SSH 密钥，首次迁移及本次提交通过原机器人服务器的已授权连接同步；日常训练不依赖 GitHub 在线连接。
