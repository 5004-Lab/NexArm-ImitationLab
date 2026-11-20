# GPU 服务器命令与真机 ACT 联调

**当前部署：只读状态接口已恢复，GPU 的 `--watch` 已实测可用，显示 `read_only=True`。完整运动服务启动被自动审批拦截；当前不能发送动作。此前 8 轮桥接测试完成 6 轮后，J3 无进展停机，完整连续验收尚未通过。**

## 登录并激活环境

```bash
ssh wangji@10.206.58.217
source /home/wangji/miniconda3/etc/profile.d/conda.sh
conda activate /usr/local/cuda/envs/nexarm-il
cd /mnt/disk_2T/wangji/projects/NexArm-ImitationLab
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
```

GPU 工程在 `/mnt/disk_2T/wangji/projects/NexArm-ImitationLab`；模型、数据和输出在数据盘。
GPU 负责训练、ACT 推理及任务启动/停止；机器人服务器负责 USB 相机、串口、运动路径检查。
`scripts/gpu_robot.py` 通过 SSH 调用机器人端执行器，使用 `/home/wangji/.ssh/nexarm_robot_ed25519`，源码不保存登录密码。

## 在 GPU 终端观察机器人

```bash
python scripts/gpu_robot.py --status
python scripts/gpu_robot.py --watch
```

`--watch` 每秒打印六轴真实编码器、控制台角度及 `read_only`，Ctrl+C 结束查看。
若机器人端 8765 未监听，命令会打印服务状态并退出，不再触发系统 Apport 异常。
目前提供状态的是临时 GET-only 观察进程；`nexarm-console.service` 保持未启动。PID 记录在机器人 `outputs/read-only-console.pid`。
只读模式拒绝全部 POST 请求，`--execute` 无法启用运动。
`J3 ELBOW` 是肘关节。显示 connected=True 才代表有实时反馈；controller 一行可能保留上次停止原因。

## 已下载 ACT：先看实时预测，再执行

推理服务已运行在 `http://10.206.58.217:8790`：

```bash
curl -fsS http://10.206.58.217:8790/health | python -m json.tool
python scripts/gpu_robot.py --shadow --cycles 20
```

实时画面与预测：<http://10.206.95.245:8788/>。表格“当前反馈”是实际编码器，“预测目标”是模型建议，顶部显示当前模式。
上述命令不会发送运动目标。推理权重是 `data/models/cypher_nexarm_act`，来源为 [CypherChen/NexarmControlModel](https://huggingface.co/CypherChen/NexarmControlModel)。

以下是完整控制服务恢复且现场就绪后的执行入口；当前只读部署会拒绝启用运动：

```bash
python scripts/gpu_robot.py --execute --cycles 8
```

开始前倒计时五秒。每轮采集 front/wrist 两路 RGB 和六轴反馈，在 GPU 生成 100 步 ACT 动作块，仅执行首步，然后重新观察。
该版本用于有限行程联调：单步约 3°、相对本轮起点最多 6°、夹爪累计最多 60 编码器单位。
目标先检查路径，再通过 `/api/policy-step` 持续更新；机器人端独立更新心跳。
控制台在目标流中断 0.8 秒、反馈/供电中断或关节持续无进展时停止。不会把每个小目标都当成需要完全到位的独立点位。
真实误差保存在 `steps.jsonl`，本轮状态在 `summary.json`，现场录像在 `external-camera.mp4`。
GPU 终端会打印这些文件所在的机器人端目录。

从另一个 GPU 终端停止本项目的执行：

```bash
python scripts/gpu_robot.py --stop
```

准备重新启动时清除停止标志，再运行执行命令：

```bash
python scripts/gpu_robot.py --clear-stop
python scripts/gpu_robot.py --execute --cycles 8
```

推理服务若退出，可在单独的 GPU 终端以前台方式启动，保持该终端运行：

```bash
python -m imitation_lab.policy_server \
  --checkpoint data/models/cypher_nexarm_act \
  --device cuda --host 10.206.58.217 --port 8790
```

当前公开权重的状态归一化统计与本机反馈差异很大，程序保留真实输入，没有通过替换均值伪造状态。有限行程执行通过不等于已验证抓取成功。

## GPU 离线评估：已有训练权重

```bash
python -m imitation_lab evaluate \
  --dataset data/processed/public_pick_5hz \
  --checkpoint outputs/public_pick_1000/act/checkpoints/1000/pretrained_model \
  --device cuda --split test \
  --output "outputs/eval-act-$(date +%Y%m%d-%H%M%S)"
```

这里评估的是我们在公共 pick 数据上训练的 ACT，和上面的 CypherChen 现成模型是两个不同权重。
指标衡量留出数据上的动作误差，不能当作实机抓取成功率。

## 训练 ACT、BC 和 Diffusion

先用现有公共数据训练 ACT 20 步，验证训练链路：

```bash
python -m imitation_lab train \
  --config configs/policy/act.yaml --device cuda --steps 20 \
  --output "outputs/act-smoke-$(date +%Y%m%d-%H%M%S)"
```

分别训练一个模型，每次输出到新目录：

```bash
python -m imitation_lab train --config configs/policy/act.yaml \
  --steps 1000 --output "outputs/act-$(date +%Y%m%d-%H%M%S)"
python -m imitation_lab train --config configs/policy/bc_mlp.yaml \
  --steps 1000 --output "outputs/bc-$(date +%Y%m%d-%H%M%S)"
python -m imitation_lab train --config configs/policy/diffusion.yaml \
  --steps 1000 --output "outputs/diffusion-$(date +%Y%m%d-%H%M%S)"
```

统一训练并评估三种模型：

```bash
python scripts/compare_models.py --steps 1000 \
  --output "outputs/compare-$(date +%Y%m%d-%H%M%S)"
```

当前已转换的数据：`data/processed/public_pick_5hz`，来自 `dev-lithion/nexarm_pick_20ep`，20 条轨迹、5 Hz、按原始轨迹划分训练/验证/测试。
用户真实采集包中 action 字段为空，尚不能直接作为监督训练的动作标签。

## 控制台适配与测试

机器人现有控制台已安装 ACT 连续目标接口。需要在同版本控制台重新安装时，在机器人服务器运行：

```bash
git -C /home/ztl/nexarm-console apply --check \
  /home/ztl/NexArm-ImitationLab/integration/console-policy-stream.patch
git -C /home/ztl/nexarm-console apply \
  /home/ztl/NexArm-ImitationLab/integration/console-policy-stream.patch
```

已安装的机器无需重复应用。该面板的 KEY1、KEY2 是功能键，面板没有 EN/RST 键。固件 1.0.2 的桥接模式退出存在恢复问题；本次已通过标准 USB EN 复位恢复常规协议，未刷写固件。

`configs/cypher_normal_sync.yaml` 是新增的官方常规同步写入协议配置，使用命令 70，已通过模拟控制器测试，尚未完成实机验证，也没有作为默认配置激活。
默认 `configs/cypher_console.yaml` 保留桥接协议。当前运动启动被自动审批阻止，工具仅返回 `blocked by policy`，未提供进一步理由。

GPU 侧软件检查：

```bash
python -m unittest discover -s tests -v
ruff check imitation_lab scripts/gpu_robot.py scripts/prepare_act_start.py tests/test_native_deploy.py
```
