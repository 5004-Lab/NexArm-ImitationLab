# NexArm 公共小数据流程与对比结果

2026-11-20，在 `wangji@10.206.58.217` 的 RTX 5090 D v2、Conda `/usr/local/cuda/envs/nexarm-il` 上执行。

## 数据

源数据：[dev-lithion/nexarm_pick_20ep](https://huggingface.co/datasets/dev-lithion/nexarm_pick_20ep)。20 条轨迹，6,000 帧，30 Hz，腕部 RGB。按每 6 帧选 1 帧转成 5 Hz、96×96 RGB，保留 6 维原生状态和动作，不改关节零位、方向或夹爪尺度。

固定 seed 42，14/3/3 个 Episode，共 700/150/150 帧。训练集来源：[19,5,14,4,9,13,15,18,6,12,17,10,1,11]；验证：[2,16,7]；测试：[8,0,3]。

数据集未声明明确许可证，不在仓库分发原始数据和权重。记录中含编码器极端值；此次运行用于工程闭环和初始对比，不是数据质量认证或任务成功率实验。

## 顺序

1. 功能与转换集成测试：11 项通过。
2. 公开数据转换后，逐值验证三份 split 抽样的状态和动作；对应 RGB 逐像素一致。
3. ACT、BC-MLP、Diffusion 分别运行 2 步短训练，保存/加载/测试推理通过。
4. 三模型各运行 1,000 步训练，测试集逐帧推理 150 帧。预测过程中保留动作队列，并在 Episode 边界重置。

## 结果

所有误差单位为 **source_native**，不是角度或 0–1 夹爪开度。关节 RMSE 对前五个轴合并计算；夹爪单独报告 MAE。

| 模型 | 关节 RMSE | 夹爪 MAE | 平均推理耗时 ms | P95 ms |
| --- | ---: | ---: | ---: | ---: |
| ACT | 535.036 | 321.282 | 4.810 | 8.361 |
| BC-MLP | 268.492 | 100.540 | 3.191 | 3.202 |
| Diffusion Policy | 393.234 | 253.379 | 32.964 | 64.755 |

耗时包含预处理、策略调用和反归一化；ACT/Diffusion 的缓存动作步也计入，因此不是每次完整网络重规划的独立耗时。未做专门硬件性能基准。

这次短训练 BC-MLP 的离线误差较低。训练预算、数据量和来源异常值均限制解释范围，不能据此宣称 BC 在 NexArm 真实操作上优于 ACT/Diffusion。下一步应扩大经过复核的成功示教，并运行多种子、数据量与视角消融以及真实机器人评估。

## 产物与复现

服务器：`/mnt/disk_2T/wangji/experiments/nexarm/public_pick_1000/`。

每模型包含训练配置、版本、loss.csv/loss.png、500 与 1000 步检查点、测试 metrics.json/predictions.npz/actions.png；根目录 comparison.json 为汇总。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /usr/local/cuda/envs/nexarm-il
cd /mnt/disk_2T/wangji/projects/NexArm-ImitationLab
source scripts/env.sh
python scripts/check_public_conversion.py
# 使用新输出目录避免覆盖已有实验
python scripts/compare_models.py --steps 1000 --output outputs/public_pick_reproduce
```

控制台输入是 degree/open_fraction，而公共数据是原生编码器坐标；本次检查点仅做离线推理，尚未在真实机械臂执行。原始自采 ZIP 的动作字段全空，不能替代这些监督标签。
