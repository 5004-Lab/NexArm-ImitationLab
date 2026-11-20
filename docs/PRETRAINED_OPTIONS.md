# 公开模型选择与真机执行状态

## 影子模式与实际执行

影子模式读取真实相机/反馈并预测动作，不发送电机目标。实际执行会发送目标，读取执行后的反馈并继续预测。模型下载、离线误差和影子模式均不等于完成真实抓取。

## 本次检索到的资源

| 资源 | 已核实的内容 | 与当前 NexArm 的关系 |
| --- | --- | --- |
| [ZibinDong/so100_grab_screwdriver](https://huggingface.co/datasets/ZibinDong/so100_grab_screwdriver) | 57 条、28,554 帧、30 Hz，螺丝刀抓取数据，base/wrist RGB | robot_type=so100_follower，不是 NexArm |
| [Beegbrain/put_screwdriver_box](https://huggingface.co/datasets/Beegbrain/put_screwdriver_box) | 30 条、9,347 帧、20 Hz，螺丝刀放入盒子 | robot_type=moss，不是 NexArm |
| [wry123456/act_screwdriver_model](https://huggingface.co/wry123456/act_screwdriver_model) | ACT 权重，训练配置指向 wry123456/pick_screwdriver_handle | 关联数据 robot_type=so101_follower，不能直接输出给 NexArm |
| [CypherChen/NexarmControlModel](https://huggingface.co/CypherChen/NexarmControlModel) | 作者称 NexArm ACT 抓放；31 条、约 27K 帧、约 29K 训练步；front/wrist 640×480；有完整权重和处理器 | 同型号候选，但物体未明确，训练数据路径为作者本机 local/test，现场相机/输入定义仍需核对 |
| [thanhkt/nexarm_stack_bowls_act](https://huggingface.co/thanhkt/nexarm_stack_bowls_act) | ACT 权重、6 维状态/动作、front/wrist 输入 | 叠碗候选，不是螺丝刀抓取；没有模型卡，不能宣称已在本机验证 |

检索日期：2026-11-20。本次未找到能核实为“当前 NexArm 螺丝刀抓放”的公开数据及配套权重组合。不是声称其他模型不存在，而是上述候选的机器人/标定/任务信息没有同时满足直接复用条件。

## 已执行的小步测试

当前 BC-MLP 已部署机器人端，并尝试发送按预测方向截取的最多约 2 度短段指令。执行中第二小步的心跳收到 HTTP 409，随后发送 stop，机械臂停止并锁定。编码器反馈有小幅变化，未完成完整动作序列，也未抓起笔。

最初的限幅器在实际姿态已经超出软限位时，最后一次静态裁剪会产生超出速度限制的瞬时回拉。现已修正为从当前实测位置按变化率平滑退回，并加入回归测试；没有移除控制台路径检查。

`trial_model_step.py` 只做单个预测目标的小幅执行接口测试，不是完整抓取程序。它保留控制台规划、心跳和 stop；发生错误记录控制台响应，不自动重试运动。

```bash
# 机器人服务器，只计算目标并检查路径
cd /home/ztl/NexArm-ImitationLab
.venv/bin/python scripts/trial_model_step.py
```

完整任务复现需要匹配机器人关节定义、夹爪尺度、相机视角、初始姿态、目标物体和位置分布。当前模型尚未证明能抓取用户桌面的笔或螺丝刀。

## CypherChen 检查点部署与加载结果

完整 ACT 权重和处理器已部署到机器人服务器：`/home/ztl/NexArm-ImitationLab/data/models/cypher_nexarm_act`，权重文件 206,699,736 字节。已用现场全局 USB 相机对应作者的 front 输入、现场腕部相机对应 wrist 输入，完成 CPU 加载和一次前向预测；没有发送该检查点的输出到电机。

实测 ARM CPU 一次前向约 2,015 ms。当前状态编码器为 `[2010,2139,1236,2168,1884,2780]`，预测约 `[2325,1759,1757,2324,1394,2297]`；其中腕部目标 2324 超出本机配置的 2121 上限。

模型自带状态归一化均值约 `[17.4,17.5,64.7,4094.5,543.9,4094.2]`，与当前反馈分布显著不同。这是需要核对作者输入定义和标定的证据，不能据模型卡名称宣称即插即用，也不据此断言作者模型在其原平台无效。

现场加载检查记录在机器人服务器 `outputs/cypher-load-check.json`。实时页面目前仍展示本项目 BC-MLP 的预测和已停止的小步试验状态，不把新 ACT 检查点冒充成成功抓取演示。
