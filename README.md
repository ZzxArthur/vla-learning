# VLA Learning：从图文编码到机器人动作

这是一个面向 **VLA 算法与训练**的学习与实践仓库，以 [SmolVLA](https://huggingface.co/docs/lerobot/main/smolvla) 为主线，从图像、语言和机器人数据出发，逐步学习模型训练、动作生成与仿真评估。项目以仿真和单卡、小预算实验为主，并记录学习过程、代码与已有数据。

> **项目状态：学习进行中。** 仓库已有 PickCube 仿真示范和以往练习的检查点；完整 SmolVLA 权重加载、微调及闭环 benchmark 仍是后续任务，不能将已有产物视为 SmolVLA 复现结果。

## 学习路线

```text
图像、指令、机器人状态和示范动作
                ↓
数据检查与清洗 → 图像/文本预处理 → 预训练视觉语言模型（VLM）
                                           ↓
                               动作专家 → 连续动作序列
                                           ↓
                           动作后处理 → 仿真执行与评估
```

| 阶段 | 学习内容 | 状态 |
| --- | --- | --- |
| 数据与基础模块 | 读取 PickCube 示范，理解图像、状态、动作与基础编码 | 已有单条 RGB 示范，重新整理练习 |
| 预训练 VLM | Hugging Face 权重下载、图文预处理、真实前向及文本解码 | 待完成 |
| 数据与 VLM 后训练 | 数据获取、筛选、清洗、多模态模型自动标注与人工复核；小型 VLM 监督微调 | 待完成 |
| SmolVLA | 理解 VLM 与动作专家、训练目标、连续动作生成及后处理 | 待完成 |
| 仿真与 benchmark | 核对机器人接口，在匹配任务上评估并分析失败 | 待完成 |

具体进度和每周 15–20 小时的任务，以 [学习交接](学习交接.md) 和 [每周任务](每周任务.md) 为准。学习中会分别记录权重加载、前向、训练和仿真执行，避免把其中一步当作完整复现。

## 仓库内容

```text
vla-learning/
├── PickCube-v1/
│   ├── smolvla_config_lesson.py   # 待完成的 VLM 配置练习
│   ├── rgbd_单条示范/              # RGB、状态与动作示范
│   ├── motionplanning/            # 运动规划轨迹和视频
│   ├── teleop/                    # 遥操作轨迹和视频
│   └── rl/                        # RL 轨迹、视频与已有检查点
├── VLA学习指南.md
├── 数据构建与VLM后训练指南.md
├── HuggingFace与Benchmark实操.md
├── 每周任务.md
└── 学习交接.md
```

`bc_policy.pt` 是已有行为克隆练习的检查点；`flow_policy.pt` 是早期小型动作生成练习的检查点，**不是 SmolVLA 权重**。其他轨迹与检查点仅表示仓库中已有相应文件，不代表本项目已完成基准评测。数据来源、用途和许可应在进一步使用或再分发前核对。

## 快速开始

克隆仓库并进入目录：

```bash
git clone https://github.com/ZzxArthur/vla-learning.git
cd vla-learning
```

从 [学习交接](学习交接.md) 开始，完成 `PickCube-v1/smolvla_config_lesson.py` 中的 VLM 配置和 HDF5 数据读取练习。该文件目前是空的，等待学习者实现；读取数据需要 `h5py`，读取配置需要 `transformers` 和本地缓存的模型配置。

完整 SmolVLA 练习需要另外核对 LeRobot、Python、PyTorch 和模型版本，见 [Hugging Face 与 Benchmark 实操](HuggingFace与Benchmark实操.md)。本仓库暂未提供统一锁定依赖的环境文件。

## PickCube 数据说明

当前 RGB 单示范包含 1 个 episode、75 条观测和 74 个动作。HDF5 中的原始单条数据为：

| 数据 | 形状 | 含义 |
| --- | --- | --- |
| 图像 | `[128, 128, 3]` | RGB，`uint8`，通道在最后 |
| 状态 | `[9]` | Panda 机器人关节状态 |
| 动作 | `[8]` | Panda `pd_joint_pos` 控制模式下的动作 |

这份 HDF5 数据适合学习和接口检查，**不是可直接用于 SmolVLA 的 LeRobotDataset**。用于训练前还需要核对任务文字、相机映射、统计量、动作时间对齐和数据格式；单条示范也不足以评估泛化。`motionplanning` 和 `teleop` 轨迹不能直接当作现成 RGB 示范。

观测与动作的时间对齐需要进一步核对。Panda 的状态、动作维度及控制模式不能直接用于 Piper 实机。

## 文档导航

- [学习交接](学习交接.md)：当前进度、下一步和学习约定。
- [每周任务](每周任务.md)：按周安排与验收要求。
- [VLA 学习指南](VLA学习指南.md)：图文编码、动作专家、训练与仿真评估。
- [数据构建与 VLM 后训练指南](数据构建与VLM后训练指南.md)：数据获取、筛选、清洗、自动标注和小 VLM 微调。
- [Hugging Face 与 Benchmark 实操](HuggingFace与Benchmark实操.md)：模型下载、版本核对及 benchmark 入门。
- [历史记录](历史记录.md)与[不懂的知识点](不懂的知识点.md)：学习记录和待理解问题。

## 参考项目

- [Hugging Face LeRobot](https://github.com/huggingface/lerobot)：机器人数据、策略训练与评估工具。
- [SmolVLA 官方文档](https://huggingface.co/docs/lerobot/main/smolvla)：模型结构与使用说明。
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)：机器人操作仿真 benchmark。

本仓库是学习记录，不附带完整的 LeRobot 源码或 SmolVLA 预训练权重。不同版本的接口可能变化，实验时应记录实际使用的源码版本、模型 revision 和环境。
