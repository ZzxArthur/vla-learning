# SmolVLA 学习与复现指南

更新：2026-11-22。主线是理解并使用 SmolVLA，以及学习如何获取数据、清理、自动标注并后训练小型 VLM。SmolVLA 内部用到的 Flow Matching 会作为动作专家的必要机制讲解。

## 目标与学习方式

你将能回答：图像和文字怎样变成模型输入？预训练 VLM 和完整 VLA 如何导入？动作专家是什么，怎么从噪声生成动作？训练数据、损失、冻结参数、优化器、检查点如何连接？如何把输出适配到仿真控制器并评估？

每周 15–20 小时；按 [每周任务](每周任务.md) 推进。学习核心由你实现或追踪；环境、数据整理和已约定脚手架可由教师协助。每一步先给目的、数据来源、shape/dtype、API 和通过标准，再写代码；问题在当前步骤解释，不额外插入脱离目标的专项。

目标岗位为 **VLA 算法与训练实习**。学习成果需要能够真实展示和解释：机器人数据来源、标注质量及序列对齐；VLM/VLA 模块、动作专家和训练目标；模型冻结、梯度与单卡训练；仿真任务的输入动作接口、基线评估和失败分析。仿真、版本和工程工具用于支撑这些算法工作。岗位各有学历、线下和硬件要求；以下按可积累的能力设计，不承诺取得实习或替代真实机器人经验。

**证据分级：** 读源码／读取 config → 实际权重加载 → 实际 forward → GPU forward/backward → 训练 → 仿真 rollout，是不同的里程碑。不能以随机权重的练习或静态代码阅读，声称已运行预训练 SmolVLA。

## 一张总图

```text
自然语言 → tokenizer → input_ids + padding mask → 文本 embedding ┐
相机 RGB → 图像预处理 → vision encoder → connector → 视觉 token ├→ VLM 上下文（prefix）
机器人状态 → 归一化/补齐 → state projection ──────────────────┘          │
                                                                          ↓
噪声动作 + 采样时间 → action/time embedding → 动作专家 ← VLM 上下文
                           │                     │
训练：真实示范动作构造目标速度 ──→ 预测速度与目标计算 loss → 更新参数
推理：噪声 → 多次预测和数值更新 → 连续动作 chunk → 后处理/反归一化
                                                      ↓
                                     根据控制模式映射为仿真机器人动作
                                                      ↓
                                      env.step → 新观测 → 周期性重规划
```

上图概括本地代码；图像、文字和状态的准确形状、处理顺序与训练／推理实现，以当前代码及加载的 checkpoint 配置为准。VLM 特征和动作专家会在其注意力结构中交互；不把图中的每个方框当作独立训练的模型。

## 1. 数据：一条示范包含什么

先检查一个 episode 的相机、语言任务、机器人状态、动作、时间和控制模式，弄清训练时的当前观察与未来动作如何对齐。现有 PickCube 原始 HDF5 是便于理解的**仿真数据**，并非已经可以直接传给 `lerobot-train` 的 LeRobotDataset。

数据来源、筛选、清理、大模型图像标注、人工抽查与 VLM 监督微调的实操路线见 [数据构建与VLM后训练指南](数据构建与VLM后训练指南.md)。图文 SFT 标签和机器人专家动作属于不同的监督，不能互相冒充。

当前可复用样本来自 `PickCube-v1/rgbd_单条示范/` 的 HDF5：原始一张 RGB 为 `[128,128,3]` uint8，qpos 状态为 `[9]`、Panda `pd_joint_pos` 动作为 `[8]`。示范只有一个 episode、74 个动作及 75 条观测，时间对齐需要核对。数据中的任务描述、统计量、camera 名称和 action chunk 对齐需要明确准备；一个 episode 可以做接口检查，不能证明泛化。

数据转换时，记录 episode 边界、fps、时间戳、每帧任务文字、相机 keys、state/action 维度与顺序、控制模式、单位、归一化统计量和末尾 chunk padding。若目标对相机不可见或状态中没有目标，不能假设图文模型能知道目标；仿真场景、提示和可观测输入必须匹配。

## 2. 文字：编码和文本解码

Tokenizer 把文字转为 ID 与 attention mask；文本 embedding 把 ID 查表变成向量，语言主干进一步产生上下文特征。不同模型的词表及 embedding 必须匹配。实作时复用已掌握的 tokenizer 知识，观察 task 结尾换行、padding、截断和有效位置 mask，再对照 [processor_smolvla.py](../预测流/LeRobot/src/lerobot/policies/smolvla/processor_smolvla.py)。

“文本解码”可用 tokenizer 的 `decode` / `batch_decode` 将**文本 ID**转回文字；它和 SmolVLA 的连续动作生成是两件事。单纯 tokenizer 或随机 embedding 不等于加载预训练 VLM。

## 3. 图像和预训练视觉语言模型

此前 CNN 和随机 patch 练习仅用于理解视觉特征和序列，旧练习代码按用户要求已删除。真实代码在 [smolvlm_with_expert.py](../预测流/LeRobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py)：`vision_model → connector`；语言通过 VLM 文本 embedding。`prepare_images` 还负责尺寸调整与像素范围转换；实际大小看配置而不是硬背数字。

区分：

- `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 是本地快照默认使用的**视觉语言骨干**；配置和 tokenizer 不等于权重。
- `lerobot/smolvla_base` 是包含动作专家的**预训练机器人策略检查点**；policy 权重、config、预处理器及其统计量共同构成推理合同。可用 `SmolVLAPolicy.from_pretrained(...)` 加载完整策略，随后按匹配模型的配置构造前后处理器。
- 单独 VLM 的文字生成一般是处理器处理图片/文字、模型生成 token、tokenizer 解码；完整 SmolVLA 不靠文本 `decode` 生成机器人动作。

先用 `AutoConfig.from_pretrained(..., local_files_only=True)` 认识 config（不会加载参数）；再在独立兼容环境中实际下载/加载权重、比较可训练参数及显存，执行真实预处理和前向。先核对本地源码版本与 checkpoint 对应配置，真实输出 shapes 以实测记录。

## 4. 融合、mask 和 action expert

阅读 [modeling_smolvla.py](../预测流/LeRobot/src/lerobot/policies/smolvla/modeling_smolvla.py) 的 `prepare_images`、`embed_prefix`、`embed_suffix`、`forward`。prefix 组织图像、文字、状态；suffix 组织带噪动作和时间信息。padding mask 表示有效位置；模型另有控制 token 之间可见关系的注意力 mask，不能将两者混为一谈。

**动作专家**是根据 VLM/状态条件以及当前带噪动作、时间，预测动作更新方向的网络。在本地实现中它是与 VLM 配合的 Transformer 专家，配合 action 输入／输出投影和时间编码；不是只有一个 `Linear`，也不是把语言 token 翻译成动作 token 的 tokenizer。模型会输出一段动作，即 action chunk，随后按执行策略取动作；chunk 的长度与执行多少步以所加载配置为准。

学习动作生成时先看 [flow_matching.py](../预测流/LeRobot/src/lerobot/policies/common/flow_matching.py) 和 SmolVLA `forward`：本快照 `x_t=t·noise+(1-t)·action`、目标速度 `noise-action`。推理从噪声端按时间逆向做数值更新。本机制是理解 SmolVLA 动作专家的必要知识，不再单独安排其他流方向项目。

## 5. 训练：预训练、微调和成本

视觉语言模型预训练学习图文能力；SmolVLA 机器人预训练进一步使用机器人示范，让图文状态条件与动作相联系；任务微调在已有 checkpoint 基础上适应具体数据。不要把随机初始化 toy MLP 的 loss，称作 VLM 预训练或完整 SmolVLA 微调。

小型 VLM 的图文后训练属于另一条连接到本主线的练习：先固定任务、收集并清洗图文数据、用多模态老师标注并人工复核，再以预训练 VLM 和图文 SFT 目标做 LoRA 微调、独立测试。视觉问答提升不能直接证明动作专家表现提升。

每次训练追踪：真实 batch → processor 和归一化 → action chunk/padding → `policy.forward` / loss → optimizer 只接收可训练参数 → backward → optimizer.step → 保存 policy、processor、config 和统计量。检查冻结参数与梯度、每次更新样本数、随机性和断点；如要精确续训，还需要 optimizer、调度器、步数和 RNG 等状态。

先基于版本匹配的公开/自采数据，用固定一个 batch 做小规模接口验证；租卡后先做**真实完整模型加载 + forward/backward 一步**，测峰值显存和耗时，再确定 batch、冻结策略、更新数与租卡预算。官方文档中的 20k 步/A100 时间是特定设置的参考，不是对本机或任意数据的承诺。优先单卡、先小后大；不从零预训练完整 VLM。

## 6. 两种解码与仿真执行

1. **文字解码：** tokenizer 将文本 token ID 转为字符串，适用于 VLM 语言生成。
2. **动作解码：** 预测速度经过有限次数值更新得到连续 action chunk；去 padding、按需反归一化，取当前应执行的一步或多步；最后转换成仿真控制器定义的动作。

参考官方示例 [using_smolvla_example.py](../预测流/LeRobot/examples/tutorial/smolvla/using_smolvla_example.py)：`from_pretrained → make_pre_post_processors → preprocess → select_action → postprocess`。示例后半段连接 SO100 硬件，不能直接拿来发送 Panda/Piper 命令。先核对 camera key、通道顺序、状态与动作维度、关节顺序、单位、归一化、夹爪含义和控制模式；不把 Panda 的 8 维 `pd_joint_pos` 或训练数据统计直接套给 Piper。

仿真评估以固定协议明确 checkpoint、环境版本、任务和目标可见性、控制频率、动作模式、随机种子、episodes、成功条件和失败情况。先同任务同数据接口单步输出，再跑多个独立 episode、记录成功数/总数、碰撞/超时和视频。单帧推理、单次前反向、单条示范拟合都不等于闭环成功。

## 7. 真实源码阅读地图

| 问题 | 本地源码入口 |
|---|---|
| 配置、chunk 和冻结选项 | [configuration_smolvla.py](../预测流/LeRobot/src/lerobot/policies/smolvla/configuration_smolvla.py) |
| tokenizer、归一化和后处理 | [processor_smolvla.py](../预测流/LeRobot/src/lerobot/policies/smolvla/processor_smolvla.py) |
| VLM、视觉编码与专家构建 | [smolvlm_with_expert.py](../预测流/LeRobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py) |
| 图文状态拼接、训练、动作预测 | [modeling_smolvla.py](../预测流/LeRobot/src/lerobot/policies/smolvla/modeling_smolvla.py) |
| 时间约定、loss 目标、数值积分 | [flow_matching.py](../预测流/LeRobot/src/lerobot/policies/common/flow_matching.py) |
| 官方直接使用示例 | [using_smolvla_example.py](../预测流/LeRobot/examples/tutorial/smolvla/using_smolvla_example.py) |

本地 LeRobot 快照未记录 Git commit；正式实验前记录版本来源及文件指纹。类默认 `chunk_size=50`、图像 resize `512×512`、解码 `num_steps=10` 等是本地配置默认值，不保证任何 checkpoint 都完全相同；最终以 checkpoint 及实测结果为准。

## 8. 当前环境与资料

2026-11-21 的本地检查：LeRobot 快照版本 0.7.1，要求 Python ≥3.12、对应 SmolVLA 依赖；`embodied_flow` 为 Python 3.10，现有环境中未验证完整 LeRobot 可用。另建独立环境，不升级原仿真环境。本机 GPU 显存会受其他进程使用影响；训练容量以实际测试为准。

官方资料：[SmolVLA 文档](https://huggingface.co/docs/lerobot/main/smolvla)、[官方模型卡](https://huggingface.co/lerobot/smolvla_base)、[SmolVLA 发布介绍](https://huggingface.co/blog/smolvla)、[LeRobot GitHub](https://github.com/huggingface/lerobot)。在线文档会随版本更新，按本地快照校对命令。

Hugging Face 模型查找、revision、dry-run、完整权重下载与本地加载，以及 LIBERO benchmark 试跑、统计和对照，见 [HuggingFace与Benchmark实操](HuggingFace与Benchmark实操.md)。当前尚未下载完整权重或运行 benchmark。

## 9. 从入门到前沿：围绕实习逐级交付

每一级以前一级的成果为基础，未通过时保留失败证据，不需要为追新模型反复搭环境。

| 级别 | 能力和可验收成果 | 资源 |
|---|---|---|
| 入门 | 用 PickCube 读出一条真实图像、状态、动作；加载真实 VLM，解释 tokenizer、图像处理、输出和动作的差异 | CPU 准备；模型前向视实际显存选择本机或租卡 |
| 数据与后训练 | 从多个 episode 构建图文样本，先分 episode，再清洗、老师标注、人审；在固定独立测试集比较 VLM 基线与一次小规模 SFT | 优先单卡；先测 GPU 微步骤与费用 |
| SmolVLA 全链路 | 完成真实 checkpoint、处理器、动作专家训练／解码的代码与 shape 追踪；匹配数据后运行 GPU 更新、重载及仿真 | 先从单任务、少量 episode 和小预算开始 |
| 可复现实验 | 在相同任务和评估种子上记录 SmolVLA 与适合的基线结果，包含成功数/总数、失败视频、数据量、更新数、时延、显存 | 依据微步结果选择卡和时长，不把小样本试运行称作论文结果 |
| 前沿扩展 | 在已有基线上只选一个问题：图像/语言扰动鲁棒性、动作 chunk 频率和延迟、数据标注质量；形成可复现对照 | 优先复用已有 checkpoint，增加新内容须替换原有周任务 |
| 架构拓展 | 读 [OpenPI/π0.5](https://github.com/Physical-Intelligence/openpi) 官方 LIBERO 数据转换、策略服务、动作接口，与 SmolVLA 比较；资源允许再选择一个最小实测 | 先读源码；大模型训练/显存以官方版本及自己的 GPU 微步骤实测为准 |

建议的实习项目作品是一个**能从数据、模型到仿真评估完整复现的 SmolVLA 项目**，附一个独立的图文数据标注/小 VLM SFT 实验。每个成果记录自己的工作与 AI/官方代码贡献、输入/输出和数据版本、运行命令、实验配置、失败与资源消耗。面试时练习在 10 分钟内画出数据流、解释训练与生成、定位 shape／归一化／控制模式错误，并说明实验的限制。

算法训练面试重点：能写出并解释训练输入／输出及 loss；说明动作 chunk、时间采样和 padding 的作用，区分图文 SFT、动作监督微调与强化学习；定位冻结参数、梯度、数据泄漏、显存不足和模型/控制器不匹配；说明同训练数据、固定初始条件下如何做基线及消融。先掌握 SmolVLA 的实际训练实现，再学习 π0.5 等更大架构；不要求一上来完整预训练大模型。

真实前沿从可验证的切片进入：[LeRobot 仿真指南](https://huggingface.co/docs/lerobot/simulation) 给出 SmolVLA/LIBERO 相关示例；[LeRobot 数据自动标注](https://huggingface.co/docs/lerobot/annotation_pipeline) 展示多模态老师标注机器人数据；[动作 chunk 的实时执行](https://huggingface.co/docs/lerobot/rtc) 可作为延迟与平滑度的后续研究。在线 `main` 文档可能与本地 0.7.1 快照或其 checkpoint 版本不一致，每次先核对版本和接口，不能把文档示例当作已运行。LIBERO+/其他仿真扩展应隔离环境；不要覆盖当前仿真安装。

可作为后续真实仿真候选的组合是官方示例中的 `lerobot/smolvla_libero` 与 LIBERO 单任务；先固定官方示例所用 checkpoint revision 和相机映射，再检查本地安装版本及 action/state 语义。先运行单 episode 核对接口，再按资源扩大种子数；一次 rollout 不是论文级复现。迁移回 PickCube 时必须重新核对机器人和控制器合同。

算力参考：OpenPI 仓库给出的单卡估计为推理 >8 GB、LoRA 微调 >22.5 GB、全量微调 >70 GB；这是其指定环境的参考，不能保证某张卡或当前空闲显存能运行。当前路线先亲跑 SmolVLA；π0.5 等先做接口和论文／源码对比，只有实际 GPU 微步骤和预算允许时再训练。

岗位样本如[上海人工智能实验室具身智能实习](https://www.shlab.org.cn/joinus/detail/7673347399540738323)覆盖仿真、VLA/WAM 后训练及评测，列举 SAPIEN、MuJoCo 和强化学习等要求；它是能力参考，招聘状态和适用条件以投递时为准。ROS 2、真实机械臂及 RL 可按具体岗位逐步补充；在现有仿真项目中不能虚称实机经验。

**现在开始：** 见 [学习交接：唯一下一步](学习交接.md)。
