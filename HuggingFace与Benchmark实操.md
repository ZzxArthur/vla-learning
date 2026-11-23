# Hugging Face 权重下载与 VLA Benchmark 实操

更新：2026-11-22。目标：能确认模型来源、版本和文件，按需下载并验证加载；能配置 benchmark、观察模型在仿真中的实际表现、解读结果。学习以 SmolVLA 为主，资源优先本机或单张租卡。当前文档中的下载和 benchmark 都是计划，**未因编写本指南而执行下载或评估**。

## 1. 先认清你要下载什么

| 名称 | 它是什么 | 典型用途 |
|---|---|---|
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | 预训练视觉语言模型 | 单图/视频 + 文本理解与生成；适用于本地 SmolVLA 默认 VLM 骨干的学习 |
| `lerobot/smolvla_base` | SmolVLA 机器人策略基础权重 | 研究完整 VLA 结构并在匹配机器人数据上微调 |
| `lerobot/smolvla_libero` | 用于 LIBERO 场景的 SmolVLA checkpoint | 匹配版本、观测和控制接口后评估 LIBERO |

下载前打开模型卡，核对 license、是否 gated、使用范围、仓库 revision、配置、权重文件、processor/tokenizer 和动作/输入格式。配置 `config.json` 描述网络结构，**不是模型学到的权重**。完整 SmolVLA 推理可能还需要配套 preprocessor/postprocessor 和归一化统计量，不要只下载一个文件后假定整个策略可运行。

2026-11-22 查询时，SmolVLM2 的一个权重文件约 2.03 GB、SmolVLA base 一个权重文件约 0.91 GB；文件列表、revision 和大小以后可能变化。**磁盘文件大小不是显存需求**：实际显存还受精度、输入、优化器和 batch 影响，须加载及微步骤实测。

## 2. Hugging Face 练习阶梯

### A：看懂模型仓库，不下载大文件

1. 打开上面三个模型的 Hugging Face 页面，记录 repo ID、用途、许可、一个明确的 revision、权重文件名和配套文件。
2. 本机 `embodied_flow` 的 `huggingface_hub` 2.4.0 已检查存在 `hf` CLI；以实际 `hf download --help` 为准。命令行 dry run 用于联网查看文件是否需要下载，**不会下载模型权重**：

```bash
/home/hatepp/miniconda3/envs/embodied_flow/bin/hf download \
  HuggingFaceTB/SmolVLM2-500M-Video-Instruct --dry-run
```

3. 阅读输出中的文件和预估大小；若网络不可用、版本或访问权限不同，先记录错误，不把离线 config 当作权重已下载。gated 模型按页面要求确认访问权并使用登录机制；不把 token 写进脚本或日志。

### B：先操作单个配置文件

使用 `hf_hub_download` 指定 repo ID 和 `config.json`；它返回文件路径，可读取 JSON 对照当前 `AutoConfig`。若只学习离线缓存，设置 `local_files_only=True`，未缓存就报错；若希望下载新的小文件，去掉该选项。执行前先确认下载目录和版本。

```python
from huggingface_hub import hf_hub_download  # hf_hub_download(repo_id, filename, ...) -> local path
```

例如使用 `revision` 固定 commit；本次练习先查看返回路径和配置字段。当前 `smolvla_config_lesson.py` 已安排读取本地配置，不与本步骤重复。

### C：下载完整模型并加载

实际需要模型时，先 `--dry-run` 确定大小、license 和版本；再将目标目录明确设置在 SSD 的工作区里，例如命令中的 `--local-dir /home/hatepp/SSD512/vla-learning/models/SmolVLM2-500M`。这是**一次具体下载**，会实际占用磁盘和网络：

```bash
/home/hatepp/miniconda3/envs/embodied_flow/bin/hf download \
  HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --revision <已核对的完整commit> \
  --local-dir /home/hatepp/SSD512/vla-learning/models/SmolVLM2-500M
```

`<已核对的完整commit>` 是占位符，运行时由实际模型页面或 metadata 替换。`--local-dir` 会在目标目录创建 Hugging Face 的下载元数据；另一方式是 `cache_dir` 使用 Hugging Face 缓存。默认缓存与显式目录不同，下载前确认是否会重复占用空间。**不修改或迁移系统整个 `~/.cache`**。

下载后用模型和 processor 的 `from_pretrained(本地模型目录, local_files_only=True)` 读取；对于完整 SmolVLA 使用相应 `SmolVLAPolicy` 和配套 processor，而不是当成普通文本 VLM。固定模型 revision 和库版本；先加载／单张前向，再检查实际输出和显存。访问受限、下载中断、文件缺失、processor 不匹配分别记录；不能把成功下载当成成功推理。

策略仓库中的文件清单不一定覆盖完整离线运行所需的所有关联资源；模型初始化可能还需从 VLM 仓库获取 tokenizer、processor 或骨干权重。记录每个来源和 revision，以离线实际加载结果验证完整性。

相关官方资料：[下载文档](https://huggingface.co/docs/huggingface_hub/guides/download)、[CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)、[缓存机制](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)。仓库和工具会更新，执行时先检查所安装版本。不要删除或清空共享缓存、已有 checkpoint。

## 3. Benchmark 是什么

Benchmark 是约定好的**任务集 + 环境与输入输出 + 执行和评分协议**，不是只有一个分数。例如机器人操作中，需要确定任务、机器人、相机、初始状态、控制模式、策略和每次回合的成功判定。

要区分三种证据：

- **接口检查：** 能不能加载模型、处理一批观测并输出动作；不涉及任务成功。
- **仿真 smoke：** 1 个任务、1 个 episode 是否能完整执行；只验证链路，不代表成功率或论文复现。
- **评估：** 按固定任务、随机种子和 episode 数重复运行，记录成功数/总数、失败与成本；样本少时要说明不确定性。

图文 VLM SFT 还需单独的图文测试集（分类/问答、格式正确率、unknown、标注质量）；VLA 的 LIBERO 是动作控制评估。两者不能用相同的“准确率”混在一起。

## 4. 第一个 VLA Benchmark：SmolVLA + LIBERO

[LeRobot 仿真指南](https://huggingface.co/docs/lerobot/simulation)给出 `lerobot/smolvla_libero` 在 `libero_spatial` 一个任务上的最小评估例子。它适合在学完 SmolVLA 输入/输出后跑通 benchmark；现有 PickCube 用于理解数据和控制接口，不要求把 Panda 的输入直接塞给 LIBERO checkpoint。

具体核对：

1. 固定 LeRobot、LIBERO/MuJoCo、模型 revision、processor 及仿真环境；安装或渲染失败先单独诊断，远程无头 EGL 需要对应驱动，不假设本机必然能用。
2. 按 checkpoint 映射主相机和腕部相机，核对语言、状态、动作和归一化；当前官方 LIBERO 数据通常是 8 维状态、7 维末端增量/夹爪动作，而本地 PickCube 示范是 9 维关节状态、8 维 `pd_joint_pos` 动作。它们不能直接互换。
3. 对照在线 [LIBERO 文档](https://huggingface.co/docs/lerobot/libero) 和实际安装版本确认任务名称、seed、reset、`rename_map`、控制频率、`n_action_steps`、输出目录；代码版本不同选项可能变化。
4. 先启动一个环境并验证 reset、观测和设备；再加载匹配策略，跑 **1 个任务 × 1 个 episode**。记录加载成功、输入形状、输出动作、结束原因和视频，仅称 smoke。
5. 复用同一 checkpoint、任务、协议和一组预先记录的随机种子，逐步跑多个 episode；统计 `成功次数/总次数`，记录超时/碰撞/动作异常等失败。样本量小不要宣传稳定排名。
6. 形成对照时控制相机、任务、初始种子、回合上限、动作执行长度、推理设置；如比较不同训练方案，还要报告数据量、训练更新数、时间与卡时。可增加失败视频和推理延迟 P50/P95。

官方在线最小示例引用的模型 revision 为 `a24f1f380461b665158d52acd140152c79e8b36d`，任务为 `libero_spatial` 的一个任务；本地 LeRobot 0.7.1 快照和在线文档可能更新。正式运行之前核对该版本是否支持相同 CLI 和 checkpoint。当前尚未安装并验证整个 benchmark 环境，**不要将上述示例视为本机已跑通**。

以下是官方示例的学习用命令，需先完成依赖、显存及渲染检查；它会加载/下载权重、使用 GPU 并向 `outputs/eval_libero` 写评估结果。修改输出目录时不要覆盖已存在的正式实验：

```bash
lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --policy.pretrained_revision=a24f1f380461b665158d52acd140152c79e8b36d \
  --policy.device=cuda \
  --env.type=libero --env.task=libero_spatial --env.task_ids='[0]' \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --eval.n_episodes=1 --eval.batch_size=1 \
  --output_dir=outputs/eval_libero
```

这份示例同时出现在本地 `../预测流/LeRobot/docs/source/simulation.mdx`；只是源码和官方文档里的示例，本机尚未执行。`policy.path` 指模型，`env` 指任务，`rename_map` 对齐两个相机名，`n_episodes` 是每任务的测试局数。完整官方 LIBERO 建议协议为四组共 40 任务、每任务 10 局并在多个评估种子上统计；它比一局试跑昂贵得多，按预算分阶段做。更多重复评估种子也不等于多个独立训练种子。

## 5. 交付与算力

每个实验保存：模型仓库/权重 revision、代码和依赖版本、处理器与数据版本、任务/相机/动作配置、评估 seeds 与数量、成功数/总数、失败记录、运行耗时和硬件；对比结果附带实际预算差异。用户请求运行的正式评估、视频和 checkpoints 保留；临时 smoke 验证的脚本、截图及临时权重按既定规则清理。

先测实际模型加载与单 episode 的耗时/显存，估算租卡预算；训练还需单个真实 batch 的 forward/backward 微步骤。超预算时减少任务或先做 CPU 输入合同，不把静态阅读声称为 benchmark 成功。

## 6. 当前安排

现在仍从 [学习交接](学习交接.md) 的本地配置与数据合同开始；随后完成仓库文件查询、版本和下载练习、真实权重加载，再进入 benchmark。两个模块已纳入 [每周任务](每周任务.md) 中对应时段，不额外叠加一整套课程。
