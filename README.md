# VLA / VLM 学习工作区

从 [学习交接](学习交接.md) 开始；[每周任务](每周任务.md) 是每周 15–20 小时的计划。

- [VLA学习指南](VLA学习指南.md)：SmolVLA 图文编码、预训练 VLM、动作专家、训练、解码和仿真。
- [数据构建与VLM后训练指南](数据构建与VLM后训练指南.md)：数据来源、筛选、清理、大模型自动标注、人工复核和小 VLM SFT。
- [HuggingFace与Benchmark实操](HuggingFace与Benchmark实操.md)：模型版本、下载、离线加载及 SmolVLA/LIBERO 评估。
- [历史记录](历史记录.md)、[不懂的知识点](不懂的知识点.md)：已有学习记录。
- `PickCube-v1/`：学习代码、仿真示范和正式模型副本。此目录是从旧子模块复制来的独立目录，不包含旧 Git 元数据；如需版本追踪，后续可自行建立新仓库。

## 当前第一步

在 VS Code 中打开 `/home/hatepp/SSD512/vla-learning`；按 [学习交接](学习交接.md) 实现空文件 `PickCube-v1/smolvla_config_lesson.py`。当前复制的 `.vscode/settings.json` 保留原工作区自动保存和符号补全设置。

原 LeRobot 源码暂时位于 `../预测流/LeRobot/`，本工作区不包含独立副本。旧 `flow_policy.pt` 作为正式结果保留，其中历史 `config.data_path` 指向迁移前路径；重载该模型时若需要数据，请显式使用本目录的 `PickCube-v1/rgbd_单条示范/`，不要覆盖原 checkpoint。

部分旧 BC 示例中的 `__main__` 路径仍指向迁移前目录；直接运行前需调整路径，并保护已有正式 checkpoint。当前 SmolVLA 起步练习应使用 `Path(__file__).parent` 定位数据。

目前资料仅包含本地配置和以前的示范，尚不代表已下载 VLM/SmolVLA 权重、完成 GPU 训练或策略仿真成功。数据和模型后训练的权限、许可证、样本条件与执行结果按新指南逐项验证。
