from pathlib import Path

import pyarrow.parquet as pq
import json
import torch
root = Path(__file__).resolve().parent.parent
info_path = root / "data" / "libero_metadata" / "meta" / "info.json"
data_path = root / "data" / "libero_metadata" / "data" / "chunk-000" / "file-000.parquet"
task_path = root / "data" / "libero_metadata" / "meta" / "tasks.parquet"
with open(info_path, "r", encoding="utf-8") as file:
    info = json.load(file)


table = pq.read_table(data_path)
print("行数：", table.num_rows)
print("字段：", table.column_names)

row = table.slice(0, 1).to_pylist()[0] # 字典
print(row.items())

task_table = pq.read_table(task_path)
task_rows = task_table.to_pylist()
task_index = row["task_index"]

for task_row in task_rows:
    if task_row["task_index"] == task_index:
        print("对应任务：", task_row["__index_level_0__"])
        break

rows = table.to_pylist()

# 先选出同一条示范，避免动作序列跨到下一条 episode。
episode_rows = [row for row in rows if row["episode_index"] == 0]
print("\nepisode 0 的记录数：", len(episode_rows))

chunk_size = 50
action_dim = 7
start_indices = [0, 200]

for start in start_indices:
    # 从当前示范中取出最多 50 行；切片到列表末尾会自动停止。
    selected_rows = episode_rows[start : start + chunk_size]
    valid_count = len(selected_rows)

    # 从 0 号位置开始写入真实动作，剩下的位置以零填充作为教学示意。
    action_chunk = torch.zeros((chunk_size, action_dim), dtype=torch.float32)
    actions = torch.tensor([row["action"] for row in selected_rows], dtype=torch.float32)
    action_chunk[:valid_count] = actions

    # SmolVLA 使用 True 标记填充，False 标记真实动作。
    action_is_pad = torch.ones(chunk_size, dtype=torch.bool)
    action_is_pad[:valid_count] = False

    print(f"\n起始帧：{start}")
    print("动作 chunk 形状：", action_chunk.shape)
    print("真实动作数：", valid_count)
    print("填充动作数：", int(action_is_pad.sum().item()))
    print("填充掩码：", action_is_pad)
    print("第一步动作：", action_chunk[0])

# 以上动作 chunk 部分主要代码由 AI 助手完成。
