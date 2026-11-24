from pathlib import Path  # Path(__file__).parent -> directory containing this script
import h5py  # h5py.File(path, "r") -> HDF5 file
import numpy as np  # np.zeros(shape, dtype) -> array filled with zeros
import torch
import torch.nn.functional as F

data_path = (
    Path(__file__).parent
    / "rgbd_单条示范"
    / "trajectory.rgbd.pd_joint_pos.physx_cpu.h5"
)

with h5py.File(data_path, "r") as data:
    actions = data["traj_0/actions"][:]
print(actions.shape)
chunk = np.zeros((5, 8), dtype=np.float32)
valid_mask = np.zeros(5, dtype=bool)
chunks = []
valid_masks = []
for step in [0, 30, 72]:
    chunk = np.zeros((5, 8), dtype=np.float32)
    valid_mask = np.zeros(5, dtype=bool)
    real_actions = actions[step:step + 5]
    num_real = len(real_actions)
    chunk[:num_real] = real_actions
    valid_mask[:num_real] = True
    chunks.append(chunk)
    valid_masks.append(valid_mask)


action_batch = np.stack(chunks, axis=0)
mask_batch = np.stack(valid_masks, axis=0)

target = torch.from_numpy(action_batch)   # 专家动作 [3, 5, 8]
valid = torch.from_numpy(mask_batch)      # 有效位置 [3, 5]

prediction = torch.ones_like(target)      # 全为 1，模拟模型预测

errors = F.mse_loss(prediction, target, reduction="none")
print(errors)
expanded_mask = valid.unsqueeze(-1)

errors = errors * expanded_mask

sum_errors = errors.sum()
loss = sum_errors / (valid.sum() * target.shape[-1])

print(loss)