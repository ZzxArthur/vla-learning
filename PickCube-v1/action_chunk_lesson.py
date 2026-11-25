from pathlib import Path  # Path(__file__).parent -> directory containing this script
import h5py  # h5py.File(path, "r") -> HDF5 file
import numpy as np  # np.zeros(shape, dtype) -> array filled with zeros
import torch.nn as nn
import torch.nn.functional as F
import torch


class Prediction_Velocity_Net(nn.Module):
    def __init__(self, input_dimension, v_hidden_dimension, action_dimension):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dimension, v_hidden_dimension),
            nn.Tanh(),

            nn.Linear(v_hidden_dimension, action_dimension)
        )

    def forward(self, input):
        prediction_velocity = self.net(input)
        return prediction_velocity


data_path = (
    Path(__file__).parent
    / "rgbd_单条示范"
    / "trajectory.rgbd.pd_joint_pos.physx_cpu.h5"
)

with h5py.File(data_path, "r") as data:
    actions = data["traj_0/actions"][:]
    states = data["traj_0/obs/agent/qpos"][:]

chunk_size = 5
chunks = []
valid_masks = []
for step in range(len(actions)):
    chunk = np.zeros((chunk_size, 8), dtype=np.float32)
    valid_mask = np.zeros(chunk_size, dtype=bool)

    real_actions = actions[step:step + chunk_size] # 超出chunk_size不会补齐
    num_real = len(real_actions) # 统计真实动作的数量

    chunk[:num_real] = real_actions
    valid_mask[:num_real] = True

    chunks.append(chunk)
    valid_masks.append(valid_mask)

actions_batch = np.stack(chunks, axis=0)
masks_batch = np.stack(valid_masks, axis=0)

target_actions = torch.from_numpy(actions_batch)   # 专家动作 [*, 5, 8]
valid = torch.from_numpy(masks_batch)      # 有效位置 [*, 5]

prediction = torch.ones_like(target_actions)      # 全为 1，模拟模型预测

errors = F.mse_loss(prediction, target_actions, reduction="none")
expanded_mask = valid.unsqueeze(-1) # 让mask维度匹配errors
vaild_errors = errors * expanded_mask

action_loss = vaild_errors.sum() / (valid.sum() * target_actions.shape[-1])
input_dimension = 18
v_hidden_dimension = 256
action_dimension = actions.shape[-1]
prediction_velocity_net = Prediction_Velocity_Net(input_dimension, v_hidden_dimension, action_dimension)
optimizer = torch.optim.Adam(prediction_velocity_net.parameters(), lr=0.01)
for step in range(200):
    noise = torch.randn_like(target_actions)
    t = torch.rand(target_actions.shape[0], 1, 1).expand(-1, chunk_size, -1)
    x_t = (1 - t) * target_actions + t * noise

    states_batch = states[:target_actions.shape[0]]
    states_batch = torch.from_numpy(states_batch).unsqueeze(1).expand(-1, chunk_size, -1)

    input = torch.cat((x_t, t, states_batch), dim=-1)

    prediction_velocity = prediction_velocity_net(input)
    target_velocity = noise - target_actions
    v_errors = F.mse_loss(prediction_velocity, target_velocity, reduction="none")
    v_valid_errors = v_errors * expanded_mask
    v_loss = v_valid_errors.sum() / (valid.sum() * target_actions.shape[-1])
    optimizer.zero_grad()
    v_loss.backward()
    with torch.no_grad():
        optimizer.step()

    if step % 50 == 0:
        print(v_loss)

prediction_velocity_net.eval()
sample_state = states_batch[0:1]  # [1, 5, 9]
x = torch.randn(1, chunk_size, action_dimension)  # [1, 5, 8]
num_steps = 10
dt = -1.0 / num_steps
with torch.no_grad():
    for i in range(num_steps):
        current_t = torch.full((1, chunk_size, 1), 1.0 + i * dt)

        input = torch.cat((x, current_t, sample_state), dim=-1)
        prediction_v = prediction_velocity_net(input)
        x = x + prediction_v * dt
        print(x.sum())