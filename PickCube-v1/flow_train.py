from pathlib import Path
from torch.utils.data import DataLoader
from dataset import DemonstrationDataset
import torch

lr = 0.001
epochs = 20
batch_size = 8
torch.manual_seed(42)

data_path = Path(__file__).parent / "rgbd_单条示范/trajectory.rgbd.pd_joint_pos.physx_cpu.h5"
dataset = DemonstrationDataset(data_path)
data = DataLoader(dataset, batch_size=batch_size, shuffle=True)

flow_net = torch.nn.Sequential(
    torch.nn.Linear(18, 64),
    torch.nn.Tanh(),

    torch.nn.Linear(64, 8)
)
flow_optimizer = torch.optim.Adam(flow_net.parameters(), lr=lr)

for i in range(epochs):
    epoch_loss = 0.0
    epoch_samples = 0
    for images, states, expert_actions in data:

        current_batch_size = expert_actions.shape[0]
        noise_actions = torch.randn_like(expert_actions)
        t = torch.rand(current_batch_size, 1)

        x_t = t * noise_actions + (1 - t) * expert_actions
        u_t = noise_actions - expert_actions

        flow_input = torch.cat((x_t, t, states), dim=1)

        flow_output = flow_net(flow_input)
        loss = torch.nn.functional.mse_loss(flow_output, u_t)
        flow_optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            flow_optimizer.step()
        epoch_loss += loss.item() * current_batch_size
        epoch_samples += current_batch_size
    print(f"epoch {i + 1:02d}/{epochs}: mean_loss={epoch_loss / epoch_samples:.6f}")

config = {
    "state_dim": 9,
    "action_dim": 8,
    "hidden_dim": 64,
    "activation": "tanh",
    "input_order": ["noisy_action", "time", "state"],
    "flow_convention": "noise_at_one",
    "control_mode": "pd_joint_pos",
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": lr,
    "optimizer": "Adam",
    "seed": 42,
    "data_path": str(data_path),
}
checkpoint_path = Path(__file__).parent / "flow_policy.pt"
torch.save({"model_state_dict": flow_net.state_dict(), "config": config}, checkpoint_path)
print(f"模型已保存: {checkpoint_path}")
