import torch  # torch.manual_seed(seed) -> seeded RNG
import torch.nn as nn  # nn.MSELoss()(prediction, target) -> scalar loss
from torch.utils.data import DataLoader  # DataLoader(dataset, batch_size, shuffle) -> batches
from dataset import DemonstrationDataset  # dataset[i] -> image, state, action
from policy import BehaviorCloningPolicy  # policy(images, states) -> actions [B,8]


if __name__ == "__main__":
    seed = 42
    batch_size = 8
    learning_rate = 1e-3
    epochs = 10
    data_path = "具身学习数据/PickCube-v1/rgbd_单条示范/trajectory.rgbd.pd_joint_pos.physx_cpu.h5"
    save_path = "具身学习数据/PickCube-v1/bc_policy.pt"
    torch.manual_seed(seed)
    behaviorcloningpolicy = BehaviorCloningPolicy()
    optimizer = torch.optim.SGD(behaviorcloningpolicy.parameters(), lr=learning_rate)
    dataset = DemonstrationDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  #batch_size代表分多少组, 里面总共n,那就有n/batch_size组数据.
    for _ in range(epochs):
        for images, states, expert_actions in loader:
            actions = behaviorcloningpolicy(images, states)
            loss = nn.functional.mse_loss(actions, expert_actions)
            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                optimizer.step()

    config = {
        "seed": seed,
        "batch_size": batch_size,
        "lr": learning_rate,
        "epochs": epochs,
        "optimizer": "sgd"
    }
    checkpoints = {
        "model": behaviorcloningpolicy.state_dict(),
        "config": config
        }
    torch.save(checkpoints, save_path)

