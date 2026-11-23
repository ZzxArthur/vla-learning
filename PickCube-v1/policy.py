import torch  # torch.cat(tensors, dim) -> Tensor
import torch.nn as nn  # nn.Module: base class for neural networks
from image_encoder import ImageEncoder  # ImageEncoder(images) -> features


class BehaviorCloningPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()  # 创建和使用分开放
        self.actor_net = nn.Sequential(
            nn.Linear(in_features=137, out_features=64),  # 图像维度 128 + 状态维度 9
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=8)
        )
        pass

    def forward(self, images, states):
        features = self.image_encoder(images)
        combined = torch.cat((features, states), dim=1)
        actions = self.actor_net(combined)
        return actions


if __name__ == "__main__":

    from dataset import DemonstrationDataset  # dataset[i] -> image, state, action
    from torch.utils.data import DataLoader  # DataLoader(dataset, batch_size) -> batches

    path = "具身学习数据/PickCube-v1/rgbd_单条示范/trajectory.rgbd.pd_joint_pos.physx_cpu.h5"
    dataset = DemonstrationDataset(path)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    images, states, expert_actions = next(iter(loader))

    policy = BehaviorCloningPolicy()
    predicted_actions = policy(images, states)

    print("预测动作:", predicted_actions.shape)
    print("专家动作:", expert_actions.shape)
    print("数值是否有限:", torch.isfinite(predicted_actions).all().item())
