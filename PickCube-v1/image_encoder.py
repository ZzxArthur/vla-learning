import torch  # torch.Tensor: multidimensional tensor
import torch.nn as nn  # nn.Module: base class for neural networks


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 第一层卷积: [B, 3, 128, 128] -> [B, 16, 64, 64]
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 第二层卷积: [B, 16, 64, 64] -> [B, 32, 32, 32]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 第三层卷积: [B, 32, 32, 32] -> [B, 32, 16, 16]
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 自适应平均池化: [B, 32, 16, 16] -> [B, 32, 4, 4]
            nn.AdaptiveAvgPool2d((4, 4)),

            # 展平（保留第0维 Batch）: [B, 32, 4, 4] -> [B, 512]
            nn.Flatten(start_dim=1),

            # 全连接层: [B, 512] -> [B, 128]
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU()
        )
        # 在这里定义卷积层、池化层和全连接层

    def forward(self, images):
        features = self.net(images)
        # 输入 images: [B, 3, 128, 128]
        # 输出 features: [B, 128]
        return features


if __name__ == "__main__":
    encoder = ImageEncoder()
    images = torch.rand(8, 3, 128, 128)

    features = encoder(images)
    print(features.shape)
