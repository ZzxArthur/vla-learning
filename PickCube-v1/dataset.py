import h5py  # h5py.File(path, mode) -> HDF5 file
import torch  # torch.from_numpy(array) -> Tensor
from torch.utils.data import Dataset, DataLoader  # Dataset[index] -> sample


class DemonstrationDataset(Dataset):
    def __init__(self, path):
        with h5py.File(path, "r") as data:
            episode = data["traj_0"]

            # [:] 将数据读入内存，关闭文件后仍然可以使用
            self.images = episode["obs/sensor_data/base_camera/rgb"][:]
            self.states = episode["obs/agent/qpos"][:]
            self.actions = episode["actions"][:]

        # 本函数主要代码由 AI 助手完成。

    def __len__(self):
        # TODO：应该有多少条“观测 → 动作”样本？
        return len(self.actions)  # 返回动作的数量，即样本数量

    def __getitem__(self, index):
        # TODO：
        # 1. 取出 index 时刻的图像、状态和动作
        image = self.images[index]
        action = self.actions[index]
        state = self.states[index]
        image = torch.as_tensor(image, dtype=torch.float32) # 转为 Tensor 并调整维度
        # 2. 图像转为 Tensor，并调整为 [3, 128, 128]
        image = image.permute(2, 0, 1)  # 调整维度为 [3, 128, 128]
        # 3. 图像转 float32，除以 255
        image = image / 255.0  # 将像素值归一化到 [0, 1]
        # 4. 状态、动作转为 Tensor
        state = torch.as_tensor(self.states[index], dtype=torch.float32)
        action = torch.as_tensor(self.actions[index], dtype=torch.float32)
        # 5. 返回 image, state, action
        return image, state, action


if __name__ == "__main__":
    path = "具身学习数据/PickCube-v1/rgbd_单条示范/trajectory.rgbd.pd_joint_pos.physx_cpu.h5"

    dataset = DemonstrationDataset(path)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    images, states, actions = next(iter(loader))

    print("图像批次:", images.shape)
    print("状态批次:", states.shape)
    print("动作批次:", actions.shape)
