import h5py
from transformers import AutoConfig

data_path = "/home/hatepp/SSD512/vla-learning/PickCube-v1/rgbd_单条示范/trajectory.rgbd.pd_joint_pos.physx_cpu.h5"

with h5py.File(data_path, "r") as data:
    images = data["traj_0/obs/sensor_data/base_camera/rgb"]

    states = data["traj_0/obs/agent/qpos"]

    actions = data["traj_0/actions"]


model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
config = AutoConfig.from_pretrained(model_id, local_files_only=True)
print("模型类型:", config.model_type)
print(
    config.vision_config.image_size,
    config.vision_config.patch_size,
    config.vision_config.hidden_size,
    config.text_config.hidden_size,
    config.scale_factor,
)