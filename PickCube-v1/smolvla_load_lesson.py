from pathlib import Path
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

import torch

root = Path(__file__).resolve().parent.parent

policy_path = root / "models" / "smolvla_base"
vlm_path = root / "models" / "SmolVLM2-500M"


config = SmolVLAConfig.from_pretrained(
    policy_path,
    local_files_only=True,
)

config.device = "cuda"
config.vlm_model_name = str(vlm_path)

policy = SmolVLAPolicy.from_pretrained(
    policy_path,
    config=config,
    local_files_only=True,
    strict=True,
)

preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=policy_path,
    preprocessor_overrides={
        "device_processor": {"device": "cuda"},
        "tokenizer_processor": {"tokenizer_name": str(vlm_path)},
    },
)
sample = {
    "observation.state": torch.zeros(6),

    "observation.images.camera1": torch.zeros(3, 256, 256),
    "observation.images.camera2": torch.zeros(3, 256, 256),
    "observation.images.camera3": torch.zeros(3, 256, 256),

    "task": "pick up the red cube",
}
processed = preprocessor(sample)
for name, value in processed.items():
    if isinstance(value, torch.Tensor):
        print(name, value.shape, value.dtype, value.device)