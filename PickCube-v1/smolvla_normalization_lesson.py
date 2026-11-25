from pathlib import Path
import torch
from safetensors.torch import load_file
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


root = Path(__file__).resolve().parent.parent
policy_path = root / "models" / "smolvla_libero"
vlm_path = root / "models" / "SmolVLM2-500M"

stats = load_file(
    str(policy_path / "policy_preprocessor_step_5_normalizer_processor.safetensors")
)

state_mean = stats["observation.state.mean"]  # [8]
state_std = stats["observation.state.std"]    # [8]
action_mean = stats["action.mean"]           # [7]

config = SmolVLAConfig.from_pretrained(policy_path, local_files_only=True)
config.device = "cuda"

config.vlm_model_name = str(vlm_path)

policy = SmolVLAPolicy.from_pretrained(policy_path, config=config, local_files_only=True, strict=True)
preprocessor, postprocessor = make_pre_post_processors(
    config,
    pretrained_path=policy_path,
    preprocessor_overrides={
        "device_processor": {"device": "cuda"},
        "tokenizer_processor": {"tokenizer_name": str(vlm_path)},
    },
)

state = state_mean + state_std
sample = {
    "observation.state": state,
    "observation.images.camera1": torch.zeros(3, 256, 256),
    "observation.images.camera2": torch.zeros(3, 256, 256),
    "task": "pick up the cube",
}

processed = preprocessor(sample)

output = policy.predict_action_chunk(processed)
first_action = output[:, 0, :]
final_action = postprocessor(first_action)

print("后处理前：", first_action)
print("后处理后：", final_action)