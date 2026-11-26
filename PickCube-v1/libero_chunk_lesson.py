from lerobot.sim.integrations.libero import env_fns, canonical_observation
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from pathlib import Path

import torch
import numpy as np

torch.manual_seed(42)
root = Path(__file__).resolve().parent.parent

factories = env_fns(
    task="libero_spatial",
    n_envs=1,
    gym_kwargs={
        "task_ids": [0],
        "obs_type": "pixels_agent_pos",
        "observation_height": 256,
        "observation_width": 256,
    },
)

policy_path = root / "models" / "smolvla_libero"
vlm_path = root / "models" / "SmolVLM2-500M"
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

env = factories["libero_spatial"][0][0]()

success_count = 0
num_episodes = 3


for episode in range(num_episodes):
    init_state_index = env.init_state_id
    observation, info = env.reset(seed=0)
    policy.reset() # 清理上一步的余块
    inference_count = 0
    # 这里放入你之前写的单回合循环，最多 280 步
    # 每一步要使用新 observation，直到 terminated 或 truncated
    for step in range(1, 281):
        chunk_step = (step - 1) % 5
        if chunk_step == 0:
            integration_obs = canonical_observation(observation)
            integration_obs['task'] = env.task_description
            camera1 = torch.from_numpy(
                integration_obs["observation.images.image"]
            ).permute(2, 0, 1).float() / 255.0
            camera2 = torch.from_numpy(
                integration_obs["observation.images.image2"]
            ).permute(2, 0, 1).float() / 255.0
            sample = {"observation.state": torch.from_numpy(
                integration_obs["observation.state"]),
                "observation.images.camera1": camera1,
                "observation.images.camera2": camera2,
                "task": integration_obs['task']}
            processed = preprocessor(sample)
            action_chunk = policy.predict_action_chunk(processed)
            inference_count += 1
        select_action = action_chunk[:, chunk_step, :]
        final_action = postprocessor(select_action)

        action = final_action[0].cpu().numpy()

        action = np.clip(action, env.action_space.low, env.action_space.high)
        next_observation, reward, terminated, truncated, info = env.step(action)
        print("步数：", step)
        print("任务成功：", info["is_success"])
        next_integration_obs = canonical_observation(next_observation)

        observation = next_observation
        if terminated or truncated:
            break
    else:
        print("达到 280 步上限，结束本回合。")
    success = info["is_success"]
    success_count += int(success)

    print("回合：", episode + 1)
    print("初始状态编号：", init_state_index)
    print("执行步数：", step)
    print("是否成功：", success)
    print("策略调用次数", inference_count)
print("成功数：", success_count, "/", num_episodes)


env.close()
