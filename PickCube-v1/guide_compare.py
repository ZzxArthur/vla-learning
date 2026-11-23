import gymnasium as gym
import mani_skill.envs
import torch

from point_guide import point_guide


env = gym.make(
    "PickCube-v1",
    obs_mode="state_dict",
    control_mode="pd_ee_delta_pos",
    sim_backend="physx_cpu",
    render_mode=None,
    max_episode_steps=150,
)

try:
    initial_positions = []
    for mode in ("feedback", "fixed_velocity"):
        obs, _ = env.reset(seed=42)
        root_quaternion = env.unwrapped.agent.robot.pose.q[0]
        assert torch.allclose(root_quaternion, root_quaternion.new_tensor([1, 0, 0, 0]), atol=1e-5)
        initial_tcp = obs["extra"]["tcp_pose"][0, :3].clone()
        initial_positions.append(initial_tcp)
        target = initial_tcp + initial_tcp.new_tensor([0.05, 0.0, 0.05])
        dt = 1.0 / env.unwrapped.control_freq
        initial_velocity = point_guide(initial_tcp, target, 1.0, 0.1)
        print(f"\n{mode}: 初始误差={torch.linalg.vector_norm(target - initial_tcp).item():.4f} m")

        for step in range(107):
            tcp = obs["extra"]["tcp_pose"][0, :3]
            velocity = point_guide(tcp, target, 1.0, 0.1) if mode == "feedback" else initial_velocity
            arm_action = velocity * dt / 0.1
            action = torch.cat((arm_action, arm_action.new_tensor([1.0])))
            obs, _, terminated, truncated, _ = env.step(action)
            if (step + 1) % 25 == 0 or step == 106:
                error = torch.linalg.vector_norm(target - obs["extra"]["tcp_pose"][0, :3]).item()
                print(f"step={step + 1:03d}, 位置误差={error:.4f} m")
            if terminated.any() or truncated.any():
                print(f"环境在第 {step + 1} 步提前结束")
                break

    assert torch.allclose(initial_positions[0], initial_positions[1], atol=1e-6)
finally:
    env.close()

# 本文件主要代码由 AI 助手完成。
