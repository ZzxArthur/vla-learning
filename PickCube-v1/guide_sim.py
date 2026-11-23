import sys
import time

import gymnasium as gym
import mani_skill.envs
import sapien
import torch

from point_guide import point_guide


show_viewer = "--render" in sys.argv
env = gym.make(
    "PickCube-v1",
    obs_mode="state_dict",
    control_mode="pd_ee_delta_pos",
    sim_backend="physx_cpu",
    render_mode="human" if show_viewer else None,
    max_episode_steps=150,
)

try:
    obs, _ = env.reset(seed=42)
    root_quaternion = env.unwrapped.agent.robot.pose.q[0]
    assert torch.allclose(root_quaternion, root_quaternion.new_tensor([1, 0, 0, 0]), atol=1e-5), (
        "本示例要求机器人根坐标系与世界坐标系的方向一致"
    )
    initial_tcp = obs["extra"]["tcp_pose"][0, :3].clone()
    target = initial_tcp + initial_tcp.new_tensor([0.05, 0.0, 0.05])
    if show_viewer:
        env.unwrapped.goal_site.set_pose(sapien.Pose(p=target.tolist()))
        env.render()
    kp, max_speed = 1.0, 0.1
    dt = 1.0 / env.unwrapped.control_freq

    print("初始 TCP (m):", initial_tcp.tolist())
    print("目标位置 (m):", target.tolist())
    for step in range(150):
        tcp = obs["extra"]["tcp_pose"][0, :3]
        error = torch.linalg.vector_norm(target - tcp).item()
        if step % 10 == 0 or error < 0.01:
            print(f"step={step:02d}, 位置误差={error:.4f} m")
        if error < 0.01:
            break

        velocity = point_guide(tcp, target, kp, max_speed)
        delta_position = velocity * dt
        # Panda 控制器的归一化动作 1 对应该轴 0.1 米的目标增量。
        arm_action = delta_position / 0.1
        action = torch.cat((arm_action, arm_action.new_tensor([1.0])))
        obs, _, terminated, truncated, _ = env.step(action)
        if show_viewer:
            env.render()
            if env.unwrapped.viewer.closed:
                break
            time.sleep(dt)
        if terminated.any() or truncated.any():
            break

    final_tcp = obs["extra"]["tcp_pose"][0, :3]
    print("最终 TCP (m):", final_tcp.tolist())
    print(f"最终位置误差: {torch.linalg.vector_norm(target - final_tcp).item():.4f} m")
    if show_viewer:
        print("关闭仿真窗口即可退出。", flush=True)
        while not env.unwrapped.viewer.closed:
            env.render()
            time.sleep(0.05)
finally:
    env.close()

# 本文件主要代码由 AI 助手完成。
