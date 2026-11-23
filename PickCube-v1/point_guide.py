import torch


def point_guide(now_point, target_point, kp, max_speed):
    v = kp * (target_point - now_point)
    speed = torch.linalg.vector_norm(v)
    if speed > max_speed:
        v = v * (max_speed / speed)
    return v
