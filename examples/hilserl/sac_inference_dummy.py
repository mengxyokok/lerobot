#!/usr/bin/env python
"""最简脚本：加载SAC模型并使用dummy输入进行推理"""

import torch
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy

# 配置checkpoint路径
checkpoint_path = "outputs/train/2026-01-05/16-33-40_franka_pick_cube_sim_sac_baseline/checkpoints/010000/pretrained_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载训练配置
cfg = TrainRLServerPipelineConfig.from_pretrained(
    f"/home/mxy/robot/lerobot/configs/sim_franka_train.json"
)

# 创建策略并加载权重
cfg.policy.pretrained_path = checkpoint_path
policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
policy.eval()
policy.to(device)

dummy_obs = {
    "observation.images.front": torch.rand(1, 3, 128, 128, device=device),  # (C, H, W) 格式，值在[0,1]
    "observation.images.wrist": torch.rand(1, 3, 128, 128, device=device),  # (C, H, W) 格式
    "observation.state": torch.rand(1, 18, device=device),
}

with torch.no_grad():
    action = policy.select_action(dummy_obs)

print(f"输出动作: {action}, 形状: {action.shape}")
