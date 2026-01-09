#!/usr/bin/env python
"""最简脚本：加载SAC模型并使用真实环境进行推理"""

import torch
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.envs.utils import preprocess_observation
import gymnasium as gym
import gym_hil

from lerobot.rl.gym_manipulator import make_processors, make_robot_env

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


# 创建环境、环境处理器和动作处理器
env, teleop_device = make_robot_env(cfg.env)
env_processor, action_processor = make_processors(env, teleop_device, cfg.env, device)
env_processor.reset()
action_processor.reset()

obs, _ = env.reset()

while True:
    # 将环境观察转换为LeRobot格式
    # 环境返回: {"pixels": {"front": ..., "wrist": ...}, "agent_pos": ...}
    # 需要转换为: {"observation.images.front": ..., "observation.images.wrist": ..., "observation.state": ...}

    processed_obs=env_processor(obs)
    with torch.no_grad():
        # 推理
        action = policy.select_action(processed_obs)

    processed_action=action_processor(action)
    
    obs, reward, terminated, truncated, info = env.step(processed_action)
    if terminated or truncated:
        obs, _ = env.reset()

