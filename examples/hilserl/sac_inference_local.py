#!/usr/bin/env python
"""最简脚本：加载SAC模型并使用真实环境进行推理"""

import torch
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.envs.utils import preprocess_observation
import gymnasium as gym
import gym_hil

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

# 创建preprocessor和postprocessor
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg.policy,
    pretrained_path=checkpoint_path,
    dataset_stats=cfg.policy.dataset_stats,
)

# 创建环境
env = gym.make("gym_hil/PandaPickCubeKeyboard-v0", render_mode="human")

# 重置环境
obs, _ = env.reset()

while True:
    # 将环境观察转换为LeRobot格式
    # 环境返回: {"pixels": {"front": ..., "wrist": ...}, "agent_pos": ...}
    # 需要转换为: {"observation.images.front": ..., "observation.images.wrist": ..., "observation.state": ...}
    lerobot_obs = preprocess_observation(obs)
    
    # 应用preprocessor（添加batch维度、移动到设备、归一化等）
    processed_obs = preprocessor(lerobot_obs)
    
    with torch.no_grad():
        # 推理
        action = policy.select_action(processed_obs)
        # 应用postprocessor（反归一化、移动到CPU）
        action = postprocessor(action)
    
    # 将action转换为numpy数组（环境期望numpy格式）
    action_np = action.squeeze(0).cpu().numpy()  # 移除batch维度并转换为numpy
    
    obs, reward, terminated, truncated, info = env.step(action_np)
    if terminated or truncated:
        obs, _ = env.reset()

