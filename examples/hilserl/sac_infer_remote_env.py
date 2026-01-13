#!/usr/bin/env python
"""最简脚本：加载SAC模型并使用真实环境进行推理"""

import torch

import gymnasium as gym
import gym_hil

from lerobot.rl.gym_manipulator import make_processors, make_robot_env
from lerobot.processor.converters import create_transition
from lerobot.processor.core import TransitionKey
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy

import requests
import numpy as np


SERVER_URL = "http://localhost:5000"


def request_predict(obs_dict, info_dict={}):
    """发送预测请求到服务器"""
    response = requests.post(
        f"{SERVER_URL}/predict",
        json={"observation": obs_dict, "info": info_dict or {}},
        timeout=100
    )
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            return np.array(result.get("action"), dtype=np.float32)
    return None


# 加载配置
cfg = TrainRLServerPipelineConfig.from_pretrained(
    f"/home/mxy/robot/lerobot/configs/sim_franka_train.json"
)

# 创建环境、环境处理器和动作处理器
env, teleop_device = make_robot_env(cfg.env)

# 环境重置
obs, info = env.reset()

while True:
     # 处理观察: numpy转list
    obs_dict = obs.tolist()
    
    # 请求推理
    action_dict = request_predict(obs_dict)

    # 处理action：list转numpy
    action = np.array(action_dict, dtype=np.float32)

    # env step
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
    

