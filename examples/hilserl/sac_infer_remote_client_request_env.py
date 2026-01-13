#!/usr/bin/env python
"""SAC推理客户端：使用PandaPickCubeGymEnv环境"""

import requests
import numpy as np
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.rl.gym_manipulator import make_robot_env

SERVER_URL = "http://localhost:5000"


def convert_env_obs_to_server_format(obs):
    """将环境观察转换为服务器期望的格式"""
    server_obs = {}
    
    if "pixels" in obs and isinstance(obs["pixels"], dict):
        for key in ["front", "wrist"]:
            if key in obs["pixels"]:
                img = obs["pixels"][key]
                server_obs[f"observation.images.{key}"] = img.tolist() if isinstance(img, np.ndarray) else img
    
    for key in ["observation.images.front", "observation.images.wrist"]:
        if key in obs:
            img = obs[key]
            server_obs[key] = img.tolist() if isinstance(img, np.ndarray) else img
    
    if "agent_pos" in obs:
        agent_pos = obs["agent_pos"]
        server_obs["observation.state"] = agent_pos.tolist() if isinstance(agent_pos, np.ndarray) else agent_pos
    elif "observation.state" in obs:
        state = obs["observation.state"]
        server_obs["observation.state"] = state.tolist() if isinstance(state, np.ndarray) else state
    
    return server_obs


def send_predict_request(obs_dict, info_dict=None):
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


def run_episode(env, max_steps=1000, render=True, delay=0.1):
    """使用远程服务器运行一个episode"""
    obs, info = env.reset()
    total_reward = 0.0
    
    for step in range(max_steps):
        server_obs = convert_env_obs_to_server_format(obs)
        action = send_predict_request(server_obs, info)
        
        if action is None:
            break
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if render:
            env.render()
        
        if terminated or truncated:
            break
    
    return total_reward


if __name__ == "__main__":
    cfg = TrainRLServerPipelineConfig.from_pretrained("/home/mxy/robot/lerobot/configs/sim_franka_train.json")
    env, _ = make_robot_env(cfg.env)
    
    try:
        run_episode(env, max_steps=1000, render=True, delay=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()