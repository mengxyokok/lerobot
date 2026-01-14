#!/usr/bin/env python
"""WebSocket客户端：使用真实环境进行推理"""

import asyncio
import json
import numpy as np
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.rl.gym_manipulator import make_robot_env
from lerobot.policies.sac.configuration_sac import SACConfig  # 导入以注册策略类型
import websockets


SERVER_URL = "ws://localhost:5000"


def convert_to_json_serializable(obj):
    """递归地将对象转换为JSON可序列化的格式"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


# 加载配置
cfg = TrainRLServerPipelineConfig.from_pretrained("/home/mxy/robot/lerobot/configs/sim_franka_train.json")
env, _ = make_robot_env(cfg.env)
obs, info = env.reset()

async def main():
    global obs, info
    async with websockets.connect(SERVER_URL) as websocket:
        print("已连接到WebSocket服务器")
        while True:
            # 处理观察: numpy转list
            obs_json = convert_to_json_serializable(obs)
            
            # 发送请求
            request = json.dumps({"observation": obs_json})
            await websocket.send(request)
            
            # 接收响应
            response = await websocket.recv()
            result = json.loads(response)
            action_json = result.get("action")
            
            if action_json is None:
                break
            
            # 处理action：list转numpy
            action = np.array(action_json, dtype=np.float32)
            
            # env step
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, info = env.reset()

asyncio.run(main())

