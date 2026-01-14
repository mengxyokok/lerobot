#!/usr/bin/env python
"""WebSocket客户端（MessagePack）：使用真实环境进行推理"""

import asyncio
import numpy as np
import msgpack
import msgpack_numpy as m
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.rl.gym_manipulator import make_robot_env
from lerobot.policies.sac.configuration_sac import SACConfig
import websockets

# 启用 msgpack-numpy 支持
m.patch()

SERVER_URL = "ws://localhost:5000"

# 加载配置
cfg = TrainRLServerPipelineConfig.from_pretrained("/home/mxy/robot/lerobot/configs/sim_franka_train.json")
env, _ = make_robot_env(cfg.env)
obs, info = env.reset()

async def main():
    global obs, info
    async with websockets.connect(SERVER_URL) as websocket:
        print("已连接到WebSocket服务器（MessagePack）")
        while True:
            # 发送请求（MessagePack自动处理numpy数组）
            request = {"observation": obs}
            request_data = msgpack.packb(request, default=m.encode)
            await websocket.send(request_data)
            
            # 接收响应（MessagePack自动恢复numpy数组）
            response_data = await websocket.recv()
            result = msgpack.unpackb(response_data, object_hook=m.decode)
            action = result.get("action")
            
            if action is None:
                break
            
            # action已经是numpy数组，无需转换
            if not isinstance(action, np.ndarray):
                action = np.array(action, dtype=np.float32)
            
            # env step
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, info = env.reset()

if __name__ == "__main__":
    asyncio.run(main())

