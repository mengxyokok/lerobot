#!/usr/bin/env python
"""WebSocket服务器：加载SAC模型并提供推理服务

使用方法:
1. 安装依赖: pip install websockets
2. 运行服务器: python sac_infer_remote_server_websocket.py
3. 客户端连接到 ws://localhost:5000
"""

import asyncio
import json
import torch
import numpy as np
from lerobot.processor.converters import create_transition
from lerobot.processor.core import TransitionKey
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    Numpy2TorchActionProcessorStep,
    Torch2NumpyActionProcessorStep,
    VanillaObservationProcessorStep,
    create_transition,
)
from lerobot.processor.converters import identity_transition
import websockets



# 加载模型
checkpoint_path = "outputs/train/2026-01-05/16-33-40_franka_pick_cube_sim_sac_baseline/checkpoints/010000/pretrained_model"
cfg = TrainRLServerPipelineConfig.from_pretrained("/home/mxy/robot/lerobot/configs/sim_franka_train.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg.policy.pretrained_path = checkpoint_path
policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
policy.eval()
policy.to(device)
print(f"模型已加载到设备: {device}")

env_steps = [
    Numpy2TorchActionProcessorStep(),
    VanillaObservationProcessorStep(),
    AddBatchDimensionProcessorStep(),
    DeviceProcessorStep(device=device),
]
env_processor = DataProcessorPipeline(
    steps=env_steps, to_transition=identity_transition, to_output=identity_transition
)
action_steps = [Torch2NumpyActionProcessorStep()]
action_processor = DataProcessorPipeline(
    steps=action_steps, to_transition=identity_transition, to_output=identity_transition
)
env_processor.reset()
action_processor.reset()

async def handle_method(websocket, path):
    """处理WebSocket客户端连接"""
    print(f"客户端连接: {websocket.remote_address}")
    try:
        async for message in websocket:
            try:
                # 解析JSON数据
                data = json.loads(message)
                obs_json = data.get("observation", {})

                # 转换为numpy数组
                obs = {}
                if "pixels" in obs_json:
                    pixels = obs_json["pixels"]
                    obs["pixels"] = {}
                    if isinstance(pixels, dict):
                        for key in ["front", "wrist"]:
                            if key in pixels:
                                obs["pixels"][key] = np.array(pixels[key], dtype=np.uint8)

                if "agent_pos" in obs_json:
                    obs["agent_pos"] = np.array(obs_json["agent_pos"], dtype=np.float32)

                # 处理观察
                transition = create_transition(observation=obs, info={})
                transition = env_processor(transition)
                processed_obs = transition[TransitionKey.OBSERVATION]

                # 推理
                with torch.no_grad():
                    action = policy.select_action(processed_obs)

                # 处理动作
                action_transition = create_transition(action=action)
                processed_action_transition = action_processor(action_transition)
                processed_action = processed_action_transition[TransitionKey.ACTION]

                # 转换为JSON
                response = {"action": processed_action.tolist()}
                await websocket.send(json.dumps(response))

            except Exception as e:
                print(f"处理错误: {e}")
                await websocket.send(json.dumps({"error": str(e)}))

    except websockets.exceptions.ConnectionClosed:
        print(f"客户端断开: {websocket.remote_address}")


async def main():
    async with websockets.serve(handle_method, "0.0.0.0", 5000):
        print("WebSocket服务器启动: ws://0.0.0.0:5000")
        await asyncio.Future()  # 永久运行

asyncio.run(main())

