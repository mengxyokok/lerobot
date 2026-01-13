#!/usr/bin/env python
"""最简脚本：加载SAC模型并提供HTTP推理服务

使用方法:
1. 安装依赖: pip install flask
2. 运行服务器: python sac_inference_server.py
3. 发送POST请求到 http://localhost:5000/predict

请求格式 (JSON):
{
    "observation.images.front": [[[0.0, 0.0, 0.0], ...], ...],  # (H, W, C) 格式，值在[0, 255]
    "observation.images.wrist": [[[0.0, 0.0, 0.0], ...], ...],
    "observation.state": [0.0, 0.0, ...]  # 状态向量
}

响应格式 (JSON):
{
    "action": [0.0, 0.0, ...]  # 动作向量
}
"""

import torch
import numpy as np


from lerobot.processor.converters import create_transition
from lerobot.processor.core import TransitionKey
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    EnvTransition,
    GripperPenaltyProcessorStep,
    ImageCropResizeProcessorStep,
    InterventionActionProcessorStep,
    JointVelocityProcessorStep,
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
    MotorCurrentProcessorStep,
    Numpy2TorchActionProcessorStep,
    RewardClassifierProcessorStep,
    RobotActionToPolicyActionProcessorStep,
    TimeLimitProcessorStep,
    Torch2NumpyActionProcessorStep,
    TransitionKey,
    VanillaObservationProcessorStep,
    create_transition,
)
from lerobot.processor.converters import identity_transition

from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    """接收观察值，返回动作"""
    try:
        # 获取JSON数据
        data = request.json
        obs_dict, info_dict = data["observation"], data["info"]
        # 转换为numpy数组
        obs = {}
        for key, value in obs_dict.items():
            obs[key] = np.array(value, dtype=np.float32)

        # list转换为tensor
        for name in obs:
            obs[name] = torch.from_numpy(obs[name])
            # if "image" in name:
            #     obs[name] = obs[name].type(torch.float32) / 255.0  # 归一化到[0,1]
            #     obs[name] = obs[name].permute(2, 0, 1).contiguous()  # (H,W,C) -> (C,H,W)
            obs[name] = obs[name].unsqueeze(0)  # 添加batch维度
            obs[name] = obs[name].to(device)

        # 处理观察
        transition = create_transition(observation=obs, info={})
        transition = env_processor(transition)
        processed_obs = transition[TransitionKey.OBSERVATION]

        # 推理
        with torch.no_grad():
            action = policy.select_action(processed_obs)

        # 创建动作transition并处理
        action_transition = create_transition(action=action)
        processed_action_transition = action_processor(action_transition)
        processed_action = processed_action_transition[TransitionKey.ACTION]

        # numpy转换为list
        action = processed_action.tolist()

        return jsonify({"action": action})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """健康检查端点"""
    return jsonify({"status": "ok", "device": str(device)})


# 配置checkpoint路径
checkpoint_path = "outputs/train/2026-01-05/16-33-40_franka_pick_cube_sim_sac_baseline/checkpoints/010000/pretrained_model"

# 加载训练配置
cfg = TrainRLServerPipelineConfig.from_pretrained(
    "/home/mxy/robot/lerobot/configs/sim_franka_train.json"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建策略并加载权重
cfg.policy.pretrained_path = checkpoint_path
policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
policy.eval()
policy.to(device)
print(f"模型已加载到设备: {device}")

env_steps = [
    Torch2NumpyActionProcessorStep(),
]
env_processor = DataProcessorPipeline(
    steps=env_steps, to_transition=identity_transition, to_output=identity_transition
)
action_steps = [
    Numpy2TorchActionProcessorStep(),
    VanillaObservationProcessorStep(),
    AddBatchDimensionProcessorStep(),
    DeviceProcessorStep(device=device),
]
action_processor = DataProcessorPipeline(
    steps=action_steps, to_transition=identity_transition, to_output=identity_transition
)
env_processor.reset()
action_processor.reset()
print("模型加载完成，启动服务器...")

print("API端点:")
print("  POST /predict - 接收观察值，返回动作")
print("  GET  /health  - 健康检查")
app.run(host="0.0.0.0", port=5000, debug=False)
