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
    "success": true,
    "action": [0.0, 0.0, ...]  # 动作向量
}
"""

import torch
import numpy as np
from flask import Flask, request, jsonify
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy

app = Flask(__name__)

# 全局变量存储模型和处理器
policy = None
device = None
cfg = None


def load_model():
    """加载SAC模型和处理器"""
    global policy, device, cfg
    
    # 配置checkpoint路径
    checkpoint_path = "outputs/train/2026-01-05/16-33-40_franka_pick_cube_sim_sac_baseline/checkpoints/010000/pretrained_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载训练配置
    cfg = TrainRLServerPipelineConfig.from_pretrained(
        "/home/mxy/robot/lerobot/configs/sim_franka_train.json"
    )
    
    # 创建策略并加载权重
    cfg.policy.pretrained_path = checkpoint_path
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy.eval()
    policy.to(device)
    print(f"模型已加载到设备: {device}")


@app.route('/predict', methods=['POST'])
def predict():
    """接收观察值，返回动作"""
    try:
        # 获取JSON数据
        data = request.json
        obs, info = data['observation'], data['info']
 
        # 推理
        with torch.no_grad():
            action = policy.select_action(obs)
        
        return jsonify({
            'success': True,
            'action': action
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """健康检查端点"""
    return jsonify({'status': 'ok', 'device': str(device)})


if __name__ == '__main__':
    print("正在加载模型...")
    load_model()
    print("模型加载完成，启动服务器...")
    print("API端点:")
    print("  POST /predict - 接收观察值，返回动作")
    print("  GET  /health  - 健康检查")
    app.run(host='0.0.0.0', port=5000, debug=False)
