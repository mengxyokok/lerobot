#!/usr/bin/env python
"""简化的SAC推理客户端（用于测试）"""

import requests
import numpy as np

# 服务器URL
SERVER_URL = "http://localhost:5000"

# 创建模拟观察数据（用于测试）
def create_dummy_observation():
    """创建模拟观察数据"""
    return {
        "observation.images.front": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8).tolist(),
        "observation.images.wrist": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8).tolist(),
        "observation.state": np.random.randn(18).tolist(),
    }

# 发送预测请求
def test_predict():
    """测试预测功能"""
    obs = create_dummy_observation()
    
    response = requests.post(
        f"{SERVER_URL}/predict",
        json={"observation": obs, "info": {}},
        timeout=10
    )
    
    if response.status_code == 200:
        result = response.json()
        if result.get("success"):
            action = result.get("action")
            print(f"成功获取动作: {action}")
            return np.array(action)
        else:
            print(f"错误: {result.get('error')}")
    else:
        print(f"请求失败: {response.status_code}")
    
    return None

if __name__ == "__main__":
    # 检查健康状态
    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=5)
        print(f"服务器状态: {health.json()}")
    except Exception as e:
        print(f"无法连接到服务器: {e}")
        exit(1)
    
    # 测试预测
    test_predict()