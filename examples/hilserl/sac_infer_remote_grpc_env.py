#!/usr/bin/env python
"""gRPC客户端（MessagePack）：使用真实环境进行推理

使用方法:
1. 安装依赖: pip install grpcio grpcio-tools msgpack msgpack-numpy
2. 生成proto文件: python -m grpc_tools.protoc -I examples/hilserl --python_out=examples/hilserl --grpc_python_out=examples/hilserl examples/hilserl/sac_inference.proto
3. 运行客户端: python sac_infer_remote_grpc_env.py
"""

import numpy as np
import msgpack
import msgpack_numpy as m
import grpc
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.rl.gym_manipulator import make_robot_env
from lerobot.policies.sac.configuration_sac import SACConfig

# 启用 msgpack-numpy 支持
m.patch()

# 导入生成的proto文件
try:
    import sac_inference_pb2
    import sac_inference_pb2_grpc
except ImportError:
    print("错误: 请先生成proto文件:")
    print("python -m grpc_tools.protoc -I examples/hilserl --python_out=examples/hilserl --grpc_python_out=examples/hilserl examples/hilserl/sac_inference.proto")
    exit(1)

SERVER_URL = "localhost:5000"

# 加载配置
cfg = TrainRLServerPipelineConfig.from_pretrained("/home/mxy/robot/lerobot/configs/sim_franka_train.json")
env, _ = make_robot_env(cfg.env)
obs, info = env.reset()

# 创建gRPC通道和存根
channel = grpc.insecure_channel(SERVER_URL)
stub = sac_inference_pb2_grpc.SACInferenceStub(channel)
print("已连接到gRPC服务器（MessagePack）")

while True:
    # 发送请求（MessagePack自动处理numpy数组）
    request_dict = {"observation": obs}
    request_data = msgpack.packb(request_dict, default=m.encode)
    request = sac_inference_pb2.ObservationRequest(observation_data=request_data)
    
    # 调用gRPC服务
    response = stub.Predict(request)
    
    if response.error:
        print(f"服务器错误: {response.error}")
        break
    
    # 接收响应（MessagePack自动恢复numpy数组）
    result = msgpack.unpackb(response.action_data, object_hook=m.decode)
    action = result.get("action")
    
    # action已经是numpy数组，无需转换
    if not isinstance(action, np.ndarray):
        action = np.array(action, dtype=np.float32)
    
    # env step
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

