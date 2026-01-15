#!/usr/bin/env python
"""gRPC服务器（MessagePack）：加载SAC模型并提供推理服务

使用方法:
1. 安装依赖: pip install grpcio grpcio-tools msgpack msgpack-numpy
2. 生成proto文件: python -m grpc_tools.protoc -I examples/hilserl --python_out=examples/hilserl --grpc_python_out=examples/hilserl examples/hilserl/sac_inference.proto
3. 运行服务器: python sac_infer_remote_grpc_server.py
4. 客户端连接到 localhost:5000
"""

import torch
import numpy as np
import msgpack
import msgpack_numpy as m
from concurrent import futures
import grpc
from lerobot.processor.converters import create_transition
from lerobot.processor.core import TransitionKey
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.configuration_sac import SACConfig
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


class SACInferenceServicer(sac_inference_pb2_grpc.SACInferenceServicer):
    """SAC推理服务实现"""

    def Predict(self, request, context):
        """处理预测请求"""
        try:
            # 解析MessagePack数据（自动处理numpy数组）
            data = msgpack.unpackb(request.observation_data, object_hook=m.decode)
            obs = data.get("observation", {})

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

            # 转换为MessagePack（自动处理numpy数组）
            response_dict = {"action": processed_action}
            action_data = msgpack.packb(response_dict, default=m.encode)

            return sac_inference_pb2.ActionResponse(action_data=action_data)

        except Exception as e:
            import traceback
            error_msg = f"处理错误: {e}\n{traceback.format_exc()}"
            print(error_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return sac_inference_pb2.ActionResponse(error=str(e))


def serve():
    """启动gRPC服务器"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sac_inference_pb2_grpc.add_SACInferenceServicer_to_server(
        SACInferenceServicer(), server
    )
    server.add_insecure_port("[::]:5000")
    server.start()
    print("gRPC服务器启动（MessagePack）: localhost:5000")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()

