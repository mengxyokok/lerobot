#!/bin/bash
# 生成 gRPC Python 代码

cd "$(dirname "$0")"
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. sac_inference.proto

if [ $? -eq 0 ]; then
    echo "Proto文件生成成功！"
    echo "生成的文件: sac_inference_pb2.py, sac_inference_pb2_grpc.py"
else
    echo "错误: 请先安装 grpcio-tools: pip install grpcio-tools"
    exit 1
fi

