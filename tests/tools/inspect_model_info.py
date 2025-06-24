from safetensors.torch import load_file

# 加载 safetensors 文件
tensors = load_file("outputs/train/act_franka_libero_0612/checkpoints/060000/pretrained_model/model.safetensors")

# 打印所有参数名和形状
total_params = 0
for name, tensor in tensors.items():
    num = tensor.numel()
    print(f"{name}: {tensor.shape} {tensor.dtype} 参数量: {num}")
    total_params += num

print(f"模型总参数量: {total_params}")