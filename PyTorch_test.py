import torch
import time
print("CUDA是否可用:",torch.cuda.is_available())  # 如果返回 True，说明 CUDA 可用
#如果 torch.version.cuda 返回 None，说明 PyTorch 没有找到 CUDA 支持，尝试 降级 CUDA 到 12.1 或 11.8 并重新安装 PyTorch。
print("CUDA版本:",torch.version.cuda)  # 这里的 CUDA 版本应该和 nvcc 的一致
print("PyTorch版本:",torch.__version__)  # 检查 PyTorch 版本,
#如果 PyTorch 版本后面没有 +cuXXX（如 +cu121），说明你安装的是 CPU 版本。
print("是否启用了cuDNN:",torch.backends.cudnn.enabled)  # 是否启用了 cuDNN
if torch.cuda.is_available():
    print("PyTorch 使用的 GPU:", torch.cuda.get_device_name(0))
else:
    print("没有检测到 CUDA 设备")

#---Test CUDA---

#1）
# 在 GPU 上创建一个 Tensor
device = torch.device("cuda")
x = torch.randn(3, 3).to(device)
print(x)  # 看看这个 Tensor 是否成功放到 GPU 上

#2）
# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用的设备:", device)

# 在 CPU 和 GPU 上分别进行矩阵乘法测试
size = 100  # 设定矩阵大小

# CPU 计算
a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)
start_time = time.time()
result_cpu = torch.mm(a_cpu, b_cpu)  # CPU 上的矩阵乘法
cpu_time = time.time() - start_time
print(f"CPU 计算时间: {cpu_time:.6f} 秒")

# GPU 计算
a_gpu = torch.randn(size, size, device=device)
b_gpu = torch.randn(size, size, device=device)
torch.cuda.synchronize()  # 确保 GPU 计算前的同步
start_time = time.time()
result_gpu = torch.mm(a_gpu, b_gpu)  # GPU 上的矩阵乘法
torch.cuda.synchronize()  # 确保 GPU 计算完成
gpu_time = time.time() - start_time
print(f"GPU 计算时间: {gpu_time:.6f} 秒")

# 对比 CPU 和 GPU 计算速度
if torch.cuda.is_available():
    speedup = cpu_time / gpu_time
    print(f"GPU 比 CPU 快 {speedup:.2f} 倍")