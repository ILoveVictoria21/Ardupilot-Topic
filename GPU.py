import torch

print("CUDA 可用嗎:", torch.cuda.is_available())
print("使用的 GPU:", torch.cuda.get_device_name(0))
print("CUDA 版本:", torch.version.cuda)
print("cuDNN 版本:", torch.backends.cudnn.version())    