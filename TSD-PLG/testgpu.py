import torch

# 获取当前 GPU 的设备 ID
device_id = torch.cuda.current_device()

# 获取当前 GPU 的名称
device_name = torch.cuda.get_device_name(device_id)

print(f"当前使用的 GPU ID: {device_id}")
print(f"当前使用的 GPU 名称: {device_name}")
