import torch
from safetensors.torch import save_file

# 加载 .pt 格式的模型
pt_model = torch.load("yolov8m.pt", map_location="cpu")

# 保存为 .safetensors 格式
save_file(pt_model, "yolov8m.safetensors")