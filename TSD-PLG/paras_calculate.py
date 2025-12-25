from ultralytics import YOLO

fold_name = "exp21"
# 加载YOLO模型
model = YOLO(f'run/train/{fold_name}/weights/best.pt')

# 获取模型的参数数量
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# 打印参数数量到控制台
print(f"Number of parameters in YOLOv11 model: {num_params}")

# 将结果写入txt文件
output_file = f"run/train/{fold_name}/yolov11_params.txt"
with open(output_file, "w") as file:
    file.write(f"Number of parameters in YOLOv11 model: {num_params}\n")

print(f"Model parameter count saved to {output_file}")
