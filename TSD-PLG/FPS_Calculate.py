import time
from ultralytics import YOLO
import os
import torch

fold_name = "exp21"
# 加载YOLO模型
model = YOLO(f'run/train/{fold_name}/weights/best.pt')

# 测试图像路径
test_image_folder = 'ultralytics/data/datasets/Lens1.1/test/images'  # 假设你的测试图像存放在此文件夹中

# 获取图像列表
test_images = [os.path.join(test_image_folder, img) for img in os.listdir(test_image_folder) if
               img.endswith(('.jpg', '.png'))]

# 记录开始时间
start_time = time.time()

# 进行推理
num_images = len(test_images)
for img_path in test_images:
    results = model.predict(source=img_path, save=True)  # 使用save=True来保存预测结果

# 记录结束时间
end_time = time.time()

# 计算推理时间和FPS
elapsed_time = end_time - start_time
fps = num_images / elapsed_time  # FPS = 总图像数 / 总时间

# 输出结果到控制台
print(f"Processed {num_images} images in {elapsed_time:.2f} seconds. FPS: {fps:.2f}")

# 将结果写入txt文件
output_file = f"run/train/{fold_name}/fps_results.txt"
with open(output_file, "w") as file:
    file.write(f"Processed {num_images} images in {elapsed_time:.2f} seconds.\n")
    file.write(f"FPS: {fps:.2f}\n")

print(f"Results saved to {output_file}")

