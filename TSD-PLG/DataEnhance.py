import os
import cv2
import albumentations as A
import numpy as np
import random

# 原始路径
image_folder = r'C:/Users/SDS/Desktop/Scientific research/code/ultralytics-main/ultralytics/data/datasets/lens/valid/images'
label_folder = r'C:/Users/SDS/Desktop/Scientific research/code/ultralytics-main/ultralytics/data/datasets/lens/valid/labels'

# 保存路径
aug_image_folder = r'C:/Users/SDS/Desktop/Scientific research/code/ultralytics-main/ultralytics/data/datasets/lens-1/valid/images'
aug_label_folder = r'C:/Users/SDS/Desktop/Scientific research/code/ultralytics-main/ultralytics/data/datasets/lens-1/valid/labels'

# 创建保存目录
os.makedirs(aug_image_folder, exist_ok=True)
os.makedirs(aug_label_folder, exist_ok=True)

# 定义数据增强
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(p=0.5),
    A.Resize(640, 640)  # 假设输出尺寸为 640x640
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 获取图像和标签文件列表
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
label_files = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in image_files]

# 总共需要1200张图像，计算循环次数
target_image_count = 200
current_image_count = 0

# 数据增强循环，直到达到目标图像数
while current_image_count < target_image_count:
    for image_name, label_name in zip(image_files, label_files):
        image_path = os.path.join(image_folder, image_name)
        label_path = os.path.join(label_folder, label_name)

        # 读取图像
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        # 读取 YOLO 标签
        bboxes = []
        class_labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                label = line.strip().split()
                class_id = int(label[0])
                x_center, y_center, width, height = map(float, label[1:])
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(class_id)

        # 进行增强
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_class_labels = augmented['class_labels']

        # 保存增强后的图像
        aug_image_path = os.path.join(aug_image_folder, f"{current_image_count}.jpg")
        cv2.imwrite(aug_image_path, aug_image)

        # 保存增强后的标签
        aug_label_path = os.path.join(aug_label_folder, f"{current_image_count}.txt")
        with open(aug_label_path, 'w') as f:
            for bbox, class_id in zip(aug_bboxes, aug_class_labels):
                x_center, y_center, width, height = bbox
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        current_image_count += 1
        if current_image_count >= target_image_count:
            break

print("数据增强完成！")
