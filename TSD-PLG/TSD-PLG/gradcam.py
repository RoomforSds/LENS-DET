import cv2
import numpy as np
import os
from ultralytics import YOLO
from yolo11_gradcam.yolo_cam.eigen_cam import EigenCAM
from yolo11_gradcam.yolo_cam.utils.image import show_cam_on_image

# Load a model
model = YOLO(r"j:\yolo11n.pt")
target_layers = [model.model.model[-6]]

# Set the folder paths
input_folder = r"j:\1"
output_folder = r"j:\2"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the list of image files in the folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Process each image in the folder
for image_file in image_files:
    # Read the image
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    # Resize and preprocess the image
    image = cv2.resize(image, (640, 640))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_img = image.copy()
    image = np.float32(image) / 255
