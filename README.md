# YOLOv8 Gender Detection Model

## Overview
This repository contains a YOLOv8-based model for gender detection, trained on a combination of publicly available and private datasets. The model has been trained on the **Men-Women Classification** dataset from Kaggle, along with additional data collected from **Ramaiah** (which cannot be disclosed).

## Dataset
### Public Dataset
- **Kaggle Dataset**: [Men-Women Classification](https://www.kaggle.com/datasets/playlist/men-women-classification)

### Private Dataset
- Additional data collected from Ramaiah (not publicly shared due to confidentiality).

## Preprocessing Steps
To enhance the dataset and improve model performance, multiple preprocessing techniques were applied:

### 1. **RGB to Infrared (IR) Conversion**
The RGB images were converted to infrared-like images using the following script:
```python
import cv2
import numpy as np
import os
from tqdm import tqdm

# Get input and output folder paths from user
input_folder = input("Enter the input folder path: ").strip()
output_folder = input("Enter the output folder path: ").strip()

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define IR transformation coefficients
alpha, beta = 0.9, 0.1

# Process images
for image_name in tqdm(os.listdir(input_folder), desc="Processing Images", unit="image"):
    image_path = os.path.join(input_folder, image_name)
    output_path = os.path.join(output_folder, image_name)
    
    image = cv2.imread(image_path)
    if image is None:
        continue
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    R, G = image[:, :, 0], image[:, :, 1]
    IR_image = (alpha * R + beta * G).astype(np.uint8)
    cv2.imwrite(output_path, IR_image)
```

### 2. **Blurring and Sharpening Augmentations**
Blurring and sharpening transformations were applied to improve model generalization:
```python
import os
import torch
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
import shutil

def process_images(input_folder, output_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blur_levels = [1, 2, 3, 4]
    sharpness_levels = [1, 2, 3]
    os.makedirs(output_folder, exist_ok=True)

    for img_name in tqdm(os.listdir(input_folder), desc="Processing images"):
        img_path = os.path.join(input_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        
        # Apply blurring
        for i, blur_intensity in enumerate(blur_levels, start=1):
            blurred_img = img.filter(ImageFilter.GaussianBlur(blur_intensity))
            blurred_img.save(os.path.join(output_folder, f"{img_name}_blur_{i}.jpg"))

        # Apply sharpening
        for i, sharpness_intensity in enumerate(sharpness_levels, start=1):
            enhancer = ImageEnhance.Sharpness(img)
            sharpened_img = enhancer.enhance(sharpness_intensity)
            sharpened_img.save(os.path.join(output_folder, f"{img_name}_sharp_{i}.jpg"))

process_images("input_folder", "output_folder")
```

## Training
- **Model Used**: YOLOv8
- **Framework**: Ultralytics YOLOv8
- **Training Data**: Augmented dataset combining Kaggle and private data
- **Hyperparameters**:
  - Optimizer: Adam
  - Learning Rate: 0.001
  - Epochs: 50
  - Batch Size: 16

## Results
The model was trained on the processed dataset and achieved high accuracy in gender classification. Further fine-tuning and evaluation metrics will be updated soon.

## Repository Structure
```
├── dataset/                      # Processed dataset
├── models/                       # Trained YOLOv8 model
├── runs/                         # Training logs and saved weights
├── scripts/                      # Preprocessing scripts
│   ├── rgb_to_ir.py              # Convert RGB to IR
│   ├── blur_sharpen.py           # Apply blurring and sharpening
├── yolov8-genderdetection.ipynb  # Training notebook
└── README.md                     # Project documentation
```

## Future Work
- Model deployment on cloud platforms
- Real-time inference implementation
- Expanding dataset for better generalization

## Acknowledgments
- Kaggle for providing the initial dataset
- Ramaiah for additional data (confidential)
- Ultralytics for YOLOv8 framework

