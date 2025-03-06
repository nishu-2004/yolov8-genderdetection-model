import cv2
import numpy as np
import os
from tqdm import tqdm

# Get input and output folder paths from user
input_folder = input("Enter the input folder path: ").strip()
output_folder = input("Enter the output folder path: ").strip()

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Check if there are images to process
if not image_files:
    print("No image files found in the input folder.")
    exit()

# Define IR transformation coefficients
alpha, beta = 0.8, 0.2

# Process images with progress tracking
for image_name in tqdm(image_files, desc="Processing Images", unit="image"):
    image_path = os.path.join(input_folder, image_name)
    output_path = os.path.join(output_folder, image_name)

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {image_name}: Unable to read file.")
        continue

    # Convert BGR to RGB (OpenCV loads in BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract R, G channels
    R, G = image[:, :, 0], image[:, :, 1]

    # Convert to IR using given formula
    IR_image = (alpha * R + beta * G).astype(np.uint8)

    # Save the result
    cv2.imwrite(output_path, IR_image)

print(f"\nProcessing complete! IR images saved in '{output_folder}'.")
