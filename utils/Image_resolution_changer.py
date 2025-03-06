"""
This script converts RGB images to infrared (IR) format using a weighted combination of Red and Green channels.
Usage:
    python utils/rgb_to_ir.py --input_folder path/to/images --output_folder path/to/save
"""
import os
import torch
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm
from tkinter import Tk, filedialog

def get_folder_path(prompt):
    Tk().withdraw()  # Hide the root window
    path = filedialog.askdirectory(title=prompt)
    if not path:
        print("No folder selected. Exiting.")
        exit()
    return path

def process_images(input_folder, output_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    blur_levels = [i for i in range(1, 6)]  # Blur intensities
    sharpness_levels = [i for i in range(1, 6)]  # Sharpness intensities
    
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not images:
        print("No images found in the input folder.")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    for idx, img_name in enumerate(tqdm(images, desc="Processing images", unit="image")):
        img_path = os.path.join(input_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        
        # Apply different blur levels
        for i, blur_intensity in enumerate(blur_levels):
            blurred_img = img.filter(ImageFilter.GaussianBlur(blur_intensity))
            output_path = os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}_blur_{i+1}.jpg")
            blurred_img.save(output_path)
        
        # Apply different sharpness levels
        for i, sharpness_intensity in enumerate(sharpness_levels):
            enhancer = ImageEnhance.Sharpness(img)
            sharpened_img = enhancer.enhance(sharpness_intensity)
            output_path = os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}_sharp_{i+1}.jpg")
            sharpened_img.save(output_path)
            
    print("Processing complete. All images saved in", output_folder)
    print("Done!")

if __name__ == "__main__":
    input_folder = get_folder_path("Select Input Folder")
    output_folder = get_folder_path("Select Output Folder")
    process_images(input_folder, output_folder)
