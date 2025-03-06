"""
Splits the dataset into train, validation, and test sets while preserving YOLO annotation files.
Usage:
    python utils/split_dataset.py --input_folder path/to/dataset --output_path path/to/save
"""


import os
import shutil
import random
from tqdm import tqdm

def split_dataset(input_folder, output_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
    
    # Create output directories
    images_dir = os.path.join(output_path, "images")
    labels_dir = os.path.join(output_path, "labels")
    
    for folder in [images_dir, labels_dir]:
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(folder, split), exist_ok=True)
    
    # Collect image and label files
    def collect_files(dataset_path):
        images = [f for f in os.listdir(dataset_path) if f.endswith(".jpeg") or f.endswith(".jpg")]
        labels = [f for f in os.listdir(dataset_path) if f.endswith(".txt")]
        return [(img, img.rsplit(".", 1)[0] + ".txt") for img in images]
    
    dataset = collect_files(input_folder)
    
    # Shuffle dataset
    random.shuffle(dataset)
    
    # Split dataset
    total_files = len(dataset)
    train_split = int(total_files * train_ratio)
    val_split = train_split + int(total_files * val_ratio)
    
    train_files = dataset[:train_split]
    val_files = dataset[train_split:val_split]
    test_files = dataset[val_split:]
    
    # Function to copy files with progress bar
    def copy_files(file_list, split):
        for img, label in tqdm(file_list, desc=f"Copying {split} files", unit="file"):
            src_img = os.path.join(input_folder, img)
            src_label = os.path.join(input_folder, label)
            
            shutil.copy(src_img, os.path.join(images_dir, split, img))
            if os.path.exists(src_label):
                shutil.copy(src_label, os.path.join(labels_dir, split, label))
    
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")
    
    print("Dataset split completed! All images should be copied correctly.")

# Example usage
input_folder = input("Enter the path for the dataset: ")
output_path = input("Enter the output path: ")

split_dataset(input_folder, output_path)
