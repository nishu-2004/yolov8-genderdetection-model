# YOLOv8 Gender Detection Model

##  Project Overview
This project is a gender classification model trained using YOLOv8. It detects whether a person in an image is male or female. The dataset used includes images from Kaggle and private sources.

##  Dataset
The dataset consists of publicly available images from Kaggle and a private dataset collected from Ramaiah (not included due to confidentiality). The Kaggle dataset can be found here:
[Men & Women Classification Dataset](https://www.kaggle.com/datasets/playlist/men-women-classification)

##  Installation
Before running the scripts, install the required dependencies:
```bash
pip install -r requirements.txt
```

##  Preprocessing & Utilities
Several preprocessing scripts are included in the `utils` folder to enhance the dataset:

###  `rgb_to_ir.py`
Converts RGB images to infrared (IR) using a weighted combination of Red and Green channels.
```bash
python utils/rgb_to_ir.py --input_folder path/to/images --output_folder path/to/save
```

###  `blur_sharpen.py`
Applies different levels of Gaussian blur and sharpening while ensuring YOLO annotation files remain consistent.
```bash
python utils/blur_sharpen.py --input_folder path/to/images --output_folder path/to/save
```

###  `split_dataset.py`
Splits the dataset into train, validation, and test sets while preserving YOLO annotation files.
```bash
python utils/split_dataset.py --input_folder path/to/dataset --output_path path/to/save
```

## Model Training
Train the YOLOv8 model using the preprocessed dataset:
```bash
yolo train model=yolov8n.pt data=dataset.yaml epochs=50 imgsz=640
```

##  Inference & Testing
To test the model on new images:
```bash
yolo predict model=best.pt source=path/to/test/images
```

##  Results & Performance
After training, the model achieves high accuracy in gender detection. More evaluation metrics and results will be added soon.

##  License
This project is for educational purposes only. Some datasets are publicly available, while others remain private.

##  Future Improvements
- Improve dataset quality with more balanced samples
- Fine-tune YOLO model for better generalization
- Deploy the model as a web app or API

---

**Contributions & Suggestions Are Welcome!** If you find this useful or have ideas for improvement, feel free to open an issue or contribute! 

