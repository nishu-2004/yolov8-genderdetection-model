# YOLOv8 Gender Detection Model

This project implements a gender detection model using YOLOv8, a state-of-the-art object detection framework. The model detects human faces in images or video streams and predicts their gender (male/female).

Note: This model is optimized for detecting a single face per image/frame and may not perform as well when multiple faces are present.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Ethical Considerations](#ethical-considerations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview
This project leverages YOLOv8 for gender detection. YOLOv8 is a cutting-edge object detection model known for its speed and accuracy. The model is trained to detect human faces and classify them into two categories: Male or Female.

### Applications
- Surveillance systems
- Retail analytics
- Demographic studies
- Human-computer interaction

## Features
- **Real-time gender detection**: Can process images and video streams in real-time.
- **High accuracy**: Utilizes YOLOv8's state-of-the-art object detection capabilities.
- **Easy to use**: Simple setup and inference scripts.
- **Customizable**: Can be fine-tuned on your own dataset.

## Installation
### Prerequisites
- Python 3.8 or higher
- CUDA (optional, for GPU acceleration)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/nishu-2004/yolov8-genderdetection-model.git
   cd yolov8-genderdetection-model
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the pre-trained YOLOv8 weights (if not training from scratch):
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   ```

## Usage
### Inference on Images
To detect gender in an image:
```bash
python detect.py --source image.jpg --weights yolov8n.pt
```

### Inference on Video
To detect gender in a video:
```bash
python detect.py --source video.mp4 --weights yolov8n.pt
```

### Real-Time Webcam Inference
To run gender detection on a webcam feed:
```bash
python detect.py --source 0 --weights yolov8n.pt
```

## Dataset
The model is trained on a combination of publicly available and internally collected datasets.

### Dataset Sources
- [Men & Women Classification Dataset](https://www.kaggle.com/datasets/playlist/men-women-classification)
- A private dataset collected as part of a previous project at **Ramaiah's CIT Lab** (not publicly available).

### Dataset Structure
```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## Training
To train the model on your own dataset:
1. Prepare your dataset in YOLO format (see Dataset section).
2. Update the `data.yaml` file with your dataset paths.
3. Run the training script:
   ```bash
   python train.py --data data.yaml --weights yolov8n.pt --epochs 50
   ```

## Evaluation
The model's performance is evaluated using standard metrics:
- **Precision**
- **Recall**
- **F1 Score**
- **mAP (mean Average Precision)**

To evaluate the model:
```bash
python val.py --data data.yaml --weights yolov8n.pt
```

## Results
### Performance Metrics
| Metric    | Value |
|-----------|------|
| Precision | 0.975 |
| Recall    | 0.968 |
| F1 Score  | 0.984 |
| mAP@0.5   | 0.903 |

### Sample Outputs
Example of gender detection on an image.

## Ethical Considerations
Gender detection models can raise ethical concerns, such as:
- **Bias**: The model may perform differently across different demographics.
- **Privacy**: Use of such models in public spaces may infringe on privacy rights.

We recommend:
- Using the model responsibly and transparently.
- Regularly auditing the model for bias and fairness.

## Contributing
Contributions are welcome! If you'd like to contribute, please:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- **Ultralytics** for the YOLOv8 framework.
- **Kaggle** for providing datasets for model training.
- **Ramaiah's CIT Lab** for contributing to the dataset collection in a previous project.
- Contributors to the dataset used for training.

---
Let me know if you need further modifications! 🚀

