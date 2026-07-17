# Hybrid Deepfake Detection

## Overview

This project presents a hybrid deepfake detection framework that combines EfficientNet-B3 for spatial feature extraction with a Temporal Convolutional Network (TCN) for temporal feature learning. The system is designed to accurately distinguish between real and manipulated videos while remaining computationally efficient.

This project was developed as part of the B.Tech Capstone Project in Computer Science and Engineering (Cyber Security) at VIT-AP University.

---

## Features

- EfficientNet-B3 backbone for spatial feature extraction
- Temporal CNN (TCN) for temporal sequence analysis
- MTCNN for face detection and alignment
- Exponential Moving Average (EMA) for stable training
- Focal Loss for handling class imbalance
- Automatic preprocessing and face caching
- Performance evaluation with confusion matrix and classification report

---

## Dataset

- **Dataset:** FaceForensics++
- **Classes:** Real and Deepfake Videos
- **Frames Extracted per Video:** 16
- **Input Size:** 224 × 224 pixels

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- MTCNN (facenet-pytorch)
- EfficientNet-PyTorch
- NumPy
- Scikit-learn
- Matplotlib

---

## Project Structure

```
Hybrid-Deepfake-Detection/
│
├── train.py
├── preprocess_faces.py
├── Project_Report.pdf
├── README.md
└── requirements.txt
```

---

## Workflow

1. Load real and fake videos
2. Detect and crop faces using MTCNN
3. Extract spatial features using EfficientNet-B3
4. Learn temporal patterns using Temporal CNN
5. Classify videos as Real or Fake
6. Evaluate using accuracy, confusion matrix, and classification report

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AKSHAYAKORRAPATI1712/Hybrid-Deepfake-Detection.git
cd Hybrid-Deepfake-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Preprocess the dataset:

```bash
python preprocess_faces.py
```

Train the model:

```bash
python train.py
```

---

## Results

The project includes:

- Training and Validation Accuracy
- Training and Validation Loss
- Confusion Matrix
- Classification Report
- Saved Best Model

---

## Future Improvements

- Support additional deepfake datasets
- Improve temporal feature extraction using Transformers
- Deploy as a web application
- Optimize for real-time video inference

---

## Author

**Akshaya Korrapati**

B.Tech Computer Science and Engineering (Cyber Security)

VIT-AP University

---

## License

This project is intended for educational and research purposes.
