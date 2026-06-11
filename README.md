# Breast Cancer Detection using Deep Learning

## Overview
This project detects breast cancer from ultrasound images using transfer learning and CNNs.

## Models Used
- MobileNetV2
- EfficientNetB0

## Features
- Binary Classification
- Transfer Learning
- Grad-CAM Explainability
- Evaluation Metrics

## Dataset
Breast Mamogram Images Dataset (BMSI)

## Technologies
- Python
- TensorFlow
- OpenCV
- NumPy
- Matplotlib

## Results
- Test Accuracy: 86%
- Precision: 93% (Malignant)
- Recall: 72% (Malignant)

## Project Structure

```text
dataset/
models/
outputs/
src/
```

## Run

```bash
pip install -r requirements.txt

cd src

python train.py
python evaluate.py
python predict.py
python gradcam.py
