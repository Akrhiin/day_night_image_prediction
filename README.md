# Day/Night Image Classifier

A machine learning model: classifies images as day or night scenes, providing probability scores for each classification. Built with scikit-learn and OpenCV.

## Features

- Image preprocessing and standardization
- Random Forest classification model
- Probability scores for day/night prediction
- Support for batch processing
- Comprehensive evaluation metrics
- Easy-to-use inference pipeline

## Requirements

```
Python 3.8+
numpy>=1.19.2
scikit-learn>=0.24.2
opencv-python>=4.5.3
```

## Installation

1. Clone repository:
```bash
git clone https://github.com/yourusername/day-night-classifier.git
cd day-night-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training a New Model

```python
from day_night_classifier import train_day_night_classifier

# Train the model
model, scaler, report = train_day_night_classifier(
    day_folder='path/to/day/images',
    night_folder='path/to/night/images'
)

# Print classification report
print(report)
```

### Classifying Images

```python
from day_night_classifier import classify_image

# Classify a single image
result = classify_image(
    image_path='path/to/image.jpg',
    model=model,
    scaler=scaler
)

print(f"Day Probability: {result['day_probability']:.2%}")
print(f"Night Probability: {result['night_probability']:.2%}")
```

## Project Structure

```
py_day_night_predictor/
├── day_images_training/
│   └── *put all day training images here*
├── night_images_training/
│   └── *put all night training images here*
├── images_testing/
│   └── *put mixed testing image set here to be selected at random by driver*
├── day_night_classifier.py
└── README.md
```

## Dataset

The model requires a dataset of day and night images organized in separate folders. Recommended dataset size:
- Minimum: 1000 images per class
- Recommended: 5000+ images per class

## Model Performance

Current model achieves:
- 94% overall accuracy
- 0.95 precision for day classification
- 0.93 precision for night classification
- 0.94 recall for day classification
- 0.95 recall for night classification

## Known Issues

- Reduced accuracy during sunset/sunrise periods
- May struggle with heavily artificially lit night scenes
- Performance variation with extreme weather conditions

## Aknowledgements

- scikit-learn team for the machine learning framework
- OpenCV team for image processing capabilities
- Dataset providers and contributors
