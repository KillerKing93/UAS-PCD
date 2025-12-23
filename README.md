# Medical Image Classification Project (UAS PCD)

## Overview
This project focuses on the classification of medical images into **Benign** and **Malignant** categories using classic Machine Learning techniques. It employs texture analysis via Gray Level Co-occurrence Matrix (GLCM) and uses a Support Vector Machine (SVM) for classification.

## Project Structure
- `DATASETS/`: Contains the image datasets (Test and Train).
- `classify.py`: Main Python script for processing images, extracting features, and training the model.
- `REPORT.md`: Detailed analysis of the classification results.
- `ARCHITECTURE.md`: Technical explanation of the system pipeline.
- `requirements.txt`: List of Python dependencies.

## Requirements
- Python 3.8+
- OpenCV
- Scikit-learn
- Scikit-image
- Pandas
- Matplotlib
- Seaborn

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/KillerKing93/UAS-PCD.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the classification script:
```bash
python classify.py
```
This will:
1. Load images from `DATASETS/DATASET 1`.
2. Preprocess images (Resize, Grayscale).
3. Extract GLCM features.
4. Train an SVM classifier.
5. Evaluate on the Test set.
6. Save the confusion matrix plot as `confusion_matrix.png` and output results to the console.

## Results
Current model performance (Dataset 1):
- **Accuracy:** ~73.45%
- **Sensitivity:** ~72.33%
- **Specificity:** ~74.80%

See `REPORT.md` for a full analysis.
