# Medical Image Classification (Benign vs Malignant)

**Author:** Alif Nurhidayat (G1A022073)  
**Program:** Informatika, Universitas Bengkulu  
**Repository:** [https://github.com/KillerKing93/UAS-PCD](https://github.com/KillerKing93/UAS-PCD)

## Project Overview
This project implements an advanced "Classic Machine Learning" pipeline to classify dermoscopic skin lesion images into **BENIGN** (Benign) and **MALIGNANT** (Malignant) categories. Unlike standard approaches that treat the whole image equally, this project employs **Digital Hair Removal** and **Automatic Segmentation (ROI)** to focus feature extraction solely on the lesion, significantly improving sensitivity.

## Methodology

### 1. Advanced Preprocessing
-   **Digital Hair Removal:** Morphological BlackHat transformation + Inpainting to remove obstructing hair.
-   **Segmentation:** Otsu's Thresholding on HSV S-Channel to create a binary mask of the lesion.
-   **ROI Extraction:** Features are calculated only from the masked area (Region of Interest).
-   **Enhancement:** CLAHE applied to grayscale ROI for texture clarity.

### 2. ROI-Based Feature Extraction
-   **Color:** Statistical moments (Mean, Std, Skew, Kurtosis) from **HSV** and **CIELAB** color spaces (masked pixels only).
-   **Texture:** 
    -   **GLCM:** Gray Level Co-occurrence Matrix properties (Contrast, Energy, etc.) on ROI.
    -   **LBP:** Local Binary Patterns histogram on ROI.
-   **Shape:** Compactness and Area derived from the segmentation mask.

### 3. Classification & Optimization
-   **Model:** **Random Forest Classifier** (`n_estimators=500`, `class_weight='balanced_subsample'`).
-   **Threshold Tuning:** Decision threshold optimized (shifted from 0.5 to ~0.3) to maximize Recall for Malignant cases, prioritizing medical safety (minimizing False Negatives).

## Results
-   **Accuracy:** ~80%
-   **Sensitivity (Recall Malignant):** **86%** (High detection rate)
-   **Specificity (Benign):** ~73%

## Files
-   `classify.py`: Main pipeline script.
-   `generate_report.py`: Script to generate the PDF report.
-   `LAPORAN_AKHIR_G1A022073.pdf`: Final project report.
-   `assets/`: Contains logo used in report.
-   `preprocessing_logs/`: Visualizations of segmentation steps.

## How to Run
1.  Install dependencies: `pip install -r requirements.txt`
2.  Run analysis: `python classify.py`
3.  Generate PDF: `python generate_report.py`
