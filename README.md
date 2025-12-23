# Medical Image Classification (Benign vs Malignant)

**Author:** Alif Nurhidayat (G1A022073)  
**Program:** Informatika, Universitas Bengkulu

## Project Overview
This project implements a Machine Learning pipeline to classify medical images into two categories: **BENIGN** (Benign) and **MALIGNANT** (Malignant). It uses advanced feature extraction techniques (GLCM, LBP, HSV Color) and a Random Forest Classifier.

## Method
1.  **Preprocessing:**
    -   Resize (128x128)
    -   Grayscale Conversion
    -   Gaussian Blur (Noise Reduction)
    -   CLAHE (Contrast Enhancement)
2.  **Feature Extraction:**
    -   **Texture:** GLCM (Contrast, Energy, etc.) & LBP (Local Binary Patterns)
    -   **Color:** HSV Statistics (Mean, Std, Skewness, Kurtosis)
3.  **Classification:**
    -   **Random Forest Classifier** with GridSearchCV.

## Results
-   **Accuracy:** 77.64%
-   **Sensitivity:** 66.67%
-   **Specificity:** 90.80%

## How to Run
1.  Install dependencies: `pip install -r requirements.txt` (ensure `fpdf` is installed for report generation).
2.  Run classification: `python classify.py`
3.  Generate PDF Report: `python generate_report.py`

## Structure
-   `classify.py`: Main script for training and evaluation.
-   `generate_report.py`: Generates the PDF report.
-   `LAPORAN_AKHIR_G1A022073.pdf`: Final Report.
-   `preprocessing_logs/`: Visualizations of image processing steps.