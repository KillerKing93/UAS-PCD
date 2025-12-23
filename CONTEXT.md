# Project Context & State

**Last Updated:** Selasa, 23 Desember 2025
**Project:** Medical Image Classification (Benign vs Malignant)
**Repository:** https://github.com/KillerKing93/UAS-PCD

## 1. Project Overview
This project aims to classify medical images into **BENIGN** (Benign) and **MALIGNANT** (Malignant) classes using Classic Machine Learning methods (no Deep Learning).

## 2. Current Implementation (`classify.py`)
The pipeline has evolved to improve accuracy and logging.

### Preprocessing
1.  **Resize:** 128x128 pixels.
2.  **Grayscale:** For texture analysis.
3.  **Gaussian Blur:** (5,5) kernel for noise reduction.
4.  **CLAHE:** Contrast Limited Adaptive Histogram Equalization to enhance local contrast.
5.  **Visualization:** Steps are saved in `preprocessing_logs/<class>/<image>_steps.png`.

### Feature Extraction
The model currently uses a robust combination of **74 features**:
1.  **GLCM (Texture):** Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM. (Multi-scale: distances [1,2,3]).
2.  **LBP (Texture):** Local Binary Patterns (Uniform, Radius=3, Points=24) to capture micro-textures.
3.  **HSV (Color):** Mean, Std Dev, Skewness, and Kurtosis of Hue, Saturation, and Value channels.

### Classifier
-   **Algorithm:** **Random Forest Classifier** (Switched from SVM to handle mixed feature types better and reduce overfitting).
-   **Validation:** GridSearchCV (CV=3).
-   **Current Metrics (Est.):** High CV accuracy (~92%) but lower Test accuracy (~78%) indicating potential overfitting or data mismatch.

## 3. Directory Structure
-   `DATASETS/DATASET 1`: Source data (Train/Test split).
-   `preprocessing_logs/`: Generated visualizations and feature logs.
-   `training_log.txt`: Appended log of training progress.
-   `classification_results.txt`: Latest metrics and confusion matrix.
-   `Learning_Materials/`: Reference documents.

## 4. Immediate Tasks
1.  **Fix Overfitting:** The gap between Train/CV accuracy and Test accuracy is significant.
2.  **Documentation:** Update `REPORT.md`, `README.md`, `ARCHITECTURE.md` with the new Random Forest + LBP/HSV approach.
3.  **Execution:** Run `classify.py` to generate fresh results.

## 5. Environment
-   Python 3.x
-   `requirements.txt`: numpy, opencv-python, scikit-learn, scikit-image, matplotlib, pandas, scipy, seaborn.

## 6. Git Rules
-   Branch: `main`
-   Commit Messages: Strict "No AI mention" policy.
-   Ignored Files: `GEMINI.md`, `CONTEXT.md`, `__pycache__`, `.gemini/`.
