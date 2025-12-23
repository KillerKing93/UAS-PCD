# System Architecture

## Pipeline Overview

```mermaid
graph LR
    A[Input Image] --> B[Preprocessing]
    B --> C[Feature Extraction]
    C --> D[Classification (Random Forest)]
    D --> E[Evaluation]
```

## Detailed Components

### 1. Preprocessing
-   **Resize:** 128x128 pixels.
-   **Gaussian Blur:** (5,5) kernel for noise reduction.
-   **CLAHE:** Contrast Limited Adaptive Histogram Equalization (Clip Limit=2.0) applied to the grayscale image to enhance local texture details.

### 2. Feature Extraction
A robust feature vector of **74 dimensions** is constructed from:
-   **GLCM (Texture):** 6 properties (Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM) calculated at 3 distances (1, 2, 3) and averaged over 4 angles.
-   **LBP (Texture):** Uniform Local Binary Patterns (Radius=3, Points=24). Histogram of LBP codes is used as features.
-   **HSV (Color):** Statistical moments (Mean, Standard Deviation, Skewness, Kurtosis) extracted from Hue, Saturation, and Value channels of the resized original image.

### 3. Classification
-   **Algorithm:** **Random Forest Classifier**.
-   **Reasoning:** Handles high-dimensional data well, robust to noise, and provides feature importance.
-   **Optimization:** GridSearchCV used to find optimal `n_estimators`, `max_depth`, and `min_samples_split`.

### 4. Evaluation Metrics
-   **Accuracy:** Overall correctness.
-   **Sensitivity (Recall):** Ability to detect Malignant cases.
-   **Specificity:** Ability to detect Benign cases.