# System Architecture

## Pipeline Overview

```mermaid
graph TD
    A[Input Image] --> B[Digital Hair Removal]
    B --> C[Otsu Segmentation (S-Channel)]
    C --> D{Mask Generation}
    D --> E[ROI Color Extraction (HSV+Lab)]
    D --> F[ROI Texture Extraction (GLCM+LBP)]
    D --> G[ROI Shape Extraction]
    E & F & G --> H[Feature Vector Concatenation]
    H --> I[Random Forest Classifier]
    I --> J[Threshold Tuning (< 0.5)]
    J --> K[Final Prediction]
```

## Detailed Components

### 1. Preprocessing & Segmentation
-   **Hair Removal:** Uses `cv2.morphologyEx` (BlackHat) to isolate hair and `cv2.inpaint` to remove it.
-   **Segmentation:** Converts image to HSV, extracts Saturation channel, applies Gaussian Blur, and performs Otsu's Binarization. Morphological Opening cleans noise. The largest contour is assumed to be the lesion.

### 2. Feature Extraction (ROI-Focused)
Features are computed **only** where `Mask > 0`.
-   **Color (Statistical):** 
    -   Mean, Std Dev, Skewness, Kurtosis.
    -   Spaces: **HSV** (Hue/Saturation/Value) and **CIELAB** (Perceptual lightness/color).
-   **Texture:**
    -   **LBP (Local Binary Patterns):** `radius=3`, `points=24`. Histogram of patterns inside the lesion.
    -   **GLCM:** Computed on the bounding box of the mask, masked to ignore background. Properties: Contrast, Energy, Homogeneity, Correlation, ASM.
-   **Shape:**
    -   **Compactness:** $4 \pi \times Area / Perimeter^2$ (Measure of circularity).
    -   **Area:** Raw pixel count of the lesion.

### 3. Classification
-   **Algorithm:** Random Forest.
-   **Strategy:** Ensemble learning with 500 trees.
-   **Class Balance:** `class_weight='balanced_subsample'` helps the model pay more attention to the minority class during bootstrap sampling.

### 4. Post-Processing
-   **Threshold Moving:** The default decision threshold (0.5) is lowered (e.g., to 0.3). This trades off some Specificity to significantly boost **Sensitivity (Recall)**, which is critical for medical diagnosis (better to flag a benign mole as suspicious than miss a cancer).
