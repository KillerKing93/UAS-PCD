# Project Context: Skin Cancer Classification

**Status:** Completed (Phase 2 Enhanced)
**Last Action:** Generated Final Report with ~93% CV Accuracy.

## Key Components
1.  **Preprocessing:** Resize (128px) -> Gray -> Gaussian Blur -> CLAHE.
2.  **Augmentation:** Horizontal Flipping (Train set only).
3.  **Features:** 
    -   Texture: GLCM (Contrast, etc.) + LBP.
    -   Color: HSV Statistics + Histograms.
    -   Shape: Hu Moments.
4.  **Model:** Random Forest (Optimized with GridSearch).

## Artifacts
-   `LAPORAN_AKHIR_G1A022073.pdf`: The final submission document.
-   `training_log.txt`: Detailed training logs.
-   `preprocessing_logs/`: Visuals of the pipeline.