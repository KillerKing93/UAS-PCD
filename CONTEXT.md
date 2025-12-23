# Project Context: Skin Cancer Classification

**Status:** Completed (Phase 3 - Segmentation & ROI)
**Last Action:** Updated all documentation to reflect the final segmentation-based pipeline.

## Key Components
1.  **Segmentation:** Hair Removal -> Otsu Masking.
2.  **Features:** ROI-based Color (HSV, Lab), Texture (LBP, GLCM), Shape.
3.  **Model:** Random Forest (500 estimators).
4.  **Optimization:** Decision Threshold Tuning (0.3).

## Artifacts
-   `classify.py`: The segmentation and classification logic.
-   `generate_report.py`: Report generator.
-   `LAPORAN_AKHIR_G1A022073.pdf`: The submitted report.
