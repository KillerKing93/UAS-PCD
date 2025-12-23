# Laporan Akhir: Klasifikasi Citra Medis

**Nama:** Alif Nurhidayat  
**NPM:** G1A022073  
**Repo:** [https://github.com/KillerKing93/UAS-PCD](https://github.com/KillerKing93/UAS-PCD)

## 1. Pendahuluan
Proyek ini bertujuan mengklasifikasikan lesi kulit (Benign vs Malignant) menggunakan pendekatan Machine Learning klasik yang ditingkatkan dengan teknik **Computer Vision (Segmentasi)**.

## 2. Metodologi
Berbeda dengan pendekatan standar, sistem ini memisahkan lesi dari kulit sehat sebelum analisis.

### 2.1 Preprocessing & Segmentasi
1.  **Hair Removal:** Menghapus rambut yang menutupi lesi.
2.  **Otsu Segmentation:** Membuat masker biner untuk mengisolasi area lesi (ROI).

### 2.2 Ekstraksi Fitur (ROI-Based)
Fitur diambil hanya dari area masker:
-   **Warna:** Statistik HSV dan CIELAB (menangkap variasi warna tumor).
-   **Tekstur:** GLCM dan LBP (menangkap ketidakteraturan permukaan).
-   **Bentuk:** Compactness (mengukur ketidakteraturan batas/border).

### 2.3 Klasifikasi & Optimasi
-   **Random Forest:** Digunakan karena robust terhadap noise.
-   **Threshold Tuning:** Ambang batas keputusan diturunkan untuk memprioritaskan deteksi kasus ganas (Recall Optimization).

## 3. Hasil
-   **Akurasi:** ~80%
-   **Sensitivitas (Malignant):** **86%** (Sangat Baik)
-   **Spesifisitas (Benign):** ~73%

## 4. Kesimpulan
Integrasi segmentasi otomatis dan penyesuaian threshold berhasil mengatasi masalah bias kelas, menghasilkan sistem yang sangat sensitif terhadap kanker kulit, memenuhi standar screening medis awal.