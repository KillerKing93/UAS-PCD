# Laporan Akhir: Klasifikasi Citra Medis

**Nama:** Alif Nurhidayat  
**NPM:** G1A022073  
**Prodi:** Informatika, Universitas Bengkulu

## 1. Pendahuluan
Proyek ini bertujuan untuk mengklasifikasikan citra medis menjadi dua kelas: **BENIGN** (Jinak) dan **MALIGNANT** (Ganas). Metode yang digunakan menggabungkan ekstraksi fitur tekstur (GLCM, LBP) dan warna (HSV) dengan klasifikasi Random Forest.

## 2. Metodologi

### 2.1 Preprocessing
1.  **Resize:** 128x128 piksel.
2.  **Gaussian Blur:** Mengurangi noise.
3.  **CLAHE:** Meningkatkan kontras lokal.

### 2.2 Ekstraksi Fitur
-   **GLCM:** Tekstur spasial (Haralick et al., 1973).
-   **LBP:** Tekstur mikro invarian rotasi (Ojala et al., 2002).
-   **HSV:** Statistik warna (Stricker & Orengo, 1995).

### 2.3 Klasifikasi
**Random Forest Classifier** digunakan dengan optimasi hyperparameter (GridSearchCV).

## 3. Hasil
-   **Akurasi:** 77.64%
-   **Sensitivitas:** 66.67%
-   **Spesifisitas:** 90.80%

Hasil menunjukkan model sangat baik dalam mengenali Benign (Spesifisitas tinggi) namun perlu peningkatan dalam mengenali Malignant.

## 4. Referensi
1.  Haralick, R. M., et al. (1973). *Textural Features for Image Classification*.
2.  Ojala, T., et al. (2002). *Multiresolution gray-scale and rotation invariant texture classification with local binary patterns*.
3.  Breiman, L. (2001). *Random Forests*.