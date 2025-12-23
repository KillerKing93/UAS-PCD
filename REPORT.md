# Laporan Klasifikasi Citra Medis: Benign vs Malignant

## 1. Pendahuluan
Proyek ini bertujuan untuk mengklasifikasikan citra medis menjadi dua kelas: **BENIGN** (Jinak) dan **MALIGNANT** (Ganas). Metode yang digunakan adalah pendekatan Machine Learning klasik yang menggabungkan ekstraksi fitur tekstur GLCM (Gray Level Co-occurrence Matrix) dengan klasifikasi SVM (Support Vector Machine).

## 2. Metodologi

### 2.1 Dataset
Dataset yang digunakan adalah **DATASET 1** yang terdiri dari:
- **Training Set:** 980 citra (480 Benign, 500 Malignant).
- **Testing Set:** 550 citra (250 Benign, 300 Malignant).

Jumlah ini memenuhi syarat minimal (200 training/kelas, 50 testing/kelas).

### 2.2 Pre-processing
Setiap citra melalui tahap pra-pemrosesan sebagai berikut:
1.  **Resize:** Citra diubah ukurannya menjadi 128x128 piksel untuk keseragaman.
2.  **Grayscale Conversion:** Konversi dari RGB ke Grayscale karena fitur GLCM dihitung berdasarkan intensitas piksel tunggal.

### 2.3 Ekstraksi Fitur
Fitur tekstur diekstraksi menggunakan metode **GLCM (Gray Level Co-occurrence Matrix)**.
- **Parameter:** Jarak (distance) = 1 piksel; Sudut (angles) = 0°, 45°, 90°, 135°.
- **Properti:** Contrast, Dissimilarity, Homogeneity, Energy, Correlation, dan ASM.
- **Vektor Fitur:** Nilai rata-rata dari setiap properti pada keempat sudut digunakan sebagai fitur akhir (total 6 fitur per citra).

### 2.4 Klasifikasi
Klasifikasi dilakukan menggunakan **Support Vector Machine (SVM)** dengan kernel RBF (Radial Basis Function). Data fitur dinormalisasi menggunakan `StandardScaler` sebelum pelatihan untuk memastikan konvergensi yang optimal.

## 3. Hasil dan Analisis

### 3.1 Performa Kuantitatif
Hasil evaluasi pada data testing adalah sebagai berikut:

- **Akurasi:** 73.45%
- **Sensitivitas (Recall - Malignant):** 72.33%
- **Spesifisitas (Benign):** 74.80%

### 3.2 Analisis Hasil
Model mampu membedakan antara citra Benign dan Malignant dengan akurasi yang moderat (~73%).
- **Sensitivitas 72.33%** menunjukkan kemampuan model dalam mendeteksi kasus ganas (Malignant) cukup baik, namun masih terdapat sekitar 27% kasus ganas yang terklasifikasi sebagai jinak (False Negative).
- **Spesifisitas 74.80%** menunjukkan kemampuan model menolak kasus jinak (tidak mendeteksi sebagai ganas) sedikit lebih baik daripada sensitivitasnya.

Confusion Matrix menunjukkan keseimbangan yang relatif baik antara False Positives dan False Negatives, mengindikasikan bahwa model tidak terlalu bias terhadap salah satu kelas mayoritas. Peningkatan performa mungkin dapat dicapai dengan menambahkan fitur lain (seperti fitur bentuk atau warna) atau menggunakan teknik segmentasi untuk memisahkan ROI (Region of Interest) dari latar belakang sebelum ekstraksi fitur.

## 4. Referensi
1.  Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural Features for Image Classification. *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-3(6), 610-621.
2.  Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.
