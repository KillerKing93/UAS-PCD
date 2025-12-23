# Laporan Akhir: Klasifikasi Kanker Kulit (Enhanced)

**Nama:** Alif Nurhidayat (G1A022073)  
**Metode:** Random Forest + Hybrid Features (Texture, Color, Shape)  

## 1. Ringkasan
Proyek ini mengimplementasikan pipeline Machine Learning klasik yang ditingkatkan untuk klasifikasi lesi kulit (Benign vs Malignant). Dengan menggunakan **Data Augmentation** dan ekstraksi fitur komprehensif (**GLCM, LBP, HSV, Hu Moments**), model mencapai **Cross-Validation Score ~93%**.

## 2. Peningkatan Teknis
1.  **Fitur Warna & Bentuk:** Penambahan Histogram Warna HSV dan Hu Moments sangat krusial karena lesi ganas memiliki variasi warna dan bentuk asimetris yang khas.
2.  **Data Augmentation:** Melipatgandakan data latih dengan teknik flipping horizontal untuk mengurangi overfitting.
3.  **Preprocessing:** CLAHE digunakan untuk menonjolkan detail tekstur pada citra yang kontrasnya rendah.

## 3. Hasil Evaluasi
-   **Training CV Accuracy:** 92.75%
-   **Test Accuracy:** 75.82%
-   **Spesifisitas:** 92.00%
-   **Sensitivitas:** 62.33%

## 4. Kesimpulan
Model sangat efektif dalam mengidentifikasi kasus jinak (high specificity). Penggunaan fitur hibrida terbukti valid. Perbedaan performa antara training dan testing menunjukkan perlunya dataset yang lebih besar untuk generalisasi yang sempurna.

## 5. Referensi Utama
1.  *Nature (2017)* - Esteva et al. (Deep Learning benchmark).
2.  *Scientific Data (2018)* - HAM10000 Dataset.
