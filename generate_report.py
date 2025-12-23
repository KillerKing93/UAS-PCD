from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Laporan Proyek Klasifikasi Citra Medis', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 5, 'Benign vs Malignant Classification using Classic Machine Learning', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def chapter_title(self, num, label):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, '%d. %s' % (num, label), 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, body)
        self.ln()
        
    def add_image(self, img_path, w=150):
        if os.path.exists(img_path):
            self.image(img_path, w=w, x=(210-w)/2)
            self.ln()
        else:
            self.cell(0, 10, f'[Image not found: {img_path}]', 0, 1)

pdf = PDF()
pdf.alias_nb_pages()
pdf.add_page()

# Title Page
pdf.set_font('Arial', 'B', 16)
pdf.cell(0, 40, '', 0, 1)
pdf.cell(0, 10, 'LAPORAN AKHIR', 0, 1, 'C')
pdf.cell(0, 10, 'PENGOLAHAN CITRA DIGITAL', 0, 1, 'C')
pdf.ln(20)
pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, 'Disusun Oleh:', 0, 1, 'C')
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Nama: Alif Nurhidayat', 0, 1, 'C')
pdf.cell(0, 10, 'NPM: G1A022073', 0, 1, 'C')
pdf.cell(0, 10, 'Program Studi: Informatika', 0, 1, 'C')
pdf.cell(0, 10, 'Fakultas: Teknik', 0, 1, 'C')
pdf.cell(0, 10, 'Universitas Bengkulu', 0, 1, 'C')
pdf.ln(30)
pdf.set_font('Arial', '', 10)
pdf.cell(0, 10, 'Tanggal: 23 Desember 2025', 0, 1, 'C')
pdf.add_page()

# Chapter 1: Pendahuluan
pdf.chapter_title(1, 'Pendahuluan')
pdf.chapter_body(
    "Proyek ini bertujuan untuk mengembangkan sistem klasifikasi citra medis otomatis untuk membedakan "
    "antara kasus jinak (Benign) dan ganas (Malignant). Sistem ini menggunakan pendekatan Machine Learning klasik "
    "yang mengandalkan ekstraksi fitur tekstur dan warna, diikuti oleh klasifikasi menggunakan algoritma Random Forest. "
    "Pendekatan ini dipilih karena efisiensinya pada dataset berukuran menengah dan kemampuannya untuk memberikan "
    "interpretasi fitur (feature importance)."
)

# Chapter 2: Metodologi
pdf.chapter_title(2, 'Metodologi')

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 10, '2.1 Dataset', 0, 1)
pdf.set_font('Arial', '', 11)
pdf.multi_cell(0, 5, 
    "Dataset yang digunakan adalah 'DATASET 1' yang terdiri dari:\n"
    "- Training Set: 980 citra (480 Benign, 500 Malignant).\n"
    "- Testing Set: 550 citra (250 Benign, 300 Malignant).\n"
    "Pembagian dataset sudah ditentukan sebelumnya dalam struktur folder."
)
pdf.ln()

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 10, '2.2 Preprocessing', 0, 1)
pdf.set_font('Arial', '', 11)
pdf.multi_cell(0, 5, 
    "Tahap pra-pemrosesan dilakukan untuk meningkatkan kualitas citra sebelum ekstraksi fitur:\n"
    "1. Resize: Citra diubah ukurannya menjadi 128x128 piksel.\n"
    "2. Grayscale: Konversi ke citra keabuan untuk analisis tekstur.\n"
    "3. Gaussian Blur: Kernel (5x5) untuk mengurangi noise frekuensi tinggi.\n"
    "4. CLAHE (Contrast Limited Adaptive Histogram Equalization): Meningkatkan kontras lokal untuk memperjelas detail tekstur."
)
pdf.ln()

# Add Preprocessing Step Image
pdf.set_font('Arial', 'I', 10)
pdf.cell(0, 10, 'Gambar 1: Visualisasi Tahapan Preprocessing', 0, 1, 'C')
# Try to find a generated image
import glob
step_images = glob.glob("preprocessing_logs/BENIGN/*_steps.png")
if step_images:
    pdf.add_image(step_images[0], w=170)
pdf.ln()

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 10, '2.3 Ekstraksi Fitur', 0, 1)
pdf.set_font('Arial', '', 11)
pdf.multi_cell(0, 5, 
    "Tiga jenis fitur diekstraksi untuk merepresentasikan karakteristik citra:\n"
    "1. GLCM (Gray Level Co-occurrence Matrix): Mengukur tekstur spasial. Properti: Contrast, Correlation, Energy, Homogeneity. (Haralick et al., 1973).\n"
    "2. LBP (Local Binary Pattern): Menangkap pola tekstur mikro invarian terhadap rotasi (Uniform LBP, R=3, P=24). (Ojala et al., 2002).\n"
    "3. Fitur Warna (HSV): Statistik (Mean, Std, Skewness, Kurtosis) dari kanal Hue, Saturation, dan Value. (Stricker & Orengo, 1995)."
)
pdf.ln()

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 10, '2.4 Klasifikasi', 0, 1)
pdf.set_font('Arial', '', 11)
pdf.multi_cell(0, 5, 
    "Algoritma Random Forest Classifier digunakan karena ketangguhannya terhadap overfitting dan kemampuannya menangani fitur campuran (tekstur dan warna). "
    "Optimasi hyperparameter dilakukan menggunakan GridSearchCV dengan 3-fold cross-validation."
)
pdf.ln()

# Chapter 3: Hasil dan Pembahasan
pdf.chapter_title(3, 'Hasil dan Pembahasan')

# Read metrics from file
acc = "N/A"
sens = "N/A"
spec = "N/A"
best_params = "N/A"

if os.path.exists("classification_results.txt"):
    with open("classification_results.txt", "r") as f:
        results_text = f.read()

    for line in results_text.split('\n'):
        if "Accuracy:" in line: acc = line.split(':')[1].strip()
        if "Sensitivity:" in line: sens = line.split(':')[1].strip()
        if "Specificity:" in line: spec = line.split(':')[1].strip()
        if "Best Params:" in line: best_params = line.split(':')[1].strip()

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 10, '3.1 Evaluasi Kuantitatif', 0, 1)
pdf.set_font('Arial', '', 11)

results_summary = (
    f"Model dievaluasi menggunakan data testing terpisah. Hasil metrik performa adalah:\n"
    f"- Akurasi: {float(acc)*100:.2f}%\n"
    f"- Sensitivitas (Recall - Malignant): {float(sens)*100:.2f}%\n"
    f"- Spesifisitas (Benign): {float(spec)*100:.2f}%\n\n"
    f"Parameter terbaik hasil Grid Search: {best_params}"
)
pdf.multi_cell(0, 5, results_summary)
pdf.ln()

pdf.set_font('Arial', 'I', 10)
pdf.cell(0, 10, 'Gambar 2: Confusion Matrix', 0, 1, 'C')
pdf.add_image('confusion_matrix.png', w=120)
pdf.ln()

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 10, '3.2 Analisis', 0, 1)
pdf.set_font('Arial', '', 11)
pdf.multi_cell(0, 5, 
    "Hasil menunjukkan bahwa model memiliki Spesifisitas yang sangat tinggi (>90%), yang berarti model sangat handal dalam mengenali kasus Jinak (Benign). "
    "Namun, Sensitivitas masih berada di angka ~66%, mengindikasikan bahwa model cenderung 'under-call' pada kasus Ganas (Malignant) dan mengklasifikasikannya sebagai jinak (False Negative).\n\n"
    "Analisis Feature Importance menunjukkan bahwa fitur Warna (khususnya fitur ke-62 dalam vektor fitur) memiliki pengaruh paling dominan. "
    "Hal ini wajar karena citra medis seringkali memiliki perbedaan karakteristik warna antara jaringan sehat dan sakit. "
    "Perbedaan performa antara Training (CV Score ~89%) dan Testing (~77%) mengindikasikan adanya overfitting atau perbedaan distribusi data test."
)
pdf.ln()

# Chapter 4: Kesimpulan
pdf.chapter_title(4, 'Kesimpulan')
pdf.chapter_body(
    "Sistem klasifikasi citra medis Benign vs Malignant telah berhasil dikembangkan menggunakan kombinasi fitur GLCM, LBP, dan Warna HSV dengan klasifikasi Random Forest. "
    "Meskipun akurasi keseluruhan mencapai ~77%, pengembangan lebih lanjut diperlukan untuk meningkatkan Sensitivitas model, misalnya dengan teknik oversampling (SMOTE) "
    "untuk menyeimbangkan kelas atau augmentasi data yang lebih agresif."
)

# Chapter 5: Referensi
pdf.chapter_title(5, 'Referensi')
pdf.set_font('Arial', '', 10)
refs = [
    "1. Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural Features for Image Classification. IEEE Transactions on Systems, Man, and Cybernetics, SMC-3(6), 610-621.",
    "2. Ojala, T., Pietikainen, M., & Maenpaa, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(7), 971-987.",
    "3. Stricker, M., & Orengo, M. (1995). Similarity of color images. Storage and Retrieval for Image and Video Databases III.",
    "4. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.",
    "5. Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297."
]
for ref in refs:
    pdf.multi_cell(0, 5, ref)
    pdf.ln(2)

pdf.output('LAPORAN_AKHIR_G1A022073.pdf', 'F')
print("PDF generated successfully: LAPORAN_AKHIR_G1A022073.pdf")