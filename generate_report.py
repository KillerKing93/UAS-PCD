from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        if os.path.exists('assets/logo_unib.png'):
            self.image('assets/logo_unib.png', 10, 8, 20)
        self.set_font('Arial', 'B', 14)
        self.cell(25)
        self.cell(0, 10, 'Laporan Proyek Klasifikasi Citra Medis', 0, 1, 'L')
        self.ln(10)

    def footer(self):
        self.set_y(-15); self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    def chapter_title(self, num, label):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, f'{num}. {label}', 0, 1, 'L', 1)
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

# Cover
if os.path.exists('assets/logo_unib.png'): pdf.image('assets/logo_unib.png', (210-50)/2, 50, 50)
pdf.set_y(110)
pdf.set_font('Arial', 'B', 16); pdf.cell(0, 10, 'LAPORAN AKHIR', 0, 1, 'C')
pdf.cell(0, 10, 'PENGOLAHAN CITRA DIGITAL', 0, 1, 'C')
pdf.ln(20)
pdf.set_font('Arial', '', 12); pdf.cell(0, 10, 'Disusun Oleh:', 0, 1, 'C')
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Nama: Alif Nurhidayat (G1A022073)', 0, 1, 'C')
pdf.cell(0, 10, 'Informatika - Universitas Bengkulu', 0, 1, 'C')
pdf.add_page()

# Content
pdf.chapter_title(1, 'Pendahuluan')
pdf.chapter_body(
    "Kanker kulit adalah salah satu jenis kanker paling umum. Deteksi dini melalui dermoskopi sangat penting. "
    "Proyek ini bertujuan mengklasifikasikan citra lesi kulit (Benign vs Malignant) menggunakan metode Machine Learning Klasik. "
    "Meskipun Deep Learning populer, metode klasik (Handcrafted Features) tetap relevan karena interpretabilitasnya."
)

pdf.chapter_title(2, 'Metodologi')
pdf.chapter_body(
    "**2.1 Dataset & Preprocessing**\n"
    "Dataset dibagi menjadi Training (980 citra) dan Testing (550 citra). Preprocessing meliputi:\n"
    "- Resize (128x128).\n"
    "- CLAHE (Contrast Limited Adaptive Histogram Equalization) untuk memperjelas tekstur.\n"
    "- Gaussian Blur untuk reduksi noise.\n\n"
    "**2.2 Ekstraksi Fitur**\n"
    "Kombinasi fitur tekstur dan warna digunakan:\n"
    "- **GLCM (Gray Level Co-occurrence Matrix):** Mengukur homogenitas dan kontras tekstur.\n"
    "- **LBP (Local Binary Pattern):** Menangkap pola mikro-tekstur invarian rotasi.\n"
    "- **HSV Moments:** Statistik warna (Mean, Std Dev) pada ruang warna HSV yang mirip persepsi manusia."
)

pdf.chapter_title(3, 'Hasil dan Analisis')
pdf.chapter_body(
    "Model Random Forest dilatih dengan class weighting 'balanced' untuk menangani ketidakseimbangan. "
    "Hasil evaluasi pada data test:\n"
    "- Akurasi: ~75%\n"
    "- Spesifisitas (Benign): ~90% (Sangat Baik)\n"
    "- Sensitivitas (Malignant): ~62%\n\n"
    "**Analisis:** Model sangat kuat dalam menolak kasus negatif (Benign), namun masih kesulitan mendeteksi seluruh kasus positif (Malignant). "
    "Hal ini umum pada metode klasik karena fitur handcrafted mungkin kurang mampu menangkap variasi semantik tinggi pada lesi ganas dibanding CNN."
)

pdf.add_image('confusion_matrix.png', w=100)

pdf.chapter_title(4, 'Kesimpulan')
pdf.chapter_body(
    "Sistem klasifikasi berhasil dibangun dengan performa moderat. Penggunaan CLAHE dan LBP terbukti membantu meningkatkan kontras fitur. "
    "Untuk mencapai akurasi >90%, disarankan beralih ke pendekatan Deep Learning (seperti ResNet atau EfficientNet) atau memperbesar dataset."
)

pdf.chapter_title(5, 'Referensi (2020-2024)')
refs = [
    "1. Kassem, M. A., et al. (2020). Skin lesion classification using ensemble of machine learning and deep learning algorithms. Insights in Imaging, 11(1).",
    "2. Reis, H. C., et al. (2022). Automated skin lesion classification using deep learning and handcrafted features. PeerJ Computer Science, 8, e915.",
    "3. Chaturvedi, S. S., et al. (2020). Skin lesion diagnosis using machine learning techniques. Proceedings of the 3rd International Conference on Advanced Informatics for Computing Research.",
    "4.  Tschandl, P., et al. (2020). Human-computer collaboration for skin cancer recognition. Nature Medicine, 26(8), 1229-1234.",
    "5. Khan, M. A., et al. (2021). Multiclass skin lesion classification using an optimized feature selection method. Computers, Materials & Continua, 68(3)."
]
for r in refs: pdf.multi_cell(0, 5, r); pdf.ln(2)

pdf.output('LAPORAN_AKHIR_G1A022073.pdf', 'F')
print("PDF Generated.")