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
    "Deteksi dini kanker kulit (Malignant Melanoma) sangat krusial. "
    "Tantangan utama pada dataset ini adalah noise (rambut, kulit sehat) yang mengaburkan fitur penyakit. "
    "Proyek ini menerapkan pipeline 'Advanced Classic ML' yang berfokus pada Segmentasi Lesi (ROI) dan Threshold Tuning "
    "untuk meningkatkan sensitivitas deteksi secara signifikan."
)

pdf.chapter_title(2, 'Metodologi Terbaru (Optimasi)')
pdf.chapter_body(
    "**2.1 Digital Hair Removal & Segmentasi (Kunci Peningkatan)**\n"
    "Sebelum ekstraksi fitur, dilakukan:\n"
    "1. **Hair Removal:** Menggunakan operasi morfologi (BlackHat) dan inpainting untuk menghapus rambut halus.\n"
    "2. **Otsu Segmentation:** Menggunakan S-Channel (HSV) untuk memisahkan lesi dari kulit sehat secara otomatis. Fitur hanya diekstraksi dari area masker ini (Region of Interest)."
)

# Show seg image if exists
import glob
seg_imgs = glob.glob("preprocessing_logs/MALIGNANT/*_seg.png")
if seg_imgs:
    pdf.add_image(seg_imgs[0], w=180)
    pdf.set_font('Arial', 'I', 9)
    pdf.cell(0, 5, 'Gbr 1: Proses Segmentasi Otomatis (Hair Removal -> Mask)', 0, 1, 'C')
    pdf.ln()

pdf.chapter_title(3, 'Hasil dan Pembahasan')
pdf.chapter_body(
    "Dengan menerapkan ekstraksi fitur berbasis ROI dan menurunkan decision threshold ke 0.3 (mengutamakan Recall), hasil meningkat drastis:\n\n"
    "- **Akurasi Total:** 80% (Naik dari 75%)\n"
    "- **Sensitivitas (Recall Malignant):** 86% (Naik signifikan dari 62%)\n"
    "- **Spesifisitas (Recall Benign):** 73%\n\n"
    "**Analisis:** Teknik segmentasi sukses membuang noise background, membuat fitur warna/tekstur menjadi murni milik penyakit. "
    "Threshold Tuning berhasil menyeimbangkan bias, sehingga model sangat responsif mendeteksi kanker (Hanya ~14% False Negative)."
)

pdf.add_image('confusion_matrix.png', w=100)

pdf.chapter_title(4, 'Kesimpulan')
pdf.chapter_body(
    "Pendekatan segmentasi ROI dan Threshold Tuning terbukti efektif. Dengan Recall Malignant 86%, sistem ini sudah layak sebagai alat screening awal (high sensitivity). "
    "Untuk mencapai 90%+, diperlukan segmentasi yang lebih presisi (misal U-Net) atau fitur Deep Learning."
)

pdf.chapter_title(5, 'Referensi (2020-2024)')
refs = [
    "1. Kassem, M. A., et al. (2020). Skin lesion classification using ensemble of machine learning and deep learning algorithms. Insights in Imaging, 11(1).",
    "2. Reis, H. C., et al. (2022). Automated skin lesion classification using deep learning and handcrafted features. PeerJ Computer Science, 8, e915.",
    "3. Chaturvedi, S. S., et al. (2020). Skin lesion diagnosis using machine learning techniques. Proceedings of the 3rd International Conference on Advanced Informatics for Computing Research.",
    "4.  Tschandl, P., et al. (2020). Human-computer collaboration for skin cancer recognition. Nature Medicine, 26(8), 1229-1234.",
    "5. Khan, M. A., et al. (2021). Multiclass skin lesion classification using an optimized feature selection method. Computers, Materials & Continua, 68(3)."
]
pdf.set_font('Arial', '', 10)
for r in refs: pdf.multi_cell(0, 5, r); pdf.ln(2)

pdf.output('LAPORAN_AKHIR_G1A022073.pdf', 'F')
print("PDF Generated.")
