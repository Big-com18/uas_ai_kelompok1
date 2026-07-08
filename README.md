# 🍏 Asisten AI di Saku Petani — Deteksi Penyakit Daun Apel & Rekomendasi Penanganan

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-Academic%20Project-lightgrey)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

Sistem AI end-to-end yang mengintegrasikan **Computer Vision (CNN)** untuk
mendeteksi jenis penyakit pada daun apel dari foto, dan **Expert System
(rule-based)** untuk memberikan rekomendasi penanganan secara otomatis —
seperti membawa seorang pakar pertanian di dalam saku petani.

Proyek ini disusun untuk memenuhi Ujian Akhir Semester (UAS) mata kuliah
Artificial Intelligence — Topik 11: *Deteksi Penyakit Tanaman &
Rekomendasi Penanganan (Computer Vision + Expert System)*.

## 🔗 Tautan Proyek

| Item | Link |
|---|---|
| 🚀 Live Demo (Streamlit) | https://uas-ai03-kelompok1.streamlit.app/ |
| 💻 Source Code (GitHub) | https://github.com/Big-com18/uas_ai_kelompok1 |
| 📝 Artikel Medium | https://medium.com/@ferrysinaga61/asisten-ai-di-saku-petani-mendiagnosis-penyakit-apel-dengan-computer-vision-dan-sistem-pakar-d5846f65838f |

## 👥 Anggota Kelompok

| Nama | NIM |
|---|---|
| Billy Andreas | 24110300050 |
| Ahmad Fuady | 24110300005 |
| Ferry Firmando | 24110300055 |

## 📌 Latar Belakang

Petani sering terlambat mendiagnosis penyakit daun apel karena keterbatasan
akses ke pakar pertanian, sehingga penanganan menjadi lambat dan berisiko
menurunkan hasil panen. Sistem ini bertujuan mempercepat proses diagnosis
awal secara otomatis melalui foto daun, dengan target **meningkatkan
efisiensi diagnosis penyakit daun apel hingga 40%** dibandingkan proses
identifikasi manual.

## 🏗️ Architecture Diagram

```
                    ┌───────────────────────────┐
                    │   INPUT LAYER              │
                    │   (Upload foto daun apel   │
                    │    via Streamlit / REST)   │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   PREPROCESSING            │
                    │   - Resize 128x128         │
                    │   - Normalisasi pixel /255 │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   INFERENCE (CNN)          │
                    │   Model: cnn_apple_        │
                    │   disease.h5               │
                    │   Output: 4 kelas +        │
                    │   confidence score         │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   EXPERT SYSTEM             │
                    │   (Rule-Based Engine)       │
                    │   Label penyakit →          │
                    │   Rekomendasi penanganan     │
                    │   + tingkat urgensi          │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   OUTPUT                    │
                    │   - Tampilan Streamlit       │
                    │   - JSON response (REST API) │
                    └───────────────────────────┘
```

**Alur singkat:** gambar daun apel diunggah → diproses (resize &
normalisasi) → diklasifikasikan oleh model CNN ke salah satu dari 4 kelas
→ hasil klasifikasi menjadi *input* bagi Expert System rule-based yang
menentukan rekomendasi penanganan dan tingkat urgensinya → hasil akhir
ditampilkan ke pengguna baik lewat antarmuka Streamlit maupun response
JSON dari REST API.

## 🧠 Dua Teknik AI yang Diintegrasikan

1. **Computer Vision (Convolutional Neural Network)** — mengklasifikasikan
   gambar daun apel ke dalam 4 kelas: *healthy*, *Apple Scab*,
   *Black Rot*, *Cedar Apple Rust*.
2. **Expert System (Rule-Based Reasoning)** — menerima output CNN sebagai
   fakta, lalu menjalankan aturan IF-THEN untuk menghasilkan rekomendasi
   penanganan dan tingkat urgensi yang relevan secara agronomis.

## 📂 Struktur Direktori & Penjelasan

```
uas_ai_kelompok1/
├── data/
│   └── raw/                       # Dataset PlantVillage (subset Apple)
│       ├── Apple___healthy/
│       ├── Apple___Apple_scab/
│       ├── Apple___Black_rot/
│       └── Apple___Cedar_apple_rust/
│
├── models/                        # Artefak hasil training (tidak ditulis manual)
│   ├── cnn_apple_disease.h5       # Model CNN terlatih (format HDF5)
│   ├── class_labels.json          # Mapping index -> nama kelas
│   └── evaluation_results.json    # Hasil evaluasi (accuracy, precision, dst)
│
├── src/                           # Seluruh source code inti
│   ├── train.py                   # Melatih model CNN dari dataset
│   ├── evaluate.py                # Menghitung Accuracy, Precision, Recall, F1
│   ├── predict.py                 # Fungsi load model & prediksi 1 gambar
│   ├── expert_system.py           # Rule-based engine untuk rekomendasi
│   └── api.py                     # REST API (FastAPI) untuk akses eksternal
│
├── notebooks/
│   └── eda.ipynb                  # Eksplorasi data awal (EDA): distribusi
│                                   # kelas, contoh gambar, dsb.
│
├── app.py                         # Aplikasi Streamlit (antarmuka pengguna)
├── requirements.txt                # Daftar dependency Python
├── .gitignore                      # Mengecualikan venv/ dan data mentah besar
└── README.md                       # Dokumentasi proyek (file ini)
```

**Penjelasan tiap bagian:**

- **`data/`** — berisi dataset mentah yang dipakai untuk melatih model.
  Folder ini **tidak ikut di-commit** ke GitHub (ukurannya besar), pengguna
  lain perlu mengunduhnya sendiri dari sumber dataset (lihat bagian
  Dataset di bawah).
- **`models/`** — tempat menyimpan hasil "belajar" sistem: bobot model CNN
  (`.h5`), pemetaan label kelas (`.json`), dan ringkasan hasil evaluasi.
  File-file ini dihasilkan otomatis oleh `train.py` dan `evaluate.py`,
  bukan ditulis manual.
- **`src/`** — jantung dari sistem. Dipisah per tanggung jawab (single
  responsibility): training terpisah dari evaluasi, terpisah dari logika
  expert system, terpisah dari REST API — sehingga masing-masing bagian
  mudah diuji dan dikembangkan secara independen.
- **`notebooks/`** — tempat eksperimen dan eksplorasi data yang sifatnya
  eksploratif, dipisahkan dari kode produksi di `src/` agar `src/` tetap
  bersih dan siap dijalankan sebagai aplikasi.
- **`app.py`** — satu-satunya entry point untuk menjalankan antarmuka
  Streamlit, memanggil fungsi-fungsi dari `src/predict.py` dan
  `src/expert_system.py`.

## 📊 Dataset

[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
(subset Apple), 4 kelas:
- `Apple___healthy`
- `Apple___Apple_scab`
- `Apple___Black_rot`
- `Apple___Cedar_apple_rust`

## ⚙️ Setup Guide (Cara Menjalankan di Lokal)

```bash
# 1. Clone repository
git clone https://github.com/Big-com18/uas_ai_kelompok1.git
cd uas_ai_kelompok1

# 2. Buat virtual environment (gunakan Python 3.10/3.11, TensorFlow
#    belum mendukung versi Python yang lebih baru)
py -3.11 -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS/Linux

# 3. Install seluruh dependency
pip install -r requirements.txt

# 4. Download dataset PlantVillage dari Kaggle, ekstrak 4 folder kelas
#    Apple ke dalam data/raw/

# 5. Latih model CNN
python src/train.py

# 6. Evaluasi model (Accuracy, Precision, Recall, F1-Score)
python src/evaluate.py

# 7. Jalankan aplikasi Streamlit
streamlit run app.py

# 8. (Opsional) Jalankan REST API secara terpisah
uvicorn src.api:app --reload --port 8000
# Dokumentasi interaktif otomatis tersedia di: http://127.0.0.1:8000/docs
```

## 📡 API Documentation

Endpoint REST API dibangun menggunakan **FastAPI**, menerima gambar
sebagai input dan mengembalikan hasil klasifikasi beserta rekomendasi
Expert System dalam format JSON.

### `POST /predict`

**Request** — `multipart/form-data`

| Field | Tipe | Keterangan |
|---|---|---|
| `file` | image (jpg/jpeg/png) | Foto daun apel yang ingin dideteksi |

Contoh pemanggilan dengan `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -F "file=@contoh_daun.jpg"
```

**Response — Status 200 (Berhasil)**
```json
{
  "status_code": 200,
  "predicted_label": "Apple___Black_rot",
  "confidence": 0.9421,
  "recommendation": "Pangkas dan musnahkan cabang atau buah yang terinfeksi segera. Hindari pemupukan nitrogen berlebih. Gunakan fungisida tembaga pada fase awal pertumbuhan daun.",
  "urgency_level": "Tinggi"
}
```

**Response — Status 400 (Input Tidak Valid)**

Dikembalikan jika file yang diunggah bukan gambar, rusak, atau kosong:
```json
{
  "status_code": 400,
  "error": "Invalid file format",
  "message": "File yang diunggah bukan gambar yang valid. Harap unggah file berformat JPG atau PNG."
}
```

**Response — Status 500 (Kesalahan Server)**

Dikembalikan jika terjadi kesalahan tak terduga saat inferensi model
(model gagal dimuat, dsb.), dibungkus dengan blok `try-except` agar
aplikasi tidak crash:
```json
{
  "status_code": 500,
  "error": "Internal server error",
  "message": "Terjadi kesalahan saat memproses gambar. Silakan coba lagi."
}
```

## 📈 Evaluasi Model

Hasil evaluasi pada data validasi (dihasilkan oleh `src/evaluate.py`):

| Metrik | Skor |
|---|---|
| Accuracy | 94.01% |
| Precision (macro avg) | 0.9314 |
| Recall (macro avg) | 0.9290 |
| F1-Score (macro avg) | 0.9284 |

Rincian per kelas serta confusion matrix lengkap tersedia di
`models/evaluation_results.json`, dan dibahas secara naratif — termasuk
kasus-kasus kegagalan model (edge case) — pada artikel Medium di atas.

## 📚 Referensi

1. PlantVillage Dataset — Kaggle
2. TensorFlow/Keras Documentation — Convolutional Neural Networks
3. FastAPI Documentation — Building REST APIs with Python
4. Streamlit Documentation — Rapid App Development for ML