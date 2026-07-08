# 🍏 Deteksi Penyakit Daun Apel & Rekomendasi Penanganan

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-orange)

Sistem AI end-to-end yang mengintegrasikan **Computer Vision (CNN)** untuk
mendeteksi jenis penyakit pada daun apel dari foto, dan **Expert System
(rule-based)** untuk memberikan rekomendasi penanganan secara otomatis.

## Latar Belakang

Petani sering terlambat mendiagnosis penyakit daun apel karena keterbatasan
akses ke pakar pertanian, sehingga penanganan menjadi lambat dan berisiko
menurunkan hasil panen. Sistem ini bertujuan mempercepat proses diagnosis
awal secara otomatis melalui foto daun.

## Arsitektur Sistem

```
Input Layer (Upload Gambar via Streamlit)
        |
Preprocessing (Resize 128x128, Normalisasi)
        |
Inference (CNN - klasifikasi 4 kelas)
        |
Expert System (rule-based rekomendasi)
        |
Output (Label penyakit + rekomendasi + JSON response)
```

## Dataset

[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
(subset Apple), 4 kelas:
- `Apple___healthy`
- `Apple___Apple_scab`
- `Apple___Black_rot`
- `Apple___Cedar_apple_rust`

## Struktur Direktori

```
project-deteksi-penyakit-apel/
├── data/raw/            # dataset (tidak di-commit, lihat .gitignore)
├── models/              # model .h5 hasil training + mapping label
├── src/
│   ├── train.py         # training CNN
│   ├── predict.py       # fungsi load model & prediksi
│   └── expert_system.py # rule-based rekomendasi
├── notebooks/           # eksperimen EDA
├── app.py               # aplikasi Streamlit
└── requirements.txt
```

## Cara Menjalankan

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset dari Kaggle, taruh di data/raw/

# 3. Latih model
python src/train.py

# 4. Jalankan aplikasi
streamlit run app.py
```

## API Response Format

Prediksi dikembalikan dalam format JSON:

```json
{
  "status_code": 200,
  "predicted_label": "Apple___Black_rot",
  "confidence": 0.9421,
  "recommendation": "Pangkas dan musnahkan cabang atau buah yang terinfeksi...",
  "urgency_level": "Tinggi"
}
```

Jika input tidak valid (bukan gambar), sistem mengembalikan pesan error
yang mudah dipahami tanpa membuat aplikasi crash (lihat `app.py`, blok
try-except).

## Evaluasi Model

Diisi setelah training selesai — catat Accuracy, Precision, Recall,
F1-Score, serta contoh kasus yang salah diklasifikasikan (edge case).

## Live Demo

- Streamlit: `<https://uas-ai03-kelompok1.streamlit.app/>`
- Artikel Medium: `<https://medium.com/@ferrysinaga61/asisten-ai-di-saku-petani-mendiagnosis-penyakit-apel-dengan-computer-vision-dan-sistem-pakar-d5846f65838f?sharedUserId=ferrysinaga61>`

## Referensi

1. PlantVillage Dataset - Kaggle
2. TensorFlow/Keras Documentation - Convolutional Neural Networks
