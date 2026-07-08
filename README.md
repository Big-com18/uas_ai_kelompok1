# 🍏 Deteksi Penyakit Daun Apel & Rekomendasi Penanganan

## Project Badges

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-orange)

Sistem AI end-to-end yang mengintegrasikan Computer Vision (CNN) untuk mendeteksi penyakit pada daun apel dari foto, dan Expert System (rule-based) untuk memberikan rekomendasi penanganan secara otomatis.

## Latar Belakang

Petani sering terlambat mendiagnosis penyakit daun apel karena keterbatasan akses ke pakar pertanian, sehingga penanganan menjadi lambat dan berisiko menurunkan hasil panen. Sistem ini bertujuan mempercepat proses diagnosis awal secara otomatis melalui foto daun.

## Architecture Diagram

```text
User / Petani
   |
   v
Streamlit UI (app.py)
   |
   v
Image Upload
   |
   v
Preprocessing (resize + normalisasi)
   |
   v
CNN Model (src/predict.py)
   |
   v
Expert System (src/expert_system.py)
   |
   v
Output: label, confidence, rekomendasi, status JSON
```

## Dataset

[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) subset Apple dengan 4 kelas:

- Apple___healthy
- Apple___Apple_scab
- Apple___Black_rot
- Apple___Cedar_apple_rust

## Struktur Direktori

```text
uas_ai_kelompok1/
├── app.py
├── requirements.txt
├── data/
│   └── raw/
├── models/
│   ├── class_labels.json
│   └── cnn_apple_disease.h5
├── notebooks/
├── src/
│   ├── expert_system.py
│   ├── predict.py
│   └── train.py
```

## Setup Guide

### 1. Install dependency

```bash
pip install -r requirements.txt
```

### 2. Siapkan dataset

Letakkan dataset di folder data/raw dengan struktur subfolder kelas sesuai nama kelas penyakit.

### 3. Latih model

```bash
cd src
python train.py
```

Model hasil training akan disimpan ke models/cnn_apple_disease.h5 dan label map ke models/class_labels.json.

### 4. Jalankan aplikasi

```bash
cd ..
streamlit run app.py
```

## API Docs

Aplikasi menerima input melalui unggahan gambar di antarmuka Streamlit. Setelah gambar diproses, sistem mengembalikan hasil dalam format JSON.

### Request Example

```json
{
  "image_file": "uploaded_image.jpg",
  "file_type": "jpg/png"
}
```

### Response Example (Status 200)

```json
{
  "status_code": 200,
  "predicted_label": "Apple___Black_rot",
  "confidence": 0.9421,
  "recommendation": "Pangkas dan musnahkan cabang atau buah yang terinfeksi segera.",
  "urgency_level": "Tinggi"
}
```

### Response Example (Status 400)

```json
{
  "status_code": 400,
  "error": "File yang diunggah bukan gambar valid."
}
```

## Evaluasi Model

Evaluasi dapat dicatat setelah training selesai, misalnya accuracy, precision, recall, F1-score, dan contoh kasus yang salah diklasifikasikan.

## Referensi

1. PlantVillage Dataset - Kaggle
2. TensorFlow/Keras Documentation - Convolutional Neural Networks
