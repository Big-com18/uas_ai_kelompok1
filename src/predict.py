"""
predict.py
----------
Fungsi untuk memuat model CNN dan melakukan prediksi terhadap satu gambar.
Dipakai oleh app.py (Streamlit) maupun bisa dipanggil langsung untuk testing.
"""

import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "models/cnn_apple_disease.h5"
LABELS_PATH = "models/class_labels.json"
IMG_SIZE = (128, 128)


def load_artifacts():
    """Load model CNN dan mapping label. Panggil sekali saja di awal aplikasi."""
    model = load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        idx_to_label = json.load(f)
    return model, idx_to_label


def predict_image(model, idx_to_label, img_path_or_file):
    """
    Prediksi kelas penyakit dari sebuah gambar.

    Args:
        model: model Keras yang sudah di-load
        idx_to_label: dict mapping index -> nama kelas
        img_path_or_file: path file gambar ATAU file-like object (dari Streamlit uploader)

    Returns:
        (predicted_label, confidence_score)
    """
    img = image.load_img(img_path_or_file, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_idx = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    predicted_label = idx_to_label[str(predicted_idx)]

    return predicted_label, confidence
