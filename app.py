"""
app.py
------
Aplikasi Streamlit: Deteksi Penyakit Daun Apel & Rekomendasi Penanganan.
Menggabungkan 2 teknik AI:
    1. Computer Vision (CNN) -> deteksi jenis penyakit dari gambar daun
    2. Expert System (rule-based) -> rekomendasi penanganan berdasarkan hasil CNN
"""

import streamlit as st
from PIL import Image
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from predict import load_artifacts, predict_image  # noqa: E402
from expert_system import get_recommendation  # noqa: E402

st.set_page_config(page_title="Deteksi Penyakit Daun Apel", page_icon="🍏")

st.title("🍏 Deteksi Penyakit Daun Apel & Rekomendasi Penanganan")
st.write(
    "Unggah foto daun apel, sistem akan mendeteksi jenis penyakitnya "
    "menggunakan model Computer Vision (CNN) dan memberikan rekomendasi "
    "penanganan otomatis (Expert System)."
)


@st.cache_resource
def get_model_and_labels():
    return load_artifacts()


uploaded_file = st.file_uploader(
    "Unggah gambar daun apel (format JPG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diunggah", use_column_width=True)

        with st.spinner("Menganalisis gambar..."):
            model, idx_to_label = get_model_and_labels()
            label, confidence = predict_image(model, idx_to_label, uploaded_file)
            hasil = get_recommendation(label)

        st.success(f"Hasil Deteksi: **{hasil['status']}**")
        st.write(f"Tingkat keyakinan model: **{confidence * 100:.1f}%**")
        st.write(f"Tingkat urgensi: **{hasil['tingkat_urgensi']}**")
        st.info(f"Rekomendasi penanganan: {hasil['rekomendasi']}")

        # Respons dalam format JSON, sesuai persyaratan dokumentasi API di soal
        st.json({
            "status_code": 200,
            "predicted_label": label,
            "confidence": round(confidence, 4),
            "recommendation": hasil["rekomendasi"],
            "urgency_level": hasil["tingkat_urgensi"],
        })

    except Exception as e:
        st.error(
            "Terjadi kesalahan saat memproses gambar. "
            "Pastikan file yang diunggah adalah gambar yang valid (JPG/PNG)."
        )
        st.caption(f"Detail teknis (untuk debugging): {e}")
else:
    st.warning("Silakan unggah gambar daun apel terlebih dahulu untuk memulai analisis.")
