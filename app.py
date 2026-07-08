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
    try:
        return load_artifacts()
    except RuntimeError as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()


model, idx_to_label = get_model_and_labels()

uploaded_file = st.file_uploader(
    "Unggah gambar daun apel (format JPG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Preview gambar - dibungkus try-except karena file bisa jadi bukan gambar valid
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)
    except Exception as e:
        error_message = (
            "File yang diunggah tidak dapat dibuka sebagai gambar. "
            "Harap unggah file berformat JPG atau PNG."
        )
        st.error(error_message)
        st.json({
            "status_code": 400,
            "error": error_message,
            "detail": str(e),
        })
        st.stop()

    with st.spinner("Menganalisis gambar..."):
        # uploaded_file sudah "dibaca" oleh Image.open di atas, reset pointer dulu
        uploaded_file.seek(0)
        result = predict_image(model, idx_to_label, uploaded_file)

    if not result["success"]:
        # Human-readable error message, bukan technical traceback
        st.error(result["error"])
    else:
        label = result["label"]
        confidence = result["confidence"]
        hasil = get_recommendation(label)

        st.success(f"Hasil Deteksi: **{hasil['status']}**")
        st.write(f"Tingkat keyakinan model: **{confidence * 100:.1f}%**")
        st.write(f"Tingkat urgensi: **{hasil['tingkat_urgensi']}**")
        st.info(f"Rekomendasi penanganan: {hasil['rekomendasi']}")

        # Peringatan tambahan kalau confidence rendah
        if confidence < 0.6:
            st.warning(
                "⚠️ Tingkat keyakinan model tergolong rendah. "
                "Coba unggah foto dengan pencahayaan lebih baik atau fokus lebih jelas."
            )

        # Respons dalam format JSON, sesuai persyaratan dokumentasi API di soal
        st.json({
            "status_code": 200,
            "predicted_label": label,
            "confidence": round(confidence, 4),
            "recommendation": hasil["rekomendasi"],
            "urgency_level": hasil["tingkat_urgensi"],
        })
else:
    st.warning("Silakan unggah gambar daun apel terlebih dahulu untuk memulai analisis.")