"""
api.py
------
REST API untuk Deteksi Penyakit Daun Apel & Rekomendasi Penanganan.

Endpoint:
    GET  /              -> health check
    POST /predict        -> terima gambar (multipart/form-data), kembalikan JSON

Cara jalankan (lokal):
    uvicorn src.api:app --reload --port 8000

Dokumentasi otomatis (Swagger UI) bisa diakses di:
    http://localhost:8000/docs
"""

import io
import os
import sys

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

sys.path.append(os.path.dirname(__file__))
from predict import load_artifacts, predict_image  # noqa: E402
from expert_system import get_recommendation  # noqa: E402

app = FastAPI(
    title="Apple Disease Detection API",
    description="REST API untuk deteksi penyakit daun apel menggunakan CNN + Expert System",
    version="1.0.0",
)

# Model di-load sekali saja saat server start (bukan setiap request)
try:
    model, idx_to_label = load_artifacts()
    MODEL_READY = True
except RuntimeError as e:
    print(f"[WARNING] Model gagal dimuat saat startup: {e}")
    model, idx_to_label = None, None
    MODEL_READY = False

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@app.get("/")
def health_check():
    """Health check endpoint - cek apakah API dan model siap dipakai."""
    return {
        "status": "ok",
        "model_ready": MODEL_READY,
        "message": "Apple Disease Detection API is running",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Terima file gambar daun apel, kembalikan hasil deteksi + rekomendasi.

    Request : multipart/form-data, field 'file' berisi gambar JPG/PNG
    Response 200 (sukses):
        {
            "status_code": 200,
            "predicted_label": "Apple___Black_rot",
            "confidence": 0.9532,
            "recommendation": "...",
            "urgency_level": "Tinggi"
        }
    Response 400 (gagal):
        {
            "status_code": 400,
            "error": "pesan error yang jelas"
        }
    """
    if not MODEL_READY:
        return JSONResponse(
            status_code=400,
            content={"status_code": 400, "error": "Model belum siap / gagal dimuat di server."},
        )

    # Validasi tipe file sebelum diproses (robustness)
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        return JSONResponse(
            status_code=400,
            content={
                "status_code": 400,
                "error": (
                    f"Format file '{file.content_type}' tidak didukung. "
                    "Harap unggah file JPG atau PNG."
                ),
            },
        )

    try:
        contents = await file.read()
        image_stream = io.BytesIO(contents)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status_code": 400, "error": f"Gagal membaca file: {e}"},
        )

    result = predict_image(model, idx_to_label, image_stream)

    if not result["success"]:
        return JSONResponse(
            status_code=400,
            content={"status_code": 400, "error": result["error"]},
        )

    hasil = get_recommendation(result["label"])

    return JSONResponse(
        status_code=200,
        content={
            "status_code": 200,
            "predicted_label": result["label"],
            "confidence": round(result["confidence"], 4),
            "recommendation": hasil["rekomendasi"],
            "urgency_level": hasil["tingkat_urgensi"],
        },
    )