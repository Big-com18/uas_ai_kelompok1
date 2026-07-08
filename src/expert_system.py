"""
expert_system.py
-----------------
Modul rule-based sederhana yang menjadi "teknik AI ke-2" dalam project ini.
Input: label penyakit hasil prediksi CNN
Output: rekomendasi penanganan (dictionary)

Ini adalah bagian integrasi/orchestration yang diminta di soal:
output model Computer Vision (CNN) menjadi input bagi Expert System.
"""

RULES = {
    "Apple___healthy": {
        "status": "Sehat",
        "rekomendasi": (
            "Tanaman dalam kondisi sehat. Lanjutkan perawatan rutin: "
            "penyiraman teratur, pemupukan berimbang, dan pemantauan berkala."
        ),
        "tingkat_urgensi": "Rendah",
    },
    "Apple___Apple_scab": {
        "status": "Apple Scab (Kudis Apel)",
        "rekomendasi": (
            "Semprotkan fungisida berbahan aktif captan atau myclobutanil. "
            "Buang dan musnahkan daun yang gugur di sekitar pohon karena jamur "
            "bertahan di sana. Perbaiki sirkulasi udara dengan pemangkasan."
        ),
        "tingkat_urgensi": "Sedang",
    },
    "Apple___Black_rot": {
        "status": "Black Rot (Busuk Hitam)",
        "rekomendasi": (
            "Pangkas dan musnahkan cabang atau buah yang terinfeksi segera. "
            "Hindari pemupukan nitrogen berlebih. Gunakan fungisida tembaga "
            "pada fase awal pertumbuhan daun."
        ),
        "tingkat_urgensi": "Tinggi",
    },
    "Apple___Cedar_apple_rust": {
        "status": "Cedar Apple Rust (Karat Daun)",
        "rekomendasi": (
            "Jika memungkinkan, jauhkan tanaman apel dari pohon cedar/juniper "
            "dalam radius 2-3 km karena jamur ini butuh dua inang. "
            "Aplikasikan fungisida preventif saat musim semi sebelum gejala muncul."
        ),
        "tingkat_urgensi": "Sedang",
    },
}

DEFAULT_RESPONSE = {
    "status": "Tidak Diketahui",
    "rekomendasi": "Label tidak dikenali oleh sistem. Silakan periksa kembali model atau data input.",
    "tingkat_urgensi": "-",
}


def get_recommendation(predicted_label: str) -> dict:
    """
    Ambil rekomendasi penanganan berdasarkan label prediksi CNN.

    Args:
        predicted_label: nama kelas persis seperti pada folder dataset,
                          contoh: "Apple___Black_rot"

    Returns:
        dict berisi status, rekomendasi, dan tingkat urgensi
    """
    return RULES.get(predicted_label, DEFAULT_RESPONSE)
