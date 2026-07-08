"""
evaluate.py
-----------
Script evaluasi menyeluruh untuk model CNN deteksi penyakit daun apel.
Menghitung: Accuracy, Precision, Recall, F1-Score (per kelas & rata-rata),
serta Confusion Matrix.

Ini yang dipakai untuk bagian "The Value" di artikel Medium,
BUKAN hanya val_accuracy yang muncul saat training.

Cara pakai:
    python src/evaluate.py
"""

import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DATA_DIR = "data/raw"
MODEL_PATH = "models/cnn_apple_disease.h5"
LABELS_PATH = "models/class_labels.json"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32


def main():
    # Load model dan label
    model = load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        idx_to_label = json.load(f)

    # Siapkan data validasi (subset yang sama seperti saat training,
    # supaya konsisten dengan split yang dipakai train.py)
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,  # penting: jangan diacak, supaya label & prediksi sinkron
    )

    print("Menjalankan prediksi terhadap seluruh data validasi...\n")
    y_pred_probs = model.predict(val_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes

    class_names = [idx_to_label[str(i)] for i in range(len(idx_to_label))]

    # ---------- METRIK UTAMA ----------
    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro")
    recall_macro = recall_score(y_true, y_pred, average="macro")
    f1_macro = f1_score(y_true, y_pred, average="macro")

    print("=" * 55)
    print("HASIL EVALUASI MODEL (data validasi)")
    print("=" * 55)
    print(f"Accuracy           : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Precision (macro)  : {precision_macro:.4f}")
    print(f"Recall (macro)     : {recall_macro:.4f}")
    print(f"F1-Score (macro)   : {f1_macro:.4f}")
    print("=" * 55)

    # ---------- METRIK PER KELAS ----------
    print("\nRincian per kelas:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # ---------- CONFUSION MATRIX ----------
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix (baris = label asli, kolom = prediksi):")
    print("Urutan kelas:", class_names)
    print(cm)

    # ---------- SIMPAN HASIL KE FILE (buat dilampirkan di artikel/README) ----------
    results = {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    with open("models/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nHasil evaluasi lengkap tersimpan di: models/evaluation_results.json")
    print("Gunakan angka-angka ini untuk bagian 'The Value' di artikel Medium kamu.")


if __name__ == "__main__":
    main()