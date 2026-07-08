"""
train.py
--------
Script untuk melatih model CNN klasifikasi penyakit daun apel.
Dataset: PlantVillage (subset Apple) - https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

Cara pakai:
    1. Download dataset dari Kaggle, ekstrak folder-folder kelas Apple ke: data/raw/
       Struktur yang diharapkan:
           data/raw/Apple___healthy/
           data/raw/Apple___Apple_scab/
           data/raw/Apple___Black_rot/
           data/raw/Apple___Cedar_apple_rust/
    2. Jalankan: python src/train.py
    3. Model hasil training akan tersimpan otomatis di models/cnn_apple_disease.h5
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

# ---------- KONFIGURASI ----------
DATA_DIR = "data/raw"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15
MODEL_OUTPUT_PATH = "models/cnn_apple_disease.h5"
LABELS_OUTPUT_PATH = "models/class_labels.json"


def build_data_generators():
    """Siapkan data training & validation dengan augmentasi sederhana."""
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )

    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )

    return train_gen, val_gen


def build_model(num_classes):
    """Arsitektur CNN sederhana - cukup untuk 4 kelas penyakit daun apel."""
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    if not os.path.isdir(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        raise FileNotFoundError(
            f"Folder '{DATA_DIR}' kosong atau tidak ditemukan. "
            "Download dataset PlantVillage lalu taruh folder kelas Apple di dalamnya."
        )

    os.makedirs("models", exist_ok=True)

    train_gen, val_gen = build_data_generators()
    num_classes = train_gen.num_classes
    print(f"Ditemukan {num_classes} kelas: {train_gen.class_indices}")

    model = build_model(num_classes)
    model.summary()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
    )

    model.save(MODEL_OUTPUT_PATH)
    print(f"Model tersimpan di: {MODEL_OUTPUT_PATH}")

    # Simpan mapping index -> nama kelas, dibutuhkan saat prediksi
    idx_to_label = {v: k for k, v in train_gen.class_indices.items()}
    with open(LABELS_OUTPUT_PATH, "w") as f:
        json.dump(idx_to_label, f, indent=2)
    print(f"Label kelas tersimpan di: {LABELS_OUTPUT_PATH}")

    # Ringkasan evaluasi akhir (untuk dicatat di artikel Medium bagian "The Value")
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"\nValidation Accuracy: {val_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
