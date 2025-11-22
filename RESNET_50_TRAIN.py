"""Training script for the Dental Cavity Detection ResNet50 model."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_OUTPUT = BASE_DIR / "ResNet50_model.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.pkl"
TRAIN_DIR = BASE_DIR / "Train"
TEST_DIR = BASE_DIR / "test"

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42


def create_data_generators():
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError("Train/test directories not found. Check dataset paths.")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    class_names = list(train_generator.class_indices.keys())
    with CLASS_NAMES_PATH.open("wb") as f:
        pickle.dump(class_names, f)

    return train_generator, test_generator, class_names


def build_model(num_classes: int):
    base_model = ResNet50(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False

    model = Sequential(
        [
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.3),
            Dense(256, activation="relu"),
            Dropout(0.2),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_training_curves(history):
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.history["accuracy"], label="Training Accuracy")
    ax.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(STATIC_DIR / "Accu_plt.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history.history["loss"], label="Training Loss")
    ax.plot(history.history["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(STATIC_DIR / "loss_plt.png")
    plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, class_names):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(STATIC_DIR / "confusion_matrix.jpg")
    plt.close(fig)


def save_f1_graph(y_true, y_pred, class_names):
    f1_scores = f1_score(y_true, y_pred, average=None)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(class_names, f1_scores, color="teal")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score by Class")
    fig.tight_layout()
    fig.savefig(STATIC_DIR / "f1_graph.jpg")
    plt.close(fig)


def evaluate_model(model, test_generator, class_names):
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix")
    print(cm)
    print("Classification Report")
    print(classification_report(y_true, y_pred, target_names=class_names))

    save_confusion_matrix(cm, class_names)
    save_f1_graph(y_true, y_pred, class_names)


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    train_generator, test_generator, class_names = create_data_generators()
    model = build_model(len(class_names))

    steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
    validation_steps = max(1, test_generator.samples // BATCH_SIZE)

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_generator,
        validation_steps=validation_steps,
    )

    plot_training_curves(history)
    model.save(MODEL_OUTPUT)
    evaluate_model(model, test_generator, class_names)


if __name__ == "__main__":
    main()
