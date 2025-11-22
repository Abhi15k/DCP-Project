from __future__ import annotations

import os
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from flask import (Flask, flash, redirect, render_template, request,
                   url_for)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename

matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
IMAGE_OUTPUT_DIR = STATIC_DIR / "images"
DB_PATH = BASE_DIR / "user_data.db"
MODEL_PATH = BASE_DIR / "ResNet50_model.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.pkl"
ACC_PATH = BASE_DIR / "acc.txt"
IMG_SIZE = (150, 150)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")

IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def init_db() -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user(
                name TEXT PRIMARY KEY,
                password TEXT NOT NULL,
                mobile TEXT,
                email TEXT
            )
            """
        )
        connection.commit()


# Flask 3.x removed before_first_request, so eagerly initialize the DB once.
init_db()


def get_db_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing trained model at {MODEL_PATH}")
    return load_model(str(MODEL_PATH))


def load_class_names() -> List[str]:
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(f"Missing class names file at {CLASS_NAMES_PATH}")
    with CLASS_NAMES_PATH.open("rb") as f:
        return pickle.load(f)


try:
    MODEL = load_trained_model()
    MODEL_ERROR = None
except Exception as exc:  # pragma: no cover - surfaces at runtime
    MODEL = None
    MODEL_ERROR = str(exc)

try:
    CLASS_NAMES = load_class_names()
    CLASS_NAMES_ERROR = None
except Exception as exc:  # pragma: no cover - surfaces at runtime
    CLASS_NAMES = []
    CLASS_NAMES_ERROR = str(exc)


DEFAULT_METADATA = {
    "display": "Unknown",
    "treatment_title": "General Guidance",
    "treatment": ["Consult your dentist for a comprehensive evaluation."],
    "recommendation_title": "Everyday Care",
    "recommendation": ["Maintain consistent brushing and flossing habits."],
    "followup_title": "Follow-up",
    "followup": ["Schedule periodic dental visits to monitor oral health."],
    "spread": "Monitoring required to understand cavity spread.",
}

LABEL_METADATA: Dict[str, Dict[str, object]] = {
    "mild cavity": {
        "display": "Mild cavity",
        "treatment_title": "Medical Treatment",
        "treatment": ["Apply fluoride treatments to remineralize enamel."],
        "recommendation_title": "Recommendations",
        "recommendation": ["Reduce sugary snacks and acidic beverages."],
        "followup_title": "Follow-up",
        "followup": ["Monitor progression and adjust fluoride frequency."],
        "spread": "No spread detected in nearby teeth.",
    },
    "moderate cavity": {
        "display": "Moderate cavity",
        "treatment_title": "Medical Treatment",
        "treatment": ["Plan a dental filling (composite resin or amalgam)."],
        "recommendation_title": "Recommendations",
        "recommendation": ["Add calcium-rich foods like milk and leafy greens."],
        "followup_title": "Follow-up",
        "followup": ["Check fillings regularly and watch for recurrence."],
        "spread": "Possible spread to adjacent teeth if untreated.",
    },
    "no cavity": {
        "display": "No cavity",
        "treatment_title": "Oral Care",
        "treatment": ["Maintain routine oral hygiene."],
        "recommendation_title": "Recommendations",
        "recommendation": ["Brush twice daily with fluoride toothpaste and floss."],
        "followup_title": "Follow-up",
        "followup": ["Visit your dentist every 6 months."],
        "spread": "No cavity spread detected.",
    },
    "sevear cavity": {
        "display": "Severe cavity",
        "treatment_title": "Medical Treatment",
        "treatment": [
            "Consider root canal therapy if pulp is infected.",
            "Plan extraction if the damage cannot be repaired.",
        ],
        "recommendation_title": "Recommendations",
        "recommendation": ["Increase vitamin D intake (eggs, fish) to support teeth."],
        "followup_title": "Follow-up",
        "followup": ["Discuss dental implants or dentures after treatment."],
        "spread": "High chance of spread to neighboring teeth.",
    },
}


def predict_single_image(image_path: Path) -> Tuple[str, float]:
    if MODEL is None:
        raise RuntimeError(f"Model not available: {MODEL_ERROR}")
    if not CLASS_NAMES:
        raise RuntimeError(f"Class names not available: {CLASS_NAMES_ERROR}")

    img = load_img(str(image_path), target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = MODEL.predict(img_array, verbose=0)
    predicted_class_index = int(np.argmax(prediction))
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = float(prediction[0][predicted_class_index]) * 100
    return predicted_class, confidence


def clean_directory(directory: Path) -> None:
    for file in directory.glob("*"):
        if file.is_file():
            file.unlink()


def generate_preprocessing_images(image_path: Path) -> Dict[str, Path]:
    source = cv2.imread(str(image_path))
    if source is None:
        raise ValueError("Unable to read uploaded image.")

    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(source, 250, 254)
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(source, -1, kernel_sharpening)

    outputs = {
        "gray": STATIC_DIR / "gray.jpg",
        "edges": STATIC_DIR / "edges.jpg",
        "threshold": STATIC_DIR / "threshold.jpg",
        "sharpened": STATIC_DIR / "sharpened.jpg",
    }

    cv2.imwrite(str(outputs["gray"]), gray)
    cv2.imwrite(str(outputs["edges"]), edges)
    cv2.imwrite(str(outputs["threshold"]), threshold)
    cv2.imwrite(str(outputs["sharpened"]), sharpened)

    return outputs


def plot_prediction_graph(predicted_label: str) -> Path:
    categories = CLASS_NAMES or list(LABEL_METADATA.keys())
    values = [1.0 if label == predicted_label else 0.0 for label in categories]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar([label.title() for label in categories], values, color="maroon", width=0.3)
    ax.set_xlabel("Category")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1)
    ax.set_title("Model Confidence by Category")
    fig.tight_layout()

    output_path = STATIC_DIR / "matrix.png"
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def get_label_metadata(label: str) -> Dict[str, object]:
    return LABEL_METADATA.get(label, DEFAULT_METADATA)


@app.context_processor
def inject_status_flags():
    return {
        "model_error": MODEL_ERROR,
        "class_names_error": CLASS_NAMES_ERROR,
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/userlog", methods=["GET", "POST"])
def userlog():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        password = request.form.get("password", "").strip()

        if not name or not password:
            flash("Both username and password are required.", "warning")
            return redirect(url_for("home"))

        with get_db_connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM user WHERE name = ? AND password = ?",
                (name, password),
            ).fetchone()

        if row:
            flash("Login successful. You can now upload an image.", "success")
            return redirect(url_for("predict"))

        flash("Incorrect credentials. Please try again.", "danger")
        return redirect(url_for("home"))

    return redirect(url_for("home"))


@app.route("/userreg", methods=["GET", "POST"])
def userreg():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        password = request.form.get("password", "").strip()
        mobile = request.form.get("phone", "").strip()
        email = request.form.get("email", "").strip()

        if not name or not password or not email:
            flash("All fields are required for registration.", "warning")
            return redirect(url_for("home"))

        with get_db_connection() as connection:
            try:
                connection.execute(
                    "INSERT INTO user(name, password, mobile, email) VALUES (?, ?, ?, ?)",
                    (name, password, mobile, email),
                )
                connection.commit()
            except sqlite3.IntegrityError:
                flash("Username already exists. Please choose another.", "danger")
                return redirect(url_for("home"))

        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("home"))

    return redirect(url_for("home"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if MODEL is None or not CLASS_NAMES:
            flash("Model is not ready. Please train the model first.", "danger")
            return redirect(url_for("predict"))

        if "filename" not in request.files:
            flash("Please select an image file to upload.", "warning")
            return redirect(url_for("predict"))

        file = request.files["filename"]
        if not file or not file.filename:
            flash("Please select an image file to upload.", "warning")
            return redirect(url_for("predict"))

        filename = secure_filename(file.filename)
        if not filename:
            flash("Invalid filename provided.", "danger")
            return redirect(url_for("predict"))

        clean_directory(IMAGE_OUTPUT_DIR)
        image_path = IMAGE_OUTPUT_DIR / filename
        file.save(image_path)

        try:
            predicted_class, confidence = predict_single_image(image_path)
            generate_preprocessing_images(image_path)
            chart_path = plot_prediction_graph(predicted_class)
        except Exception as exc:  # pragma: no cover - surfaces at runtime
            flash(f"Failed to process image: {exc}", "danger")
            return redirect(url_for("predict"))

        metadata = get_label_metadata(predicted_class)
        confidence_text = f"{confidence:.2f}%"
        ACC_PATH.write_text(confidence_text)

        return render_template(
            "results.html",
            status=metadata["display"],
            status2=f"Accuracy: {confidence_text}",
            Treatment=metadata["treatment_title"],
            Treatment1=metadata["treatment"],
            Recommendation=metadata["recommendation_title"],
            Recommendation1=metadata["recommendation"],
            FollowUp=metadata["followup_title"],
            FollowUp1=metadata["followup"],
            spread=metadata["spread"],
            ImageDisplay=url_for("static", filename=f"images/{filename}"),
            ImageDisplay1=url_for("static", filename="gray.jpg"),
            ImageDisplay2=url_for("static", filename="edges.jpg"),
            ImageDisplay3=url_for("static", filename="threshold.jpg"),
            ImageDisplay4=url_for("static", filename="sharpened.jpg"),
            ImageDisplay5=url_for("static", filename=chart_path.name),
        )

    return render_template("userlog.html")


@app.route("/graph", methods=["GET"])
def graph():
    images = [
        url_for("static", filename="Accu_plt.png"),
        url_for("static", filename="loss_plt.png"),
        url_for("static", filename="f1_graph.jpg"),
        url_for("static", filename="confusion_matrix.jpg"),
    ]

    content = [
        "Accuracy Graph",
        "Loss Graph",
        "F1-Score Graph",
        "Confusion Matrix Graph",
    ]

    return render_template("graph.html", images=images, content=content)


@app.route("/logout")
def logout():
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true", use_reloader=False)
