from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import requests
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

# =====================
# APP CONFIG
# =====================
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_cache")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {"jpg", "jpeg", "png"}
CLASS_LABELS = ["benign", "malignant", "normal"]

# =====================
# MODEL CONFIG (TFLITE)
# =====================
MODEL_URL = (
    "https://huggingface.co/mani880740255/skin_care_tflite/"
    "resolve/main/skin_cancer_mobilenetv2.tflite"
)
MODEL_PATH = os.path.join(MODEL_DIR, "skin_cancer_mobilenetv2.tflite")

interpreter = None
input_details = None
output_details = None

# =====================
# CHAT DATA
# =====================
CHAT_RESPONSES = {
    "what is skin care?": "Skin care is the practice of maintaining healthy, clean, and protected skin.",
    "what is a benign lesion?": "A benign lesion is non-cancerous and does not spread.",
    "what is a malignant lesion?": "A malignant lesion is cancerous and can spread.",
    "signs of skin cancer": "Irregular shape, color change, bleeding, rapid growth.",
    "how to prevent skin cancer?": "Use sunscreen, avoid excess sun, wear protective clothing."
}

# =====================
# UTILITIES
# =====================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def load_model():
    """Download and load TFLite model (lazy loading)."""
    global interpreter, input_details, output_details

    if interpreter is not None:
        return

    if not os.path.exists(MODEL_PATH):
        print("Downloading TFLite model...")
        r = requests.get(MODEL_URL, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed to download model")

        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print("Loading TFLite model...")
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


def preprocess_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_image(image_path: str):
    load_model()

    img = preprocess_image(image_path)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = int(np.argmax(preds))

    return idx, preds.tolist()

# =====================
# ROUTES
# =====================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Image file required"}), 400

    file = request.files["image"]

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid image file"}), 400

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    try:
        idx, probs = predict_image(save_path)
        return jsonify({
            "model": "MobileNetV2 (TFLite)",
            "prediction": CLASS_LABELS[idx],
            "confidence": float(probs[idx]),
            "probabilities": {
                CLASS_LABELS[i]: float(probs[i]) for i in range(len(CLASS_LABELS))
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    msg = data.get("message", "").lower().strip()

    if not msg:
        return jsonify({
            "reply": "",
            "suggestions": list(CHAT_RESPONSES.keys())[:3]
        })

    if msg in CHAT_RESPONSES:
        return jsonify({
            "reply": CHAT_RESPONSES[msg],
            "suggestions": []
        })

    return jsonify({
        "reply": "I can answer basic skin health questions. Please use the suggestions.",
        "suggestions": list(CHAT_RESPONSES.keys())
    })

# =====================
# ENTRY POINT
# =====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
