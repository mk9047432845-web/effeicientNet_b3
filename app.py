from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, requests
import numpy as np
from PIL import Image
import tensorflow as tf

# =====================
# CONFIG & PATHS
# =====================
app = Flask(__name__)
CORS(app)

MODEL_DIR = "models_cache"
UPLOAD_FOLDER = "uploads"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLASS_LABELS = ["benign", "malignant", "normal"]
ALLOWED_EXT = {"jpg", "jpeg", "png"}

# =====================
# MODEL CONFIG (MobileNetV2 H5)
# =====================
MODEL_URL = (
    "https://huggingface.co/mani880740255/skin_care_tflite/"
    "resolve/main/skin_cancer_mobilenetv2%20(1).h5"
)
MODEL_PATH = os.path.join(MODEL_DIR, "skin_cancer_mobilenetv2.h5")

model = None  # global model cache

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
# HELPERS
# =====================
def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def ensure_model_exists():
    global model

    if model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        print("Downloading MobileNetV2 (.h5) model...")
        r = requests.get(MODEL_URL, stream=True)
        if r.status_code != 200:
            raise Exception("Model download failed")
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

    print("Loading MobileNetV2 model...")
    model = tf.keras.models.load_model(MODEL_PATH)

# =====================
# MOBILENETV2 PREDICT
# =====================
def predict_mobilenet(img_path):
    ensure_model_exists()

    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))          # MobileNetV2 input size
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
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
        return jsonify({"error": "Image required"}), 400

    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    try:
        idx, probs = predict_mobilenet(path)
        return jsonify({
            "model_used": "mobilenetv2",
            "prediction": CLASS_LABELS[idx],
            "confidence": float(probs[idx]),
            "probabilities": {
                CLASS_LABELS[i]: float(probs[i]) for i in range(3)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)

# =====================
# CHATBOT ROUTE
# =====================
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    user_msg = data.get("message", "").lower().strip()

    if user_msg == "":
        return jsonify({
            "suggestions": list(CHAT_RESPONSES.keys())[:3]
        })

    if user_msg in CHAT_RESPONSES:
        return jsonify({
            "reply": CHAT_RESPONSES[user_msg],
            "suggestions": []
        })

    return jsonify({
        "reply": "I'm sorry, I only answer specific skin health questions. Try using the suggested buttons.",
        "suggestions": list(CHAT_RESPONSES.keys())
    })

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
