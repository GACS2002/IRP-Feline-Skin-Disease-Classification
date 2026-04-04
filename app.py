from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import timm
import joblib
import numpy as np
from torchvision import transforms
from PIL import Image
import io
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# ── Class mapping (update if class_to_idx differs) ───────
CLASS_NAMES = ["Flea Allergy", "Health", "Ringworm", "Scabies"]

# ── Load models once at startup ───────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

deit = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=0)
deit.load_state_dict(torch.load("models/deit_feature_extractor.pth", map_location=device))
deit.to(device)
deit.eval()

pca = joblib.load("models/pca_transformer.pkl")
svm = joblib.load("models/svm_pca_model.pkl")

# ── Preprocessing ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = deit(tensor).cpu().numpy()

        pca_features = pca.transform(embedding)
        pred_idx     = svm.predict(pca_features)[0]
        pred_class   = CLASS_NAMES[pred_idx]

        scores = svm.decision_function(pca_features)[0]
        e = np.exp(scores - np.max(scores))
        proba = e / e.sum()
        confidence = {CLASS_NAMES[i]: round(float(p) * 100, 1) for i, p in enumerate(proba)}

        return jsonify({
            "prediction": pred_class,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
