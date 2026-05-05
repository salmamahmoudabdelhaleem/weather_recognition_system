"""
Weather Classifier - Flask Backend
Run this in a Jupyter cell:

    from flask_app import app
    app.run(port=5000, debug=False, use_reloader=False)

Then open: http://localhost:5000
"""

import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# ── Move index.html to templates if needed ───────────────────
os.makedirs('templates', exist_ok=True)
if os.path.exists('static/index.html') and not os.path.exists('templates/index.html'):
    os.rename('static/index.html', 'templates/index.html')

# ── Load model once at startup ────────────────────────────────
MODEL_PATH = "weather_best.keras"
model = load_model(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")

CLASS_NAMES = ['dew', 'fogsmog', 'frost', 'glaze', 'hail',
               'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

app = Flask(__name__)

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    try:
        # Read and preprocess image
        img_bytes   = file.read()
        pil_img     = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        pil_resized = pil_img.resize((300, 300))

        img_array = np.array(pil_resized).astype('float32')
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        prediction  = model.predict(img_array, verbose=0)
        pred_idx    = int(np.argmax(prediction))
        pred_class  = CLASS_NAMES[pred_idx]
        confidence  = float(np.max(prediction)) * 100

        # All class probabilities
        all_probs = {
            CLASS_NAMES[i]: round(float(prediction[0][i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        return jsonify({
            'predicted_class': pred_class,
            'confidence':      round(confidence, 2),
            'all_probs':       all_probs
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=False, use_reloader=False)