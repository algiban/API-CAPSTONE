from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model/model_buah.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Label sesuai model
labels = ['alpukat', 'anggur', 'apel', 'belimbing', 'blueberry', 'buah naga', 'ceri', 'delima', 'duku', 'durian', 'jambu air', 'jambu biji', 'jeruk', 'kelapa', 'kiwi', 'kurma', 'leci', 'mangga', 'manggis', 'markisa', 'melon', 'nanas', 'nangka', 'pepaya', 'pir', 'pisang', 'rambutan', 'salak', 'sawo', 'semangka', 'sirsak', 'stroberi', 'tomat']

# Path file JSON
JSON_FILE = 'predictions.json'

# Fungsi simpan riwayat prediksi
def save_prediction_to_json(label, confidence):
    data = {
        "label": label,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat()
    }

    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_data.insert(0, data)

    with open(JSON_FILE, 'w') as f:
        json.dump(existing_data, f, indent=4)

@app.route('/')
def index():
    return "TFLite Prediction API is Running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        img = Image.open(file.stream).resize((224, 224)).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Sesuaikan bentuk input ke model
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        class_index = np.argmax(predictions)
        confidence = float(np.max(predictions))
        predicted_label = labels[class_index]

        # Simpan prediksi
        save_prediction_to_json(predicted_label, confidence)

        return jsonify({
            'prediction': predicted_label,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    if not os.path.exists(JSON_FILE):
        return jsonify([])

    with open(JSON_FILE, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
