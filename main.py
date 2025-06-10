from flask import Flask, request, jsonify
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

app = Flask(__name__)

# Path lokal untuk menyimpan model
MODEL_PATH = 'model.h5'

# ID Google Drive dan URL download
GDRIVE_ID = '1ZfRZXMj4qiBSRKookrB3SW5w_IWFrv-S'
GDRIVE_URL = f'https://drive.google.com/uc?id={GDRIVE_ID}'

# Cek dan download model jika belum ada
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# Load model TensorFlow
model = tf.keras.models.load_model(MODEL_PATH)

# Fungsi untuk memproses gambar
def prepare_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # normalisasi
    image_array = np.expand_dims(image_array, axis=0)  # buat batch
    return image_array

@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})

# Endpoint untuk prediksi gambar
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    try:
        image = Image.open(file.stream)
        processed_image = prepare_image(image)
        prediction = model.predict(processed_image)

        # Jika model klasifikasi, bisa pakai argmax:
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
