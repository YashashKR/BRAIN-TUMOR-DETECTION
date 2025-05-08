from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join("model", "final_brain_tumor_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

# Class names
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Define function to make predictions
def predict(image):
    img = image.resize((224, 224))  # Resize to model input size
    img = np.array(img) / 255.0     # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    predictions = model.predict(img)
    result = CLASS_NAMES[np.argmax(predictions)]
    return result

@app.route('/')
def index():
    return render_template('index.html')  # Renders HTML page

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"})
    
    image = Image.open(file)
    result = predict(image)
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)