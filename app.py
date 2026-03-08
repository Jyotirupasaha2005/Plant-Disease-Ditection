import os
import io
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# DLL configuration for GPU environment
env_base = r"C:\Users\ROSHAN MISHRA\anaconda3\envs\plantcare_gpu"
dll_bin = os.path.join(env_base, "Library", "bin")

if os.path.exists(dll_bin):
    try:
        os.add_dll_directory(dll_bin)
        os.environ['PATH'] = dll_bin + os.pathsep + os.environ['PATH']
        print(f"Injected GPU DLL path: {dll_bin}")
    except Exception as e:
        print(f"Error configuring DLL path: {e}")

import tensorflow as tf
from tensorflow import keras
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = 'mobilenetv2_best.keras'
CLASS_INDICES_PATH = 'class_indices.json'
IMG_SIZE = (224, 224)

# Load Model
model = None
class_names = None

def load_resources():
    global model, class_names
    error_log = []
    
    if os.path.exists(MODEL_PATH):
        try:
            model = keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            msg = f"Error loading model: {e}"
            print(msg)
            error_log.append(msg)
    else:
        msg = f"Warning: Model not found at {MODEL_PATH}."
        print(msg)
        error_log.append(msg)

    if os.path.exists(CLASS_INDICES_PATH):
        try:
            with open(CLASS_INDICES_PATH, 'r') as f:
                class_indices = json.load(f)
                class_names = {v: k for k, v in class_indices.items()}
            print("Class labels loaded.")
        except Exception as e:
            msg = f"Error loading class labels: {e}"
            print(msg)
            error_log.append(msg)
    else:
        msg = f"Warning: Class labels not found at {CLASS_INDICES_PATH}."
        print(msg)
        error_log.append(msg)
    
    return error_log

load_resources()

def prepare_image(img_path):
    # Load the image
    img = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    # Convert to array
    img_array = keras.preprocessing.image.img_to_array(img)
    # Expand dimensions to match batch size
    img_array = np.expand_dims(img_array, axis=0)
    # Rescale consistently with training
    img_array = img_array / 255.0
    return img_array

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, class_names
    # Reload resources if not already loaded
    load_errors = []
    if model is None or class_names is None:
        load_errors = load_resources()
        
    if model is None or class_names is None:
        error_msg = 'Model or class definitions not loaded. '
        if load_errors:
            error_msg += f" Details: {'; '.join(load_errors)}"
        return jsonify({'error': error_msg}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            
            # Predict
            processed_image = prepare_image(filepath)
            predictions = model.predict(processed_image)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence = float(np.max(predictions))
            
            # Format the output replacing underscores with spaces
            predicted_class_name = class_names.get(predicted_class_index, "Unknown Class")
            formatted_name = predicted_class_name.replace('___', ' - ').replace('_', ' ')
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'prediction': formatted_name,
                'confidence': f"{confidence * 100:.2f}%"
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
