import os
import json
import tensorflow as tf

MODEL_PATH = 'mobilenetv2_best.keras'
CLASS_INDICES_PATH = 'class_indices.json'

print(f"Checking {MODEL_PATH}: {os.path.exists(MODEL_PATH)}")
print(f"Size: {os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 'N/A'}")

print(f"Checking {CLASS_INDICES_PATH}: {os.path.exists(CLASS_INDICES_PATH)}")

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Successfully loaded model via script.")
    except Exception as e:
        print(f"Failed to load model: {e}")

if os.path.exists(CLASS_INDICES_PATH):
    try:
        with open(CLASS_INDICES_PATH, 'r') as f:
            indices = json.load(f)
            print(f"Successfully loaded {len(indices)} class indices.")
    except Exception as e:
        print(f"Failed to load indices: {e}")
