import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "dataset/New Plant Diseases Dataset(Augmented)" # Default path
MODEL_PATH = "mobilenetv2_best.keras"
CLASS_INDICES_PATH = "class_indices.json"

def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Please run train_model.py first.")
        return

    if not os.path.exists(os.path.join(DATA_DIR, 'valid')):
        print(f"Error: Validation data directory not found in '{DATA_DIR}'.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)

    val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    print("Loading validation dataset...")
    validation_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'valid'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False # Very important for evaluation matching predictions to true labels
    )

    print("\nEvaluating model on validation data...")
    loss, accuracy = model.evaluate(validation_generator)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy*100:.2f}%")

    print("\nGenerating predictions for detailed metrics...")
    predictions = model.predict(validation_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = validation_generator.classes
    
    class_labels = list(validation_generator.class_indices.keys())
    
    print("\nClassification Report:")
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)

    # Confusion Matrix
    print("Generating confusion matrix plot...")
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d", xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to 'confusion_matrix.png'.")

if __name__ == "__main__":
    evaluate()
