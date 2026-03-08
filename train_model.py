import os
import tensorflow as tf

# DLL configuration for GPU 1 (RTX 2050)
# Force TensorFlow to find the local Conda libraries
env_base = r"C:\Users\ROSHAN MISHRA\anaconda3\envs\plantcare_gpu"
dll_bin = os.path.join(env_base, "Library", "bin")

if os.path.exists(dll_bin):
    try:
        # 1. Modern way (Python 3.8+)
        os.add_dll_directory(dll_bin)
        # 2. Legacy way (Prepending to PATH for sub-dependencies)
        os.environ['PATH'] = dll_bin + os.pathsep + os.environ['PATH']
        print(f"Injected GPU DLL path: {dll_bin}")
    except Exception as e:
        print(f"Error configuring DLL path: {e}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Specifically target the discrete NVIDIA GPU
        device_index = 0 # If only one shows up in this env, use 0
        tf.config.set_visible_devices(gpus[device_index], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[device_index], True)
        print(f"Using GPU: {gpus[device_index]}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, falling back to CPU.")
    print("Path Checked: ", dll_bin)

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 38
DATA_DIR = "dataset" # Updated to match structured dataset directory
EPOCHS = 20

def create_model():
    """Builds the MobileNetV2 transfer learning model as specified."""
    # Build the complete model
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    
    # Pre-trained MobileNetV2 Base
    base_model = keras.applications.MobileNetV2(
        weights="imagenet", 
        include_top=False, 
        input_tensor=inputs
    )
    
    # Freeze all layers initially
    for layer in base_model.layers:
        layer.trainable = False
        
    # Unfreeze the last 10 layers for fine-tuning
    N_LAST_LAYERS = 10
    if N_LAST_LAYERS > 0:
        for layer in base_model.layers[-N_LAST_LAYERS:]:
            layer.trainable = True

    # Count trainable parameters for info
    trainable_count = sum([tf.size(w).numpy() for w in base_model.trainable_weights])
    non_trainable_count = sum([tf.size(w).numpy() for w in base_model.non_trainable_weights])
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Non-trainable parameters: {non_trainable_count:,}")
    print(f"Unfrozen last {N_LAST_LAYERS} layers for fine-tuning")

    # Add custom classification head
    x = base_model(inputs, training=False) # Keep BatchNorm frozen
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name="mobilenetv2_plant_disease_classifier")
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        print("Please download and extract the dataset before running this script.")
        return

    # Data Generators handling train/val splits and augmentation
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    import json
    with open('class_indices.json', 'w') as f:
        json.dump(train_generator.class_indices, f)
    print("Saved class indices to class_indices.json")

    print("Loading validation data...")
    validation_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'valid'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    model = create_model()
    model.summary()

    # Callbacks
    callbacks = [
        # Save best model based on validation accuracy
        keras.callbacks.ModelCheckpoint(
            "mobilenetv2_best.keras",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Reduce learning rate when validation loss plateaus
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        ),
        # Stop training if no improvement
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    print("Callbacks configured:")
    print(" 1. ModelCheckpoint - Saves best model")
    print(" 2. ReduceLROnPlateau - Adjusts learning rate")
    print(" 3. EarlyStopping - Prevents overfitting")

    # Train the model
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.savefig('training_history.png')
    print("Training complete. Model saved to 'mobilenetv2_best.keras'. History plot saved to 'training_history.png'.")

if __name__ == "__main__":
    main()
