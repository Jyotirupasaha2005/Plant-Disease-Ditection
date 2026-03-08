import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
for i, device in enumerate(tf.config.list_physical_devices('GPU')):
    print(f"Device {i}: {device}")
