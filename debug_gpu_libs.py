import os
import tensorflow as tf
import sys

print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")

# Check for Physical Devices
physical_devices = tf.config.list_physical_devices('GPU')
print(f"Physical GPU Devices: {physical_devices}")

# Check for Logical Devices
logical_devices = tf.config.list_logical_devices('GPU')
print(f"Logical GPU Devices: {logical_devices}")

# Try to force GPU and catch errors
try:
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Successfully set memory growth")
except Exception as e:
    print(f"Error setting memory growth: {e}")

# Check for CUDA built-in
print(f"Is built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"Is device name: {tf.test.gpu_device_name()}")

# Print Library Paths
print("\nEnvironment Variables:")
for key in ['PATH', 'CUDA_PATH', 'CUDA_PATH_V11_2', 'CUDA_PATH_V11_3', 'CUDA_PATH_V12_0']:
    print(f"{key}: {os.environ.get(key, 'Not Set')[:100]}...")
