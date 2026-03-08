import os
import sys
import tensorflow as tf

def check_dlls():
    search_dirs = os.environ.get('PATH', '').split(os.pathsep)
    target_dlls = [
        "cudart64_110.dll",
        "cublas64_11.dll",
        "cublasLt64_11.dll",
        "cufft64_10.dll",
        "curand64_10.dll",
        "cusolver64_11.dll",
        "cusparse64_11.dll",
        "cudnn64_8.dll",
        "zlibwapi.dll"
    ]
    
    print("--- DLL Search Results ---")
    for dll in target_dlls:
        found = False
        for directory in search_dirs:
            if os.path.exists(os.path.join(directory, dll)):
                print(f"[FOUND] {dll} in {directory}")
                found = True
                break
        if not found:
            print(f"[MISSING] {dll}")

print(f"Python: {sys.version}")
print(f"TF Version: {tf.__version__}")
check_dlls()

# Log TensorFlow's own device check
print("\n--- TF Device Check ---")
print(f"Physical Devices: {tf.config.list_physical_devices()}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
