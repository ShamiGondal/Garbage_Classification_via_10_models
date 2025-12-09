"""Check CUDA setup and provide instructions"""
import os
import sys

print("Checking CUDA setup...")
print("="*60)

# Check for CUDA in common locations
cuda_paths = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA",
    os.environ.get("CUDA_PATH", ""),
    os.environ.get("CUDA_HOME", ""),
]

print("\nChecking CUDA installation paths:")
cuda_found = False
for path in cuda_paths:
    if path and os.path.exists(path):
        print(f"✓ Found: {path}")
        cuda_found = True
        # List versions
        try:
            versions = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith('v')]
            print(f"  CUDA versions: {versions}")
        except:
            pass
    elif path:
        print(f"✗ Not found: {path}")

if not cuda_found:
    print("\n⚠ CUDA Toolkit not found in standard locations")
    print("TensorFlow on Windows requires CUDA Toolkit installed from NVIDIA")

print("\n" + "="*60)
print("SOLUTION:")
print("="*60)
print("""
For GPU support on Windows, you need to:

1. Install CUDA Toolkit 12.x from NVIDIA:
   https://developer.nvidia.com/cuda-downloads

2. Install cuDNN from NVIDIA (requires free account):
   https://developer.nvidia.com/cudnn

3. Add CUDA to PATH:
   - Add: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin
   - Add: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\libnvvp

4. Restart your terminal/Python environment

ALTERNATIVE: Use WSL2 (Windows Subsystem for Linux) which has
better TensorFlow GPU support, or use Google Colab for free GPU.

For now, the code will run on CPU (slower but functional).
""")

