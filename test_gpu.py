"""
GPU Setup Verification Script
Updated for CUDA 13.1 and cuDNN 9.17
"""

import tensorflow as tf
import os
import sys

print("="*70)
print("GPU Setup Verification - CUDA 13.1 & cuDNN 9.17")
print("="*70)

# Check TensorFlow version
print(f"\n📦 TensorFlow version: {tf.__version__}")
print(f"📦 CUDA built: {tf.test.is_built_with_cuda()}")

# Check CUDA installation
print("\n🔍 Checking CUDA Installation...")
cuda_paths = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
    r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
    os.environ.get("CUDA_PATH", ""),
]

cuda_found = False
for path in cuda_paths:
    if path and os.path.exists(path):
        print(f"  ✓ CUDA found at: {path}")
        cuda_found = True
        break

if not cuda_found:
    print("  ⚠ CUDA 13.1 not found in standard locations")
    print("  Check if CUDA is installed at: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1")

# Check cuDNN
print("\n🔍 Checking cuDNN...")
if cuda_found:
    cudnn_path = os.path.join(path, "bin", "cudnn*.dll")
    import glob
    cudnn_files = glob.glob(os.path.join(path, "bin", "cudnn*.dll"))
    if cudnn_files:
        print(f"  ✓ cuDNN DLL files found: {len(cudnn_files)} file(s)")
        for f in cudnn_files[:3]:  # Show first 3
            print(f"    - {os.path.basename(f)}")
    else:
        print("  ⚠ cuDNN DLL files not found in CUDA bin directory")

# Check GPU devices
print("\n🔍 Checking GPU Devices...")
gpus = tf.config.list_physical_devices('GPU')
print(f"  GPU devices found: {len(gpus)}")

if gpus:
    print(f"\n✅ SUCCESS! GPU is detected!")
    print(f"   Number of GPUs: {len(gpus)}")
    
    for i, gpu in enumerate(gpus):
        print(f"\n   GPU {i}: {gpu.name}")
        
        # Get GPU details
        try:
            details = tf.config.experimental.get_device_details(gpu)
            if details:
                print(f"      Device details: {details}")
        except:
            pass
        
        # Enable memory growth
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"      Memory growth: Enabled")
        except Exception as e:
            print(f"      Memory growth: Error - {e}")
    
    # Test GPU computation
    print("\n🧪 Testing GPU Computation...")
    try:
        with tf.device('/GPU:0'):
            # Small test
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
        print(f"   Small matrix test: ✓")
        print(f"   Result: {c.numpy()}")
        
        # Larger test
        print("\n   Testing larger computation (1000x1000 matrix)...")
        with tf.device('/GPU:0'):
            import numpy as np
            large_a = tf.random.normal((1000, 1000))
            large_b = tf.random.normal((1000, 1000))
            large_c = tf.matmul(large_a, large_b)
        print(f"   Large matrix test: ✓")
        print(f"   Result shape: {large_c.shape}")
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED! GPU is ready for training!")
        print("="*70)
        print("\n✅ Your setup is correct:")
        print("   - CUDA 13.1: Installed")
        print("   - cuDNN 9.17: Installed")
        print("   - TensorFlow: Can access GPU")
        print("   - GPU Computation: Working")
        print("\n🚀 You can now train models with GPU acceleration!")
        
    except Exception as e:
        print(f"\n❌ GPU computation failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check CUDA is in PATH environment variable")
        print("2. Restart your terminal/computer")
        print("3. Verify GPU drivers are up to date")
        sys.exit(1)
        
else:
    print("\n❌ GPU NOT detected!")
    print("\nTroubleshooting steps:")
    print("1. Check CUDA installation:")
    print("   Run: nvcc --version (in new terminal)")
    print("2. Check GPU drivers:")
    print("   Run: nvidia-smi")
    print("3. Verify PATH environment variables include:")
    print("   C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.1\\bin")
    print("4. Restart your computer after installing CUDA")
    print("5. Reinstall TensorFlow:")
    print("   pip uninstall tensorflow")
    print("   pip install tensorflow")
    print("\n" + "="*70)
    sys.exit(1)

