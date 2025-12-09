"""
Complete GPU Setup Verification Script
Run this after completing GPU setup to verify everything works
"""

import tensorflow as tf
import sys

print("="*60)
print("GPU Setup Verification")
print("="*60)

print(f"\nTensorFlow version: {tf.__version__}")
print(f"CUDA built: {tf.test.is_built_with_cuda()}")

gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPU devices found: {len(gpus)}")

if gpus:
    print("\n✓ SUCCESS! GPU is detected!")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
        # Get GPU details
        try:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"    Device details: {details}")
        except:
            pass
        
        # Enable memory growth
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"    Memory growth: Enabled")
        except:
            pass
    
    # Test GPU computation
    print("\nTesting GPU computation...")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
        print(f"  Matrix multiplication result:")
        print(f"  {c.numpy()}")
        print("\n✓ GPU computation works!")
        
        # Test larger computation
        print("\nTesting larger computation (1000x1000 matrix)...")
        with tf.device('/GPU:0'):
            import numpy as np
            large_a = tf.random.normal((1000, 1000))
            large_b = tf.random.normal((1000, 1000))
            large_c = tf.matmul(large_a, large_b)
        print(f"  Result shape: {large_c.shape}")
        print("✓ Large computation successful!")
        
    except Exception as e:
        print(f"✗ GPU computation failed: {e}")
        sys.exit(1)
        
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED! GPU is ready for training!")
    print("="*60)
    
else:
    print("\n✗ GPU NOT detected!")
    print("\nTroubleshooting steps:")
    print("1. Check CUDA installation:")
    print("   Run: nvcc --version")
    print("2. Check GPU drivers:")
    print("   Run: nvidia-smi")
    print("3. Verify PATH environment variables include:")
    print("   C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin")
    print("4. Restart your computer after installing CUDA")
    print("5. Reinstall TensorFlow:")
    print("   pip uninstall tensorflow")
    print("   pip install tensorflow")
    print("\n" + "="*60)
    sys.exit(1)

