"""
Quick test script to verify the setup is correct
"""

import sys
import os

print("Testing setup...")
print("="*60)

# Test imports
try:
    import tensorflow as tf
    print(f"✓ TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"✗ TensorFlow import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"✓ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print(f"✓ Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"✗ Pandas import failed: {e}")
    sys.exit(1)

try:
    from tensorflow import keras
    from tensorflow.keras import layers, models
    print("✓ Keras imports successful")
except ImportError as e:
    print(f"✗ Keras import failed: {e}")
    sys.exit(1)

# Test GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU detected: {len(gpus)} GPU(s)")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
else:
    print("⚠ No GPU detected - training will use CPU (slower)")

# Test data directory
data_dir = "Garbage classification/Garbage classification"
if os.path.exists(data_dir):
    print(f"✓ Data directory found: {data_dir}")
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        if os.path.exists(cls_path):
            count = len([f for f in os.listdir(cls_path) if f.endswith('.jpg')])
            print(f"  {cls}: {count} images")
else:
    print(f"✗ Data directory not found: {data_dir}")

# Test split files
split_files = [
    "one-indexed-files-notrash_train.txt",
    "one-indexed-files-notrash_val.txt",
    "one-indexed-files-notrash_test.txt"
]
for split_file in split_files:
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            lines = len(f.readlines())
        print(f"✓ {split_file}: {lines} samples")
    else:
        print(f"✗ {split_file} not found")

print("="*60)
print("Setup test complete!")
print("\nIf all checks passed, you can run: python train_garbage_classification.py")

