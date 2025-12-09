"""
Garbage Classification - Training 10 Different Models
Semester Project - IDS
Uses GPU for training
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import (
    ResNet50, VGG16, MobileNetV2, DenseNet121, 
    InceptionV3, Xception, EfficientNetB0
)
# NASNetMobile might not be available in all TensorFlow versions
try:
    from tensorflow.keras.applications import NASNetMobile
    NASNET_AVAILABLE = True
except ImportError:
    NASNET_AVAILABLE = False
    print("Warning: NASNetMobile not available in this TensorFlow version")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

# Set GPU memory growth to avoid allocation errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU available: {len(gpus)} GPU(s)")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        # Set default GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
        print("Falling back to CPU")
else:
    print("⚠ No GPU detected. Using CPU (training will be slower).")
    print("\nTo enable GPU support on Windows:")
    print("1. Install CUDA Toolkit 12.x from: https://developer.nvidia.com/cuda-downloads")
    print("2. Install cuDNN from: https://developer.nvidia.com/cudnn")
    print("3. Add CUDA to your PATH environment variable")
    print("4. Restart your terminal and run again")
    print("\nProceeding with CPU training...")

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 6
CLASS_NAMES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
DATA_DIR = "Garbage classification/Garbage classification"

# Create directories for saving models and results
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

def load_data_from_splits():
    """Load data from train/test/val split files"""
    def parse_split_file(filename):
        data = []
        labels = []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    label = int(parts[1]) - 1  # Convert to 0-indexed
                    data.append(img_name)
                    labels.append(label)
        return data, labels
    
    train_files, train_labels = parse_split_file("one-indexed-files-notrash_train.txt")
    val_files, val_labels = parse_split_file("one-indexed-files-notrash_val.txt")
    test_files, test_labels = parse_split_file("one-indexed-files-notrash_test.txt")
    
    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)

class CustomDataGenerator(keras.utils.Sequence):
    """Custom data generator for loading images from filenames"""
    def __init__(self, filenames, labels, batch_size, img_size, datagen, shuffle=True):
        self.filenames = filenames
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.datagen = datagen
        self.shuffle = shuffle
        self.class_folders = self._find_class_folders()
        self.on_epoch_end()
    
    def _find_class_folders(self):
        """Find class folders"""
        class_folders = {}
        for class_name in CLASS_NAMES:
            class_path = os.path.join(DATA_DIR, class_name)
            if os.path.exists(class_path):
                class_folders[class_name] = class_path
        return class_folders
    
    def _find_image_path(self, filename):
        """Find the full path of an image"""
        for class_name, class_path in self.class_folders.items():
            img_path = os.path.join(class_path, filename)
            if os.path.exists(img_path):
                return img_path
        return None
    
    def __len__(self):
        return int(np.ceil(len(self.filenames) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_filenames = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_images = []
        batch_y = []
        
        for filename, label in zip(batch_filenames, batch_labels):
            img_path = self._find_image_path(filename)
            if img_path:
                img = keras.preprocessing.image.load_img(img_path, target_size=(self.img_size, self.img_size))
                img_array = keras.preprocessing.image.img_to_array(img)
                img_array = self.datagen.random_transform(img_array) if self.shuffle else img_array
                img_array = self.datagen.standardize(img_array)
                batch_images.append(img_array)
                batch_y.append(keras.utils.to_categorical(label, NUM_CLASSES))
        
        return np.array(batch_images), np.array(batch_y)
    
    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.filenames))
            np.random.shuffle(indices)
            self.filenames = [self.filenames[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

def create_data_generators(train_files, train_labels, val_files, val_labels, test_files, test_labels):
    """Create data generators for training, validation, and testing"""
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    # No augmentation for validation and test
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_gen = CustomDataGenerator(
        train_files, train_labels, BATCH_SIZE, IMG_SIZE, train_datagen, shuffle=True
    )
    val_gen = CustomDataGenerator(
        val_files, val_labels, BATCH_SIZE, IMG_SIZE, val_test_datagen, shuffle=False
    )
    test_gen = CustomDataGenerator(
        test_files, test_labels, BATCH_SIZE, IMG_SIZE, val_test_datagen, shuffle=False
    )
    
    return train_gen, val_gen, test_gen, len(train_files), len(val_files), len(test_files)

# Model 1: Simple CNN
def create_simple_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Model 2: ResNet50
def create_resnet50():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = True
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Model 3: VGG16
def create_vgg16():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = True
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Model 4: EfficientNetB0
def create_efficientnet():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = True
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Model 5: MobileNetV2
def create_mobilenet():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = True
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Model 6: DenseNet121
def create_densenet():
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = True
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Model 7: InceptionV3
def create_inception():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = True
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Model 8: Xception
def create_xception():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = True
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Model 9: NASNetMobile (or alternative if not available)
def create_nasnet():
    if NASNET_AVAILABLE:
        base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    else:
        # Use EfficientNetB1 as alternative if NASNet not available
        from tensorflow.keras.applications import EfficientNetB1
        base_model = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = True
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Model 10: Custom CNN with Attention
def create_custom_cnn_attention():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Feature extraction
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    
    # Attention mechanism
    attention = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    x = layers.Multiply()([x, attention])
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

# Dictionary of all models
MODELS = {
    'Simple_CNN': create_simple_cnn,
    'ResNet50': create_resnet50,
    'VGG16': create_vgg16,
    'EfficientNetB0': create_efficientnet,
    'MobileNetV2': create_mobilenet,
    'DenseNet121': create_densenet,
    'InceptionV3': create_inception,
    'Xception': create_xception,
    'NASNetMobile': create_nasnet,
    'Custom_CNN_Attention': create_custom_cnn_attention
}

def train_model(model_name, model_func, train_gen, val_gen, train_size, val_size):
    """Train a single model"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Create model
    model = model_func()
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        ),
        callbacks.ModelCheckpoint(
            f'models/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Load best model
    model.load_weights(f'models/{model_name}_best.h5')
    
    return model, history

def evaluate_model(model, test_gen, test_size, model_name):
    """Evaluate model and generate predictions"""
    print(f"\nEvaluating {model_name}...")
    
    # Get predictions
    predictions = []
    true_labels = []
    
    # Use the generator's __getitem__ method
    for i in range(len(test_gen)):
        batch_x, batch_y = test_gen[i]
        pred = model.predict(batch_x, verbose=0)
        predictions.extend(np.argmax(pred, axis=1))
        true_labels.extend(np.argmax(batch_y, axis=1))
    
    # Limit to actual test size
    predictions = predictions[:test_size]
    true_labels = true_labels[:test_size]
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(true_labels, predictions)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': predictions,
        'true_labels': true_labels
    }

def plot_results(all_results):
    """Plot comparison results"""
    # Accuracy comparison
    model_names = [r['model_name'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results]
    
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, accuracies)
    plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1])
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('results/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrices
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for idx, result in enumerate(all_results):
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        axes[idx].set_title(f"{result['model_name']}\nAcc: {result['accuracy']:.3f}")
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results(all_results):
    """Save results to files"""
    # Create summary dataframe
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Model': result['model_name'],
            'Accuracy': result['accuracy'],
            'Precision_Avg': result['classification_report']['weighted avg']['precision'],
            'Recall_Avg': result['classification_report']['weighted avg']['recall'],
            'F1_Score_Avg': result['classification_report']['weighted avg']['f1-score']
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('Accuracy', ascending=False)
    df_summary.to_csv('results/model_comparison_summary.csv', index=False)
    
    # Save detailed reports
    with open('results/detailed_results.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("GARBAGE CLASSIFICATION - MODEL COMPARISON RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for result in all_results:
            f.write(f"\n{'='*80}\n")
            f.write(f"Model: {result['model_name']}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(result['true_labels'], result['predictions'], 
                                         target_names=CLASS_NAMES))
            f.write("\n\n")
    
    print("\nResults saved to 'results/' directory")

def main():
    """Main training and evaluation function"""
    print("="*80)
    print("GARBAGE CLASSIFICATION - TRAINING 10 DIFFERENT MODELS")
    print("="*80)
    
    # Load data
    print("\nLoading dataset...")
    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = load_data_from_splits()
    print(f"Train samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")
    
    # Create data generators
    print("\nCreating data generators...")
    train_gen, val_gen, test_gen, train_size, val_size, test_size = create_data_generators(
        train_files, train_labels, val_files, val_labels, test_files, test_labels
    )
    
    # Train and evaluate all models
    all_results = []
    
    for model_name, model_func in MODELS.items():
        # Check if model already exists
        model_path = f'models/{model_name}_final.h5'
        if os.path.exists(model_path):
            print(f"\n{'='*60}")
            print(f"⏭️  Skipping {model_name} - Model already trained!")
            print(f"   Found: {model_path}")
            print(f"{'='*60}")
            
            # Load existing model and evaluate
            try:
                print(f"Loading existing model: {model_name}")
                model = keras.models.load_model(model_path)
                
                # Recreate test generator for evaluation
                _, _, test_gen, _, _, test_size = create_data_generators(
                    train_files, train_labels, val_files, val_labels, test_files, test_labels
                )
                
                # Evaluate existing model
                result = evaluate_model(model, test_gen, test_size, model_name)
                all_results.append(result)
                
                print(f"✓ {model_name} - Accuracy: {result['accuracy']:.4f}")
            except Exception as e:
                print(f"⚠ Error loading {model_name}: {str(e)}")
                print("   Will retrain this model...")
                # Fall through to training
            else:
                continue  # Skip to next model if loading was successful
        
        # Train new model
        try:
            # Recreate generators for each model (they get exhausted after training)
            train_gen, val_gen, test_gen, train_size, val_size, test_size = create_data_generators(
                train_files, train_labels, val_files, val_labels, test_files, test_labels
            )
            
            # Train model
            model, history = train_model(model_name, model_func, train_gen, val_gen, train_size, val_size)
            
            # Recreate test generator for evaluation
            _, _, test_gen, _, _, test_size = create_data_generators(
                train_files, train_labels, val_files, val_labels, test_files, test_labels
            )
            
            # Evaluate model
            result = evaluate_model(model, test_gen, test_size, model_name)
            all_results.append(result)
            
            # Save model
            model.save(f'models/{model_name}_final.h5')
            
            print(f"\n✓ {model_name} - Accuracy: {result['accuracy']:.4f}")
            
        except Exception as e:
            print(f"\n✗ Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate plots and save results
    print("\nGenerating results...")
    plot_results(all_results)
    save_results(all_results)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    df_summary = pd.DataFrame([
        {'Model': r['model_name'], 'Accuracy': r['accuracy']} 
        for r in all_results
    ])
    df_summary = df_summary.sort_values('Accuracy', ascending=False)
    print(df_summary.to_string(index=False))
    print("\nBest Model:", df_summary.iloc[0]['Model'])
    print("Best Accuracy:", df_summary.iloc[0]['Accuracy'])
    print("\nAll results saved in 'results/' directory")
    print("All models saved in 'models/' directory")

if __name__ == "__main__":
    main()

