"""
ASL Fingerspelling Letter Recognition - MEDIAPIPE-OPTIMIZED TRAINER
- Uses TensorFlow/Keras (MLP) for static letters
- Motion letters (J, Z) use rule-based detection (no LSTM training needed)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time


class ASLMediaPipeTrainer:
    """Train models optimized for MediaPipe hand landmarks"""
    
    def __init__(self, data_dir: str = "asl_data"):
        self.data_dir = Path(data_dir)
        self.static_model = None
        # No motion model needed - using rule-based detection
    
    def train_static_model_tensorflow(self, model_type='mlp'):
        """
        Train TensorFlow model for 24 static letters (A-I, K-Y)
        Multi-Layer Perceptron (MLP) architecture
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            print("❌ TensorFlow not installed. Run: pip install tensorflow")
            return False
        
        filepath = self.data_dir / "asl_static_dataset.csv"
        
        if not filepath.exists():
            print(f"❌ Static dataset not found: {filepath}")
            return False
        
        print("\n" + "="*70)
        print(f"TRAINING STATIC LETTERS - MLP {model_type.upper()}")
        print("Multi-Layer Perceptron architecture")
        print("="*70)
        
        # Load data
        df = pd.read_csv(filepath)
        X = df.drop('label', axis=1).values.astype(np.float32)
        y = df['label'].values
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)
        
        print(f"\n✓ Loaded {len(X)} samples for {n_classes} letters")
        print(f"  Features: {X.shape[1]}")
        print(f"  Classes: {list(self.label_encoder.classes_)}")
        
        # Split data: 70% train, 15% validation, 15% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
        )
        
        print(f"\nDataset split:")
        print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Normalize features (important for neural networks)
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std
        
        # Save normalization parameters
        self.normalization_params = {'mean': mean, 'std': std}
        
        # Create model based on type
        if model_type == 'mlp':
            model = self._create_mlp_model(X.shape[1], n_classes)
        else:
            print(f"❌ Unknown model type: {model_type}")
            return False
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel Architecture:")
        model.summary()
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Train
        print("\n  Training...")
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        train_time = time.time() - start_time
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val, verbose=0)
        y_val_pred_classes = np.argmax(y_val_pred, axis=1)
        val_acc = accuracy_score(y_val, y_val_pred_classes)
        
        # Evaluate on test set (final evaluation)
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        test_acc = accuracy_score(y_test, y_pred_classes)
        
        print(f"\n  Training time: {train_time:.2f}s")
        print(f"  Validation Accuracy: {val_acc:.2%}")
        print(f"  Test Accuracy: {test_acc:.2%}")
        
        # Classification report
        print("\nPer-Letter Performance (Test Set):")
        print(classification_report(
            y_test, 
            y_pred_classes,
            target_names=self.label_encoder.classes_
        ))
        
        # Save model
        model_path = self.data_dir / "asl_static_model.h5"
        model.save(model_path)
        print(f"\n✓ TensorFlow model saved: {model_path}")
        
        # Save label encoder and normalization
        metadata = {
            'label_encoder': self.label_encoder,
            'normalization_params': self.normalization_params
        }
        metadata_path = self.data_dir / "asl_static_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ Metadata saved: {metadata_path}")
        
        self.static_model = model
        return True
    
    def _create_mlp_model(self, input_dim, n_classes):
        """
        Multi-Layer Perceptron (MLP) architecture
        Dense layers with BatchNorm and Dropout regularization
        """
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            
            # Feature extraction layers
            layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            # Classification layer
            layers.Dense(n_classes, activation='softmax')
        ], name='mlp_classifier')
        
        return model
    

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ASL Letter Recognition - MLP + LSTM Training'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='asl_data',
        help='Directory containing datasets'
    )
    parser.add_argument(
        '--static-model',
        type=str,
        choices=['mlp'],
        default='mlp',
        help='Type of model for static letters (Multi-Layer Perceptron)'
    )
    parser.add_argument(
        '--seq-length',
        type=int,
        default=30,
        help='Sequence length for LSTM motion model'
    )
    parser.add_argument(
        '--static-only',
        action='store_true',
        help='Train only static letter model'
    )
    parser.add_argument(
        '--motion-only',
        action='store_true',
        help='Train only motion letter model (LSTM)'
    )
    
    args = parser.parse_args()
    
    trainer = ASLMediaPipeTrainer(args.data_dir)
    
    if args.static_only:
        trainer.train_static_model_tensorflow(args.static_model)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ASL Letter Recognition - MLP Training (Static Letters Only)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='asl_data',
        help='Directory containing datasets'
    )
    parser.add_argument(
        '--static-model',
        type=str,
        choices=['mlp'],
        default='mlp',
        help='Type of model for static letters (Multi-Layer Perceptron)'
    )
    
    args = parser.parse_args()
    
    trainer = ASLMediaPipeTrainer(args.data_dir)
    trainer.train_static_model_tensorflow(args.static_model)


if __name__ == '__main__':
    main()