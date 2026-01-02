"""
ASL Fingerspelling Letter Recognition - ENHANCED TRAINER
- Uses TensorFlow/Keras (MLP) for static letters
- Motion letters (J, Z) use rule-based detection (no training needed)
- Includes comprehensive validation, visualization, and model analysis
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score
)
import time
import json
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class ASLMediaPipeTrainer:
    """Enhanced trainer for ASL recognition with comprehensive validation"""
    
    def __init__(self, data_dir: str = "asl_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.static_model = None
        self.label_encoder = None
        self.normalization_params = None
        self.training_history = None
        self.test_results = None
    
    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """Validate dataset quality and balance"""
        print("\n" + "="*70)
        print("DATASET VALIDATION")
        print("="*70)
        
        # Check for missing values
        if df.isnull().any().any():
            print("❌ Dataset contains missing values")
            null_cols = df.columns[df.isnull().any()].tolist()
            print(f"   Columns with nulls: {null_cols}")
            return False
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if np.isinf(df[numeric_cols].values).any():
            print("❌ Dataset contains infinite values")
            return False
        
        # Check label column
        if 'label' not in df.columns:
            print("❌ 'label' column not found")
            return False
        
        # Class distribution
        label_counts = df['label'].value_counts().sort_index()
        print(f"\n✓ Dataset is valid")
        print(f"  Total samples: {len(df)}")
        print(f"  Number of classes: {len(label_counts)}")
        print(f"  Features: {len(df.columns) - 1}")
        
        print("\n  Class distribution:")
        min_samples = label_counts.min()
        max_samples = label_counts.max()
        
        for letter, count in label_counts.items():
            bar = "█" * int(count / max_samples * 30)
            print(f"    {letter}: {count:4d} {bar}")
        
        # Check for class imbalance
        imbalance_ratio = max_samples / min_samples
        if imbalance_ratio > 2.0:
            print(f"\n⚠ Warning: Class imbalance detected (ratio: {imbalance_ratio:.2f})")
            print(f"   Most common: {label_counts.idxmax()} ({max_samples} samples)")
            print(f"   Least common: {label_counts.idxmin()} ({min_samples} samples)")
            print("   Consider collecting more data for underrepresented classes")
        else:
            print(f"\n✓ Classes are well balanced (ratio: {imbalance_ratio:.2f})")
        
        return True
    
    def preprocess_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[np.ndarray, ...]:
        """Split and normalize data"""
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)
        
        print(f"\n✓ Encoded {n_classes} classes: {list(self.label_encoder.classes_)}")
        
        # Split: train/val/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y_encoded
        )
        
        # Calculate validation split from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"\nDataset split:")
        print(f"  Train:      {len(X_train):5d} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {len(X_val):5d} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test:       {len(X_test):5d} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Normalize features
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        X_test_norm = (X_test - mean) / std
        
        # Save normalization parameters
        self.normalization_params = {
            'mean': mean.tolist(),
            'std': std.tolist()
        }
        
        print(f"\n✓ Features normalized (mean: {mean.mean():.4f}, std: {std.mean():.4f})")
        
        return X_train_norm, X_val_norm, X_test_norm, y_train, y_val, y_test
    
    def _create_mlp_model(self, input_dim: int, n_classes: int, architecture: str = 'default'):
        """Create MLP model with different architecture options"""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")
        
        if architecture == 'default':
            # Balanced architecture
            model = keras.Sequential([
                layers.Input(shape=(input_dim,)),
                
                layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                
                layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                
                layers.Dense(n_classes, activation='softmax')
            ], name='mlp_default')
        
        elif architecture == 'deep':
            # Deeper network for complex patterns
            model = keras.Sequential([
                layers.Input(shape=(input_dim,)),
                
                layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                
                layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                
                layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                
                layers.Dense(n_classes, activation='softmax')
            ], name='mlp_deep')
        
        elif architecture == 'light':
            # Lighter model for faster inference
            model = keras.Sequential([
                layers.Input(shape=(input_dim,)),
                
                layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                
                layers.Dense(n_classes, activation='softmax')
            ], name='mlp_light')
        
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        return model
    
    def train_static_model_tensorflow(
        self, 
        architecture: str = 'default',
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 15,
        reduce_lr_patience: int = 5
    ) -> bool:
        """Train TensorFlow MLP model for static letters"""
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            print(f"\n✓ TensorFlow {tf.__version__} detected")
        except ImportError:
            print("❌ TensorFlow not installed. Run: pip install tensorflow")
            return False
        
        filepath = self.data_dir / "asl_static_dataset.csv"
        
        if not filepath.exists():
            print(f"❌ Static dataset not found: {filepath}")
            print(f"   Expected location: {filepath.absolute()}")
            return False
        
        print("\n" + "="*70)
        print("TRAINING STATIC LETTERS MODEL")
        print("="*70)
        print(f"Architecture: {architecture.upper()}")
        print(f"Model type: Multi-Layer Perceptron (MLP)")
        
        # Load data
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return False
        
        # Validate dataset
        if not self.validate_dataset(df):
            return False
        
        # Prepare features and labels
        X = df.drop('label', axis=1).values.astype(np.float32)
        y = df['label'].values
        
        # Preprocess
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_data(X, y)
        n_classes = len(self.label_encoder.classes_)
        
        # Create model
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE")
        print("="*70)
        model = self._create_mlp_model(X.shape[1], n_classes, architecture)
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        
        # Calculate total parameters
        total_params = model.count_params()
        print(f"\nTotal parameters: {total_params:,}")
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        )
        
        # Custom callback for progress
        class ProgressCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 10 == 0:
                    print(f"\nEpoch {epoch + 1}/{epochs}")
                    print(f"  Loss: {logs['loss']:.4f} | Acc: {logs['accuracy']:.4f}")
                    print(f"  Val Loss: {logs['val_loss']:.4f} | Val Acc: {logs['val_accuracy']:.4f}")
        
        # Train
        print("\n" + "="*70)
        print("TRAINING")
        print("="*70)
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr, ProgressCallback()],
            verbose=0
        )
        
        train_time = time.time() - start_time
        self.training_history = history.history
        
        print(f"\n✓ Training completed in {train_time:.2f}s")
        print(f"  Epochs trained: {len(history.history['loss'])}")
        
        # Evaluate
        print("\n" + "="*70)
        print("EVALUATION")
        print("="*70)
        
        # Validation set
        y_val_pred = model.predict(X_val, verbose=0)
        y_val_pred_classes = np.argmax(y_val_pred, axis=1)
        val_acc = accuracy_score(y_val, y_val_pred_classes)
        val_f1 = f1_score(y_val, y_val_pred_classes, average='weighted')
        
        # Test set (final evaluation)
        y_test_pred = model.predict(X_test, verbose=0)
        y_test_pred_classes = np.argmax(y_test_pred, axis=1)
        test_acc = accuracy_score(y_test, y_test_pred_classes)
        test_f1 = f1_score(y_test, y_test_pred_classes, average='weighted')
        
        print(f"\nValidation Performance:")
        print(f"  Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  F1-Score: {val_f1:.4f}")
        
        print(f"\nTest Performance:")
        print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"  F1-Score: {test_f1:.4f}")
        
        # Store results
        self.test_results = {
            'accuracy': float(test_acc),
            'f1_score': float(test_f1),
            'predictions': y_test_pred_classes.tolist(),
            'true_labels': y_test.tolist()
        }
        
        # Classification report
        print("\n" + "="*70)
        print("PER-LETTER PERFORMANCE (Test Set)")
        print("="*70)
        report = classification_report(
            y_test, 
            y_test_pred_classes,
            target_names=self.label_encoder.classes_,
            digits=4
        )
        print(report)
        
        # Identify problematic letters
        report_dict = classification_report(
            y_test, 
            y_test_pred_classes,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        weak_letters = []
        for letter in self.label_encoder.classes_:
            if letter in report_dict and report_dict[letter]['f1-score'] < 0.85:
                weak_letters.append((letter, report_dict[letter]['f1-score']))
        
        if weak_letters:
            print("\n⚠ Letters needing improvement (F1 < 0.85):")
            for letter, f1 in sorted(weak_letters, key=lambda x: x[1]):
                print(f"  {letter}: {f1:.4f}")
            print("\n  Tip: Collect more samples for these letters")
        else:
            print("\n✓ All letters performing well (F1 ≥ 0.85)")
        
        # Confusion matrix analysis
        self._analyze_confusion_matrix(y_test, y_test_pred_classes)
        
        # Save model
        self._save_model(model, train_time, test_acc, architecture)
        
        # Generate visualizations
        self._plot_training_history()
        self._plot_confusion_matrix(y_test, y_test_pred_classes)
        
        self.static_model = model
        return True
    
    def _analyze_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Analyze confusion matrix to find common mistakes"""
        cm = confusion_matrix(y_true, y_pred)
        
        print("\n" + "="*70)
        print("CONFUSION ANALYSIS")
        print("="*70)
        
        # Find most confused pairs
        confused_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i][j] > 0:
                    confused_pairs.append((
                        self.label_encoder.classes_[i],
                        self.label_encoder.classes_[j],
                        cm[i][j]
                    ))
        
        if confused_pairs:
            confused_pairs.sort(key=lambda x: x[2], reverse=True)
            print("\nMost common confusions:")
            for true_label, pred_label, count in confused_pairs[:10]:
                print(f"  {true_label} → {pred_label}: {count} times")
        else:
            print("\n✓ No significant confusions detected")
    
    def _save_model(self, model, train_time: float, test_acc: float, architecture: str):
        """Save model and metadata"""
        print("\n" + "="*70)
        print("SAVING MODEL")
        print("="*70)
        
        # Save TensorFlow model
        model_path = self.data_dir / "asl_static_model.h5"
        model.save(model_path)
        print(f"✓ Model saved: {model_path}")
        
        # Save metadata
        metadata = {
            'label_encoder': self.label_encoder,
            'normalization_params': self.normalization_params,
            'architecture': architecture,
            'training_time': train_time,
            'test_accuracy': test_acc,
            'n_classes': len(self.label_encoder.classes_),
            'classes': self.label_encoder.classes_.tolist(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = self.data_dir / "asl_static_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ Metadata saved: {metadata_path}")
        
        # Save training history as JSON
        if self.training_history:
            history_path = self.data_dir / "training_history.json"
            # Convert numpy types to native Python types
            history_json = {
                k: [float(v) for v in vals] 
                for k, vals in self.training_history.items()
            }
            with open(history_path, 'w') as f:
                json.dump(history_json, f, indent=2)
            print(f"✓ Training history saved: {history_path}")
        
        # Save test results
        if self.test_results:
            results_path = self.data_dir / "test_results.json"
            with open(results_path, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            print(f"✓ Test results saved: {results_path}")
        
        print(f"\n✓ All files saved to: {self.data_dir.absolute()}")
    
    def _plot_training_history(self):
        """Plot training and validation metrics"""
        if not self.training_history:
            return
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Accuracy plot
            axes[0].plot(self.training_history['accuracy'], label='Training', linewidth=2)
            axes[0].plot(self.training_history['val_accuracy'], label='Validation', linewidth=2)
            axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Loss plot
            axes[1].plot(self.training_history['loss'], label='Training', linewidth=2)
            axes[1].plot(self.training_history['val_loss'], label='Validation', linewidth=2)
            axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.data_dir / "training_history.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"✓ Training plots saved: {plot_path}")
            plt.close()
            
        except Exception as e:
            print(f"⚠ Could not generate training plots: {e}")
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Plot confusion matrix heatmap"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=self.label_encoder.classes_,
                yticklabels=self.label_encoder.classes_,
                cbar_kws={'label': 'Count'}
            )
            plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.tight_layout()
            
            plot_path = self.data_dir / "confusion_matrix.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"✓ Confusion matrix saved: {plot_path}")
            plt.close()
            
        except Exception as e:
            print(f"⚠ Could not generate confusion matrix: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced ASL Letter Recognition Trainer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trainer.py                          # Train with default settings
  python trainer.py --architecture deep      # Use deeper network
  python trainer.py --epochs 150             # Train for more epochs
  python trainer.py --batch-size 64          # Larger batch size
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='asl_data',
        help='Directory containing datasets (default: asl_data)'
    )
    parser.add_argument(
        '--architecture',
        type=str,
        choices=['default', 'deep', 'light'],
        default='default',
        help='Model architecture (default: default)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size (default: 32)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Initial learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--early-stopping',
        type=int,
        default=15,
        help='Early stopping patience in epochs (default: 15)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.epochs < 10:
        print("⚠ Warning: Very few epochs may result in underfitting")
    
    if args.batch_size > 128:
        print("⚠ Warning: Large batch size may slow convergence")
    
    # Create trainer
    print("\n" + "="*70)
    print("ASL RECOGNITION MODEL TRAINER")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Max epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Early stopping: {args.early_stopping} epochs")
    
    trainer = ASLMediaPipeTrainer(args.data_dir)
    
    # Train model
    success = trainer.train_static_model_tensorflow(
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping
    )
    
    if success:
        print("\n" + "="*70)
        print("✓ TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nNext steps:")
        print("  1. Review training plots and confusion matrix")
        print("  2. Test the model with real-time recognition")
        print("  3. Collect more data for weak letters if needed")
    else:
        print("\n" + "="*70)
        print("❌ TRAINING FAILED")
        print("="*70)
        print("\nTroubleshooting:")
        print("  • Check that dataset file exists")
        print("  • Verify TensorFlow is installed: pip install tensorflow")
        print("  • Ensure dataset has sufficient samples (50+ per letter)")


if __name__ == '__main__':
    main()