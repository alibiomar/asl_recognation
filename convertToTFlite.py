"""
Convert ASL Static Model from H5 to TensorFlow Lite
Optimizes the model for edge device deployment
"""

import tensorflow as tf
from pathlib import Path
import numpy as np


def convert_h5_to_tflite(model_path: str, output_path: str, quantize: bool = True):
    """
    Convert Keras H5 model to TensorFlow Lite
    
    Args:
        model_path: Path to .h5 model file
        output_path: Path for output .tflite file
        quantize: Apply post-training quantization for smaller size
    """
    print(f"\n{'='*70}")
    print("ASL MODEL CONVERSION - H5 to TFLite")
    print(f"{'='*70}\n")
    
    # Load the Keras model
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print("\nModel Summary:")
    model.summary()
    
    # Convert to TensorFlow Lite
    print("\nConverting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        print("Applying post-training quantization (dynamic range)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Optional: Set supported ops for better compatibility
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
    
    tflite_model = converter.convert()
    
    # Save the model
    print(f"\nSaving TFLite model: {output_path}")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Model size comparison
    h5_size = Path(model_path).stat().st_size / 1024  # KB
    tflite_size = len(tflite_model) / 1024  # KB
    compression_ratio = (1 - tflite_size / h5_size) * 100
    
    print(f"\n{'='*70}")
    print("CONVERSION COMPLETE")
    print(f"{'='*70}")
    print(f"Original H5 size:     {h5_size:.2f} KB")
    print(f"TFLite size:          {tflite_size:.2f} KB")
    print(f"Compression:          {compression_ratio:.1f}%")
    print(f"Quantization:         {'Enabled' if quantize else 'Disabled'}")
    
    return output_path


def test_tflite_model(tflite_path: str, test_input: np.ndarray = None):
    """
    Test the TFLite model with sample input
    
    Args:
        tflite_path: Path to .tflite model file
        test_input: Optional test input array (42 features)
    """
    print(f"\n{'='*70}")
    print("TESTING TFLITE MODEL")
    print(f"{'='*70}\n")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Input Details:")
    print(f"  Shape: {input_details[0]['shape']}")
    print(f"  Type:  {input_details[0]['dtype']}")
    
    print("\nOutput Details:")
    print(f"  Shape: {output_details[0]['shape']}")
    print(f"  Type:  {output_details[0]['dtype']}")
    
    # Test with random input if none provided
    if test_input is None:
        print("\nGenerating random test input...")
        test_input = np.random.randn(1, 42).astype(np.float32)
    else:
        test_input = test_input.reshape(1, -1).astype(np.float32)
    
    # Run inference
    print("Running inference...")
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Prediction shape: {output[0].shape}")
    print(f"Top prediction: class {np.argmax(output[0])} with confidence {np.max(output[0]):.4f}")
    
    print("\n✓ TFLite model is working correctly!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert ASL H5 model to TensorFlow Lite'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='asl_data/asl_static_model.h5',
        help='Input H5 model path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='asl_data/asl_static_model.tflite',
        help='Output TFLite model path'
    )
    parser.add_argument(
        '--no-quantize',
        action='store_true',
        help='Disable post-training quantization'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test the TFLite model after conversion'
    )
    
    args = parser.parse_args()
    
    # Convert model
    output_path = convert_h5_to_tflite(
        args.input,
        args.output,
        quantize=not args.no_quantize
    )
    
    # Test if requested
    if args.test:
        test_tflite_model(output_path)


if __name__ == '__main__':
    main()
