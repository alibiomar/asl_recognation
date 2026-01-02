"""
Convert ASL Static Model from H5 to ONNX
Prepares model for TensorRT on Jetson Nano
"""

import tensorflow as tf
import tf2onnx
from pathlib import Path


def convert_h5_to_onnx(model_path: str, onnx_path: str):
    print(f"\n{'='*70}")
    print("ASL MODEL CONVERSION - H5 to ONNX")
    print(f"{'='*70}\n")

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print("\nModel Summary:")
    model.summary()

    print("\nConverting to ONNX...")
    input_signature = (
        tf.TensorSpec((None, 42), tf.float32, name="input"),
    )

    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,
        output_path=onnx_path
    )

    onnx_size = Path(onnx_path).stat().st_size / 1024

    print(f"\n{'='*70}")
    print("CONVERSION COMPLETE")
    print(f"{'='*70}")
    print(f"ONNX model saved: {onnx_path}")
    print(f"ONNX size: {onnx_size:.2f} KB")

    return onnx_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert ASL H5 model to ONNX"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="asl_data/asl_static_model.h5",
        help="Input H5 model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="asl_data/asl_static_model.onnx",
        help="Output ONNX model"
    )

    args = parser.parse_args()
    convert_h5_to_onnx(args.input, args.output)


if __name__ == "__main__":
    main()
