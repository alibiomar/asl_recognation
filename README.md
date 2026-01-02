# ASL Fingerspelling Recognition for NVIDIA Jetson Nano

Real-time American Sign Language (ASL) fingerspelling letter recognition system optimized for **NVIDIA Jetson Nano** edge devices. Recognizes all 26 letters using MediaPipe hand tracking, TensorFlow/TensorRT-ONNX models, and rule-based motion detection.

## 🎯 Target Platform

**NVIDIA Jetson Nano** - Edge AI device with CUDA-accelerated inference
- ARM Cortex-A57 quad-core processor
- 4GB LPDDR4 RAM
- 128-core Maxwell GPU
- USB camera support

## ✨ Features

- **All 26 ASL Letters**: Complete fingerspelling alphabet recognition
  - 24 static letters (A-I, K-Y): TensorFlow MLP neural network
  - 2 motion letters (J, Z): Rule-based trajectory detection
- **Two Recognition Modes**:
  - **Full Model (PC)** (`asl_recognize_pc.py`): TensorFlow + Piper TTS voice feedback
  - **Jetson TensorRT** (`asl_recognize_jetson.py`): TensorRT-ONNX, GPU-accelerated inference on Jetson
- **MediaPipe Hand Tracking**: 21-point hand landmark detection (42 features)
- **Real-Time Processing**: Optimized for 30 FPS on Jetson Nano
- **Word Building**: Interactive letter-by-letter word construction
- **Voice Feedback** (optional): Piper TTS for spoken letters and words
- **Data Collection Tool**: Gather training data for custom models
- **Model Training Pipeline**: Train MLP models from collected data

## 📋 Requirements

### Hardware
- **NVIDIA Jetson Nano** (4GB recommended)
- USB webcam (720p or higher recommended)
- MicroSD card (32GB+ for models and data)
- Power supply (5V 4A barrel jack recommended for full performance)

### Software
- **JetPack 4.6+** (Ubuntu 18.04 LTS with CUDA, cuDNN, TensorRT)
- Python 3.6+ (included with JetPack)
- pip package manager

### Python Packages
```
tensorflow>=2.4.0,<2.16    # TensorFlow with CUDA support
opencv-python>=4.5.0       # Computer vision
mediapipe>=0.10.0          # Hand tracking
numpy>=1.19.0,<2.0.0       # NumPy 1.x (TensorFlow compatibility)
scikit-learn>=0.24.0       # Model training utilities
h5py>=3.1.0                # Model serialization
piper-tts                  # Voice synthesis (optional)
pycuda                     # CUDA buffers for TensorRT runtime (Jetson)
# TensorRT Python bindings are installed via JetPack (nvidia-tensorrt)
```

## 🚀 Setup for Jetson Nano

### 1. Flash JetPack to MicroSD Card
- Download JetPack 4.6+ from NVIDIA Developer site
- Flash to 32GB+ microSD card using Etcher or balenaEtcher
- Boot Jetson Nano and complete initial setup

### 2. Clone Repository
```bash
git clone <repository-url>
cd edgeAI
```

### 3. Create Virtual Environment (Optional but Recommended)
```bash
python3 -m venv asl_env
source asl_env/bin/activate
```

### 4. Install Dependencies
```bash
# Update pip
pip install --upgrade pip

# Install core packages
pip install -r requirements.txt

# Install TensorFlow for Jetson (if not using JetPack's version)
# JetPack 4.6 includes TensorFlow 2.7.0+nv21.11
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
```

### 5. Verify CUDA and TensorFlow
```bash
# Check CUDA installation
nvcc --version

# Verify TensorFlow GPU support
python3 -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

### 6. Collect Training Data
```bash
# Collect 100 samples per static letter (24 letters)
python asl_data_collector_pc.py --samples 100
```

### 7. Train Model
```bash
# Train MLP model for static letters
python asl_trainer.py
```

### 8. Convert to ONNX (For TensorRT on Jetson)
```bash
# Convert H5 model to ONNX for TensorRT
python convert_h5_to_onnx.py
```

### 9. (Optional) Download Piper TTS Voice Model
```bash
# For voice feedback (requires additional storage)
mkdir -p voices
# Download from: https://github.com/rhasspy/piper/releases
# Place en_US-amy-medium.onnx and .json in voices/
```

## 📱 Usage

### Real-Time Recognition (Full Model with Voice)
```bash
# With voice feedback
python asl_recognize_pc.py --voice voices/en_US-amy-medium.onnx

# Without voice
python asl_recognize_pc.py
```

### Jetson-Optimized Recognition (TensorRT-ONNX, GPU)
```bash
# Highest performance on Jetson using TensorRT-ONNX
python asl_recognize_jetson.py
```

### Controls
- **SPACE**: Add current letter to word
- **BACKSPACE**: Delete last letter
- **C**: Clear word
- **Q**: Quit application

### Performance on Jetson Nano
- **TensorRT-ONNX version** (`asl_recognize_jetson.py`): GPU-accelerated, typically highest FPS on Jetson
- **Full TensorFlow (PC)** (`asl_recognize_pc.py`): ~15–25 FPS (depending on resolution and lighting)

## 📁 Project Structure

```
edgeAI/
├── asl_recognize_pc.py         # Full TensorFlow recognition (with TTS, PC)
├── asl_recognize_jetson.py     # TensorRT-ONNX Jetson-optimized recognition
├── asl_data_collector_pc.py    # Data collection tool (24 static letters, PC)
├── asl_trainer.py              # Model training pipeline (MLP)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── asl_data/                   # Models and datasets
│   ├── asl_static_dataset.csv  # Collected training data
│   ├── asl_static_model.h5     # Trained Keras model
│   ├── asl_static_model.onnx   # ONNX model for TensorRT (Jetson)
│   └── asl_static_metadata.pkl # Model metadata (normalization params, label encoder)
├── voices/                     # Optional TTS voice models
│   ├── en_US-amy-medium.onnx
│   └── en_US-amy-medium.onnx.json
└── old/                        # Legacy/backup files
```

## 🧠 Architecture

### Static Letter Recognition (A-I, K-Y)
- **Input**: 42 features (21 hand landmarks × 2 coordinates)
- **Preprocessing**: Normalized to wrist, scaled by hand size
- **Model**: MLP Neural Network
  - Dense(256) → BatchNorm → Dropout(0.4)
  - Dense(128) → BatchNorm → Dropout(0.3)
  - Dense(64) → Dropout(0.2)
  - Softmax(24) output
- **Training**: 70/15/15 train/val/test split, early stopping
- **Accuracy**: ~95%+ on test set

### Motion Letter Detection (J, Z)
- **Method**: Rule-based trajectory analysis
- **Features**: finger_type, path_length, straightness, direction_changes
- **J Detection**: Pinky finger + vertical downward motion
- **Z Detection**: Index finger + zigzag pattern (2+ direction changes)
- **Confidence**: Based on trajectory characteristics

## 🔧 Model Conversion Details

The `convert_h5_to_onnx.py` script:
- Converts the same Keras `.h5` model to an ONNX MLP
- Preserves the 42-feature input and class ordering
- Produces `asl_static_model.onnx` used by `asl_recognize_jetson.py`
- Enables TensorRT engine building for maximum performance on Jetson

## 📊 Data Collection Tips

1. **Lighting**: Use consistent, bright lighting
2. **Background**: Plain background for better hand detection
3. **Hand Position**: Keep hand centered, 30-60cm from camera
4. **Variation**: Vary hand rotation, position, distance
5. **Samples**: Collect 100+ samples per letter for best results

## 🎓 Training Your Own Model

1. **Collect Data**: Use `asl_data_collector_pc.py` to gather samples
2. **Train**: Run `asl_trainer.py` (creates `.h5` and `.pkl` files)
3. **Convert (Jetson)**: Use `convert_h5_to_onnx.py` for TensorRT deployment
4. **Test**: Verify with `asl_recognize_pc.py` on PC or `asl_recognize_jetson.py` on Jetson

## ⚡ Performance Optimization for Jetson Nano

### Enable Maximum Performance Mode
```bash
# Set Jetson to max performance (10W mode)
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Monitor Performance
```bash
# Real-time monitoring
sudo tegrastats

# Check temperature
cat /sys/devices/virtual/thermal/thermal_zone*/temp
```

### Reduce Memory Usage
- Prefer the TensorRT-ONNX path (`asl_recognize_jetson.py`) for best performance/efficiency
- Close unnecessary applications
- Disable GUI for headless operation

## 🐛 Troubleshooting

### MediaPipe Installation Issues
```bash
# If MediaPipe fails to install, use wheel:
wget https://github.com/PINTO0309/mediapipe-bin/releases/download/v0.10.9/mediapipe-0.10.9-cp38-cp38-linux_aarch64.whl
pip install mediapipe-0.10.9-cp38-cp38-linux_aarch64.whl
```

### Camera Not Detected
```bash
# List available cameras
v4l2-ctl --list-devices

# Test camera
gst-launch-1.0 nvarguscamerasrc ! nvoverlaysink
```

### NumPy Compatibility Issues
```bash
# Ensure NumPy 1.x for TensorFlow compatibility
pip install "numpy<2.0.0"
```

### Low FPS
- Use TensorRT-ONNX (`asl_recognize_jetson.py`) instead of pure TensorFlow
- Enable max performance mode (`sudo nvpmodel -m 0`)
- Reduce camera resolution
- Close background applications

## 🔗 Resources

- **MediaPipe**: https://mediapipe.dev/
- **TensorFlow for Jetson**: https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/
- **TensorRT**: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html
- **Piper TTS**: https://github.com/rhasspy/piper
- **Jetson Nano Guide**: https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Credits

- **MediaPipe** by Google for hand tracking
- **TensorFlow** by Google for deep learning
- **Piper TTS** by Rhasspy for voice synthesis
- **NVIDIA** for Jetson platform and CUDA acceleration
- OpenCV for computer vision utilities

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional gesture recognition (words, phrases)
- Support for more languages
- Mobile deployment (Android, iOS)
- Real-time captioning mode
- Multi-hand support

---

**Optimized for NVIDIA Jetson Nano** 🚀 | **Edge AI** 🤖 | **Real-Time ASL Recognition** ✋

## License
This project is for educational and research purposes. See individual package licenses for details.
