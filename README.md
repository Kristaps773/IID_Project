# Speech Recognition System

A PyQt5-based speech recognition system that uses PANNs CNN14 for command recognition and includes a car control simulation UI.

## Features

- Real-time speech command recognition
- Support for 10 commands: "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
- Additional support for "silence" and "unknown" classes
- Live audio processing and visualization
- Car control simulation via voice or buttons

## Requirements

- Python 3.11+
- PyQt5
- PyTorch
- torchaudio
- librosa
- sounddevice
- panns-inference
- scikit-learn
- numpy
- scipy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kristaps773/IID_Project.git
cd IID_Project
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main application:
```bash
python main.py
```

## Project Structure

- `main.py`: Main application entry point
- `load_model.py`: Model loading utility
- `core/`: Core functionality
  - `speech_recognition_core.py`: Speech recognition and audio processing logic
- `ui/`: User interface components
  - `speech_recognition_ui.py`: Main speech recognition UI
  - `car_control_ui.py`: Car control simulation UI
- `Models/`: Trained model files and label encoder
  - `cnn14.py`: Model architecture
  - `label_encoder.pkl`: Label encoder for commands
  - `best_model.pth`: Trained speech command model

## License

This project is licensed under the MIT License - see the LICENSE file for details. 