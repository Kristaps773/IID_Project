# Speech Recognition System

A PyQt5-based speech recognition system that uses PANNs CNN14 for command recognition.

## Features

- Real-time speech command recognition
- Support for 10 commands: "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
- Additional support for "silence" and "unknown" classes
- Live audio processing and visualization
- Statistics and process logs

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
- `core/`: Core functionality
  - `speech_recognition.py`: Speech recognition implementation
  - `process_logs.py`: Process logging functionality
  - `statistics.py`: Statistics tracking
- `ui/`: User interface components
  - `speech_recognition_ui.py`: Main speech recognition UI
  - `process_logs_ui.py`: Process logs UI
  - `statistics_ui.py`: Statistics UI
- `Models/`: Trained model files
  - `speech_cmd_cnn14.pth`: Trained speech command model

## License

This project is licensed under the MIT License - see the LICENSE file for details. 