import numpy as np
import torch
import torchaudio
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
import librosa
import sounddevice as sd
from scipy.signal import find_peaks
import torch.nn.functional as F
from torchaudio.transforms import Resample
import queue
import threading
from Models.cnn14 import SpeechCommandNet
import collections
import time

class SpeechRecognitionCore(QObject):
    """Core class for speech recognition functionality.
    Handles audio input, processing, and word detection using a neural network model.
    Emits signals for word detection, audio processing, and state changes."""
    
    word_detected = pyqtSignal(str, float)  # word, confidence
    audio_processed = pyqtSignal(object, int, dict)  # audio_data, sample_rate, result
    recording_state_changed = pyqtSignal(bool)  # True if recording, False if not
    inference_state_changed = pyqtSignal(bool)  # True if inference is running, False if not
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 16000  # Changed to match training
        self.chunk_size = 16000   # 1 second chunks at 16kHz
        self.is_recording = False
        self.audio_buffer = []
        self.current_audio_data = None
        self.last_inference_time = 0  # Track last inference time
        
        # Command mapping for presentation
        self.command_map = {
            "up": "go",
            "right": "right",
            "left": "left",
            "down": "stop"
        }
        
        # Try to find default input device
        try:
            default_input = sd.query_devices(kind='input')
            self.device_id = default_input['index']
            print(f"Found default input device: {default_input['name']} (ID: {self.device_id})")
        except Exception as e:
            print(f"Error finding default input device: {e}")
            self.device_id = None
        
        # Real-time processing buffer (15 seconds at 16kHz)
        self.buffer_size = 15 * self.sample_rate
        self.audio_queue = collections.deque(maxlen=self.buffer_size)
        self.processing_timer = QTimer()
        self.processing_timer.timeout.connect(self.process_realtime_audio)
        self.processing_timer.start(500)  # Process every 500ms (0.5 seconds)
        
        # Word detection parameters
        self.silence_threshold = 0.001  # Reduced from 0.01 to be more sensitive
        self.min_word_duration = 0.1  # seconds
        self.max_word_duration = 1.0  # seconds
        self.silence_duration = 0.5   # seconds
        
        # Initialize model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
        
    def load_model(self):
        try:
            self.model = SpeechCommandNet(n_classes=12)
            state_dict = torch.load('Models/best_model.pth', map_location=self.device)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully from best_model.pth")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
    
    def preprocess_audio(self, audio_data):
        # Convert to tensor if numpy array
        if isinstance(audio_data, np.ndarray):
            audio_data = torch.from_numpy(audio_data).float()
        
        # Ensure correct shape [1, L]
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)
        elif audio_data.dim() == 2 and audio_data.shape[0] == 1:
            pass  # Already correct shape
        else:
            # If multi-channel, take mean
            audio_data = audio_data.mean(0, keepdim=True)
            
        # Resample to 16kHz if needed (matching training)
        if self.sample_rate != 16000:
            resampler = Resample(self.sample_rate, 16000)
            audio_data = resampler(audio_data)
            
        # Pad or trim to exactly 1 second at 16kHz
        target_length = 16000
        if audio_data.shape[1] < target_length:
            audio_data = F.pad(audio_data, (0, target_length - audio_data.shape[1]))
        else:
            audio_data = audio_data[:, :target_length]
            
        # Normalize to [-1, 1] range
        audio_data = audio_data.clamp(-1, 1)
        
        return audio_data.to(self.device)
    
    def process_audio(self, audio_data):
        if self.model is None:
            return None
        try:
            audio_tensor = self.preprocess_audio(audio_data)
            # Check if audio is too quiet (silence)
            if audio_tensor.abs().max() < self.silence_threshold:
                return None  # Removed print to reduce spam
            with torch.no_grad():
                logits = self.model(audio_tensor)
                probs = F.softmax(logits, dim=1)
                confidence, pred = torch.max(probs, dim=1)
            labels = ["yes", "no", "up", "down", "left", "right", 
                     "on", "off", "stop", "go", "silence", "unknown"]
            word = labels[pred.item()]
            conf = confidence.item()
            
            # Map the word if it's in our command map
            mapped_word = self.command_map.get(word, word)
            
            # Always print prediction info
            print(f"Detected word: {word}")
            print(f"Mapped to: {mapped_word}")
            print(f"Confidence: {conf:.4f}")
            top3_conf, top3_idx = torch.topk(probs[0], 3)
            print("\nTop 3 predictions:")
            for conf3, idx in zip(top3_conf, top3_idx):
                print(f"- {labels[idx]}: {conf3:.4f}")
            if conf > 0.2 and word not in ["silence", "unknown"]:
                print(f"Emitting word detected signal: {mapped_word} ({conf:.4f})")
                self.word_detected.emit(mapped_word, conf)
            else:
                print(f"Word not emitted - confidence too low or silence/unknown")
            return {
                'word': mapped_word,  # Use mapped word in result
                'confidence': conf,
                'probabilities': probs[0].cpu().numpy()
            }
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return None
    
    def process_realtime_audio(self):
        now = time.time()
        # Only run inference every 0.5 seconds
        if now - self.last_inference_time < 0.5:
            return
        if len(self.audio_queue) >= self.chunk_size:
            # Get the last second of audio
            audio_data = np.array(list(self.audio_queue)[-self.chunk_size:])
            
            # Signal inference start
            self.inference_state_changed.emit(True)
            
            # Process audio through model
            result = self.process_audio(audio_data)
            if result:
                self.audio_processed.emit(audio_data, self.sample_rate, result)
            self.last_inference_time = now
            
            # Signal inference end
            self.inference_state_changed.emit(False)
    
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
        if self.is_recording:
            # Convert to mono
            audio_data = indata.mean(axis=1)  # Convert to mono
            # Removed per-chunk normalization to avoid boosting silence
            # Add new audio data to the queue
            self.audio_queue.extend(audio_data)
            self.current_audio_data = audio_data.copy()
    
    def start_recording(self):
        print("\n=== STARTING RECORDING ===")
        print("Current recording state:", self.is_recording)
        if self.is_recording:
            print("ALREADY RECORDING - Ignoring start request")
            return
        try:
            # Test if we can access the microphone
            test_stream = sd.InputStream(
                device=self.device_id,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=1024
            )
            test_stream.close()
            self.is_recording = True
            self.audio_queue.clear()
            self.current_audio_data = None
            print("Initializing audio stream...")
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=1,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                blocksize=1024  # Smaller blocksize for more frequent updates
            )
            print("Starting audio stream...")
            self.stream.start()
            self.recording_state_changed.emit(True)
        except Exception as e:
            print(f"Error starting recording: {e}")
    
    def stop_recording(self):
        print("\n=== STOPPING RECORDING ===")
        print("Current recording state:", self.is_recording)
        if not self.is_recording:
            print("NOT RECORDING - Ignoring stop request")
            return
        try:
            self.is_recording = False
            if hasattr(self, 'stream') and self.stream is not None:
                print("Stopping audio stream...")
                self.stream.stop()
                self.stream.close()
                self.stream = None
            self.audio_queue.clear()
            self.current_audio_data = None
            self.recording_state_changed.emit(False)
        except Exception as e:
            print(f"Error stopping recording: {e}")
    
    def get_audio_data(self):
        if self.current_audio_data is not None:
            return self.current_audio_data
        return np.zeros((self.chunk_size, 1))

    def list_audio_devices(self):
        """List all available audio input devices."""
        devices = sd.query_devices()
        input_devices = []
        
        print("\nAvailable Audio Input Devices:")
        print("=" * 50)
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:  # If it's an input device
                input_devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'default_samplerate': device['default_samplerate']
                })
                print(f"[{i}] {device['name']}")
                print(f"    Channels: {device['max_input_channels']}")
                print(f"    Default Sample Rate: {device['default_samplerate']}")
                print("-" * 50)
        
        return input_devices

    def select_audio_device(self, device_index):
        """Select an audio input device by its index."""
        devices = sd.query_devices()
        if 0 <= device_index < len(devices):
            device = devices[device_index]
            if device['max_input_channels'] > 0:
                self.device_id = device_index
                print(f"\nSelected audio device: [{device_index}] {device['name']}")
                return True
        print(f"\nInvalid device index: {device_index}")
        return False 