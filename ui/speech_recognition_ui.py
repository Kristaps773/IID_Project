import sys
import time
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                           QPushButton, QTextEdit, QFrame, QSizePolicy,
                           QComboBox)
from PyQt5.QtCore import Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import librosa
import librosa.display
from core.speech_recognition_core import SpeechRecognitionCore

class SpeechRecognitionUI(QWidget):
    """Main UI component for speech recognition visualization and control.
    Provides real-time audio visualization, device selection, and transcription."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.colors = {
            'background': '#1e1e1e',
            'surface': '#252526',
            'primary': '#007acc',
            'secondary': '#3c3c3c',
            'text': '#ffffff',
            'text_secondary': '#cccccc',
            'success': '#4CAF50',
            'error': '#f44336'
        }
        
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self.colors['background']};
                color: {self.colors['text']};
            }}
        """)
        
        self.core = SpeechRecognitionCore()
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(20, 20, 20, 20)
        
        self.audio_buffer = np.zeros(15 * 32000)  # 15 seconds buffer at 32kHz
        self.buffer_index = 0
        
        self.create_visualization_panel()
        self.create_control_panel()
        self.create_transcript_panel()
        
        self.core.audio_processed.connect(self.update_visualization)
        self.core.word_detected.connect(self.update_transcript)
        self.core.recording_state_changed.connect(self.on_recording_state_changed)
        self.core.inference_state_changed.connect(self.on_inference_state_changed)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(50)  # 20 FPS for smooth visualization
        
        self.is_recording = False
        self.is_test_mode = False
        self.is_inference_running = False
        self.last_word_time = 0
        self.program_start_time = time.time()
        
    def create_visualization_panel(self):
        """Creates the visualization panel with waveform and spectrogram plots."""
        vis_frame = QFrame()
        vis_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {self.colors['surface']};
                border-radius: 10px;
                padding: 10px;
            }}
        """)
        vis_layout = QVBoxLayout(vis_frame)
        
        plt.style.use('dark_background')
        self.fig, (self.wave_ax, self.spec_ax) = plt.subplots(1, 2, figsize=(12, 4))
        self.fig.patch.set_facecolor('#1e1e1e')
        self.canvas = FigureCanvas(self.fig)
        vis_layout.addWidget(self.canvas)
        
        self.wave_ax.plot([], [], color='#007acc')
        
        n_freq = 1 + 1024 // 2
        n_time = int(np.ceil((15 * 32000) / 512))
        S_db_init = np.zeros((n_freq, n_time))
        
        self.spec_img = self.spec_ax.imshow(S_db_init, 
                                          aspect='auto', 
                                          origin='lower',
                                          cmap='viridis',
                                          extent=[0, 15, 0, 32000/2])
        
        self.spec_cbar = self.fig.colorbar(self.spec_img, ax=self.spec_ax)
        self.spec_cbar.set_label('Magnitude (dB)', color='white')
        
        # Configure waveform plot
        self.wave_ax.set_title('Waveform', color='white')
        self.wave_ax.set_ylim(-0.3, 0.3)
        self.wave_ax.set_xlim(0, 15)
        self.wave_ax.grid(True, color='gray', alpha=0.3)
        self.wave_ax.set_facecolor('#1e1e1e')
        self.wave_ax.set_xlabel('Time (s)', color='white')
        self.wave_ax.set_ylabel('Amplitude', color='white')
        
        # Configure spectrogram plot
        self.spec_ax.set_title('Spectrogram', color='white')
        self.spec_ax.set_xlim(0, 15)
        self.spec_ax.set_xlabel('Time (s)', color='white')
        self.spec_ax.set_ylabel('Frequency (Hz)', color='white')
        self.spec_ax.set_facecolor('#1e1e1e')
        
        self.fig.tight_layout()
        self.layout.addWidget(vis_frame)
        
    def create_control_panel(self):
        # Create container frame
        control_frame = QFrame()
        control_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {self.colors['surface']};
                border-radius: 10px;
                padding: 10px;
            }}
        """)
        control_layout = QVBoxLayout(control_frame)  # Changed to QVBoxLayout
        control_layout.setSpacing(15)
        
        # --- Top section: Device selection ---
        top_section = QHBoxLayout()
        
        # Device selection label
        device_label = QLabel("Audio Input Device:")
        device_label.setStyleSheet("font-size: 14pt;")
        top_section.addWidget(device_label)
        
        # Device selection combobox
        self.device_combo = QComboBox()
        self.device_combo.setStyleSheet("""
            QComboBox {
                background-color: #3c3c3c;
                color: white;
                padding: 5px;
                font-size: 12pt;
                border: 1px solid #555555;
                border-radius: 3px;
                min-width: 200px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #3c3c3c;
                color: white;
                selection-background-color: #007acc;
            }
        """)
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)
        top_section.addWidget(self.device_combo)
        
        # Refresh devices button
        refresh_button = QPushButton("Refresh")
        refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c;
                color: white;
                padding: 5px 10px;
                font-size: 12pt;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4c4c4c;
            }
        """)
        refresh_button.clicked.connect(self.refresh_devices)
        top_section.addWidget(refresh_button)
        
        top_section.addStretch()
        control_layout.addLayout(top_section)
        
        # --- Bottom section: Start/Stop button and indicators in one row ---
        bottom_section = QHBoxLayout()
        
        # Start/Stop button
        self.record_button = QPushButton("Start Recording")
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.record_button.clicked.connect(self.toggle_recording)
        bottom_section.addWidget(self.record_button)
        
        # Add spacing between button and indicators
        bottom_section.addSpacing(20)
        
        # Listening indicator with label
        self.listening_label = QLabel("Listening:")
        self.listening_label.setStyleSheet("font-size: 16pt;")
        bottom_section.addWidget(self.listening_label)
        self.listening_indicator = QLabel()
        self.listening_indicator.setStyleSheet("background-color: red; border-radius: 10px; min-width: 20px; min-height: 20px;")
        bottom_section.addWidget(self.listening_indicator)
        
        # Add spacing
        bottom_section.addSpacing(10)
        
        # Inference indicator with label
        self.inference_label = QLabel("Inference:")
        self.inference_label.setStyleSheet("font-size: 16pt;")
        bottom_section.addWidget(self.inference_label)
        self.inference_indicator = QLabel()
        self.inference_indicator.setStyleSheet("background-color: gray; border-radius: 10px; min-width: 20px; min-height: 20px;")
        bottom_section.addWidget(self.inference_indicator)
        
        # Add spacing
        bottom_section.addSpacing(10)
        
        # Silence indicator with label
        self.silence_label = QLabel("Silence:")
        self.silence_label.setStyleSheet("font-size: 16pt;")
        bottom_section.addWidget(self.silence_label)
        self.silence_indicator = QLabel()
        self.silence_indicator.setStyleSheet("background-color: red; border-radius: 10px; min-width: 20px; min-height: 20px;")
        bottom_section.addWidget(self.silence_indicator)
        
        control_layout.addLayout(bottom_section)
        control_layout.addStretch()
        
        self.layout.addWidget(control_frame)
        
        # Initialize device list
        self.refresh_devices()
        
    def create_transcript_panel(self):
        # Create container frame
        transcript_frame = QFrame()
        transcript_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {self.colors['surface']};
                border-radius: 10px;
                padding: 10px;
            }}
        """)
        transcript_layout = QVBoxLayout(transcript_frame)
        
        # Add title
        title = QLabel("Transcript")
        title.setStyleSheet(f"""
            QLabel {{
                color: {self.colors['text']};
                font-size: 16px;
                font-weight: bold;
                padding: 5px;
            }}
        """)
        transcript_layout.addWidget(title)
        
        # Create transcript text area
        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setStyleSheet(f"""
            QTextEdit {{
                background-color: {self.colors['secondary']};
                color: {self.colors['text']};
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }}
        """)
        transcript_layout.addWidget(self.transcript)
        
        self.layout.addWidget(transcript_frame)
        
    def toggle_recording(self):
        # Only trigger if the state is not already correct
        if not self.core.is_recording:
            self.core.start_recording()
        else:
            self.core.stop_recording()

    def on_recording_state_changed(self, is_recording):
        self.is_recording = is_recording
        if is_recording:
            self.record_button.setText("Stop Recording")
            self.record_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {self.colors['error']};
                    color: {self.colors['text']};
                    border: none;
                    padding: 12px 24px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: #d32f2f;
                }}
                QPushButton:pressed {{
                    background-color: #b71c1c;
                }}
            """)
        else:
            self.record_button.setText("Start Recording")
            self.record_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {self.colors['primary']};
                    color: {self.colors['text']};
                    border: none;
                    padding: 12px 24px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: #0086ef;
                }}
                QPushButton:pressed {{
                    background-color: #0069b3;
                }}
            """)
            
    def update_transcript(self, word, confidence):
        try:
            import time
            elapsed = time.time() - self.program_start_time
            timestamp = f"[{elapsed:06.2f}s]"
            added = False
            if confidence >= 0.23:
                self.transcript.append(f"{timestamp} {word} ({confidence:.2f})")
                self.transcript.verticalScrollBar().setValue(
                    self.transcript.verticalScrollBar().maximum()
                )
                added = True
            # If not added and top prediction is silence, show silence
            if not added and hasattr(self, 'last_result') and self.last_result is not None:
                if 'top_predictions' in self.last_result:
                    top_pred, top_conf = self.last_result['top_predictions'][0]
                    if top_pred == 'silence':
                        self.transcript.append(f"{timestamp} silence ({top_conf:.2f})")
                        self.transcript.verticalScrollBar().setValue(
                            self.transcript.verticalScrollBar().maximum()
                        )
        except Exception as e:
            print(f"Error updating transcript: {e}")
            
    def update_ui(self):
        current_time = time.time()

        if not self.is_recording:
            self.listening_indicator.setStyleSheet("background-color: gray; border-radius: 10px; min-width: 20px; min-height: 20px;")
            self.inference_indicator.setStyleSheet("background-color: gray; border-radius: 10px; min-width: 20px; min-height: 20px;")
            self.silence_indicator.setStyleSheet("background-color: gray; border-radius: 10px; min-width: 20px; min-height: 20px;")
            return

        # Update listening indicator (green when recording)
        self.listening_indicator.setStyleSheet("background-color: green; border-radius: 10px; min-width: 20px; min-height: 20px;")

        # Update inference indicator (blue when processing)
        if self.is_inference_running:
            self.inference_indicator.setStyleSheet("background-color: blue; border-radius: 10px; min-width: 20px; min-height: 20px;")
        else:
            self.inference_indicator.setStyleSheet("background-color: gray; border-radius: 10px; min-width: 20px; min-height: 20px;")

        # Update silence indicator
        if hasattr(self, 'last_result') and self.last_result is not None:
            word = self.last_result.get('word', '')
            confidence = self.last_result.get('confidence', 0)
            
            # Red when a word is predicted with high confidence
            if word not in ['silence', 'unknown'] and confidence >= 0.23:
                self.silence_indicator.setStyleSheet("background-color: red; border-radius: 10px; min-width: 20px; min-height: 20px;")
            else:
                # Green for silence or low confidence predictions
                self.silence_indicator.setStyleSheet("background-color: green; border-radius: 10px; min-width: 20px; min-height: 20px;")
        else:
            # Green when no prediction (silence)
            self.silence_indicator.setStyleSheet("background-color: green; border-radius: 10px; min-width: 20px; min-height: 20px;")

    def update_visualization(self, audio_data, sample_rate, result):
        try:
            # Store result for UI updates
            if result:
                self.last_result = result
            
            # Update audio buffer with new data
            if len(audio_data) > 0:
                # Shift buffer and add new data
                self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
                self.audio_buffer[-len(audio_data):] = audio_data

            # Only clear waveform axis
            self.wave_ax.clear()

            # Plot waveform with proper time axis (left-to-right)
            # Reverse the buffer so newest data is on the right
            display_buffer = self.audio_buffer[::-1]  # Reverse the buffer for display
            time = np.linspace(0, 15, len(display_buffer))
            self.wave_ax.plot(time, display_buffer, color='#007acc', linewidth=1.2)
            self.wave_ax.set_xlim(0, 15)  # 0 (left, oldest), 15 (right, newest)
            self.wave_ax.set_title('Waveform', color='white', fontsize=12)
            self.wave_ax.set_ylim(-0.3, 0.3)
            self.wave_ax.grid(True, color='gray', alpha=0.2, linestyle='--')
            self.wave_ax.set_facecolor('#1e1e1e')
            self.wave_ax.set_xlabel('Time (s)', color='white', fontsize=10)
            self.wave_ax.set_ylabel('Amplitude', color='white', fontsize=10)
            self.wave_ax.tick_params(colors='white', labelsize=9)

            try:
                # Calculate spectrogram
                D = librosa.stft(display_buffer, 
                               n_fft=1024,
                               hop_length=512,
                               win_length=1024,
                               window='hann')
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                
                # Update spectrogram data without recreating the image
                self.spec_img.set_array(S_db)
                self.spec_img.set_extent([0, 15, 0, sample_rate/2])
                self.spec_img.set_clim(vmin=-80, vmax=0)  # Fixed range for better visibility
                
                # Update spectrogram axes
                self.spec_ax.set_xlim(0, 15)
                self.spec_ax.set_title('Spectrogram', color='white', fontsize=12)
                self.spec_ax.set_xlabel('Time (s)', color='white', fontsize=10)
                self.spec_ax.set_ylabel('Frequency (Hz)', color='white', fontsize=10)
                self.spec_ax.set_facecolor('#1e1e1e')
                self.spec_ax.tick_params(colors='white', labelsize=9)
                
            except Exception as e:
                print(f"Error updating spectrogram: {str(e)}")
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating visualization: {str(e)}")

    def process_overlapping_prediction(self):
        # Disabled: model inference is now only handled by the core
        return

    def process_overlapping_prediction(self):
        if not self.is_recording:
            self.buffer_index = 0
            return
        buffer_len = len(self.audio_buffer)
        if self.buffer_index + 32000 > buffer_len:
            return
        while self.buffer_index + 32000 <= buffer_len:
            one_sec_audio = self.audio_buffer[self.buffer_index:self.buffer_index+32000]
            if np.abs(one_sec_audio).max() > 0.01:
                self.is_inference_running = True
                self.core.process_audio(one_sec_audio)
                self.is_inference_running = False
            self.buffer_index += 16000
        if self.buffer_index > buffer_len - 32000:
            self.buffer_index = buffer_len - 32000 

    def refresh_devices(self):
        """Refresh the list of available audio input devices."""
        self.device_combo.clear()
        devices = self.core.list_audio_devices()
        for device in devices:
            self.device_combo.addItem(f"{device['name']}", device['index'])
    
    def on_device_changed(self, index):
        """Handle device selection change."""
        if index >= 0:  # Valid selection
            device_index = self.device_combo.itemData(index)
            if self.core.select_audio_device(device_index):
                print(f"Successfully switched to audio device {device_index}")
            else:
                print(f"Failed to switch to audio device {device_index}") 

    def on_inference_state_changed(self, is_running):
        """Update the inference state and UI"""
        self.is_inference_running = is_running
        # The UI will be updated in the next update_ui call 