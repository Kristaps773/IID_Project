import sys
import math
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QLabel, QFrame, QGraphicsView, QGraphicsScene)
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QPainterPath, QTransform
import time

class CarControlUI(QWidget):
    """UI component for controlling a virtual car through voice commands or buttons.
    Provides visual feedback of car movement and system state."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.colors = {
            'background': '#1e1e1e',
            'surface': '#252526',
            'primary': '#007acc',
            'secondary': '#3c3c3c',
            'text': '#ffffff',
            'success': '#4CAF50',
            'error': '#f44336'
        }
        
        self.car_x = 600
        self.car_y = 300
        self.car_angle = 0
        self.car_speed = 2
        self.is_moving = False
        
        self.last_inference_flash = 0
        self.last_command = None
        self.last_confidence = 0
        
        self.is_recording = False
        self.core = None
        
        self.init_ui()
        
        self.move_timer = QTimer()
        self.move_timer.timeout.connect(self.update_car_position)
        self.move_timer.start(16)  # 60 FPS for smooth animation
        
    def init_ui(self):
        self.setStyleSheet(f"background-color: {self.colors['background']};")
        layout = QVBoxLayout(self)
        
        # Create car view
        self.scene = QGraphicsScene(self)
        self.scene.setBackgroundBrush(QColor(self.colors['background']))
        self.view = QGraphicsView(self.scene)
        self.view.setStyleSheet(f"border: none; background: {self.colors['background']};")
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setFixedSize(1200, 600)
        self.scene.setSceneRect(0, 0, 1200, 600)
        layout.addWidget(self.view)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        # Create control buttons with the same style
        button_style = f"""
            QPushButton {{
                background-color: {self.colors['secondary']};
                color: {self.colors['text']};
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                min-width: 80px;
            }}
            QPushButton:hover {{
                background-color: {self.colors['primary']};
            }}
            QPushButton:pressed {{
                background-color: #005999;
            }}
        """
        
        # Create buttons
        self.left_btn = QPushButton("Left")
        self.right_btn = QPushButton("Right")
        self.forward_btn = QPushButton("Go")
        self.stop_btn = QPushButton("Stop")
        
        # Disable keyboard focus for all control buttons
        for btn in [self.left_btn, self.right_btn, self.forward_btn, self.stop_btn]:
            btn.setFocusPolicy(Qt.NoFocus)
            btn.setStyleSheet(button_style)
            button_layout.addWidget(btn)
        
        self.left_btn.clicked.connect(lambda: self.turn_car("left"))
        self.right_btn.clicked.connect(lambda: self.turn_car("right"))
        self.forward_btn.clicked.connect(lambda: self.control_movement("go"))
        self.stop_btn.clicked.connect(lambda: self.control_movement("stop"))
        
        # Add recording button
        self.record_button = QPushButton("Start Recording")
        self.record_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors['success']};
                color: {self.colors['text']};
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: #45a049;
            }}
        """)
        self.record_button.clicked.connect(self.toggle_recording)
        button_layout.addWidget(self.record_button)
        
        layout.addLayout(button_layout)
        
        # Status and indicators layout
        status_layout = QHBoxLayout()
        
        # Status label
        self.status_label = QLabel("Car Status: Stopped")
        self.status_label.setStyleSheet(f"color: {self.colors['text']}; font-size: 14px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)
        
        # Inference indicator with label
        inference_layout = QHBoxLayout()
        inference_label = QLabel("Inference:")
        inference_label.setStyleSheet(f"color: {self.colors['text']}; font-size: 14px;")
        inference_layout.addWidget(inference_label)
        
        self.inference_indicator = QLabel()
        self.inference_indicator.setFixedSize(20, 20)
        self.inference_indicator.setStyleSheet("""
            background-color: gray;
            border-radius: 10px;
            margin: 5px;
        """)
        inference_layout.addWidget(self.inference_indicator)
        
        # Silence indicator with label
        silence_label = QLabel("Silence:")
        silence_label.setStyleSheet(f"color: {self.colors['text']}; font-size: 14px;")
        inference_layout.addWidget(silence_label)
        
        self.silence_indicator = QLabel()
        self.silence_indicator.setFixedSize(20, 20)
        self.silence_indicator.setStyleSheet("""
            background-color: gray;
            border-radius: 10px;
            margin: 5px;
        """)
        inference_layout.addWidget(self.silence_indicator)
        
        status_layout.addLayout(inference_layout)
        layout.addLayout(status_layout)
        
        # Update timer for indicators
        self.indicator_timer = QTimer()
        self.indicator_timer.timeout.connect(self.update_indicators)
        self.indicator_timer.start(50)  # Update every 50ms

    def showEvent(self, event):
        super().showEvent(event)
        self.setFocus()

    def toggle_recording(self):
        if self.core is None:
            print("Warning: Speech recognition core not connected")
            return
        if not self.is_recording:
            self.core.start_recording()
            self.is_recording = True
            self.record_button.setText("Stop Recording")
            self.record_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {self.colors['error']};
                    color: {self.colors['text']};
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 14px;
                    min-width: 120px;
                }}
                QPushButton:hover {{
                    background-color: #d32f2f;
                }}
            """)
        else:
            self.core.stop_recording()
            self.is_recording = False
            self.record_button.setText("Start Recording")
            self.record_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {self.colors['success']};
                    color: {self.colors['text']};
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 14px;
                    min-width: 120px;
                }}
                QPushButton:hover {{
                    background-color: #45a049;
                }}
            """)

    def set_core(self, core):
        self.core = core
        if core is not None:
            core.recording_state_changed.connect(self.on_recording_state_changed)

    def on_recording_state_changed(self, is_recording):
        self.is_recording = is_recording
        if is_recording:
            self.record_button.setText("Stop Recording")
            self.record_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {self.colors['error']};
                    color: {self.colors['text']};
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 14px;
                    min-width: 120px;
                }}
                QPushButton:hover {{
                    background-color: #d32f2f;
                }}
            """)
        else:
            self.record_button.setText("Start Recording")
            self.record_button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {self.colors['success']};
                    color: {self.colors['text']};
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 14px;
                    min-width: 120px;
                }}
                QPushButton:hover {{
                    background-color: #45a049;
                }}
            """)

    def draw_car(self):
        """Draws the car on the scene with current position and rotation."""
        self.scene.clear()
        
        car_path = QPainterPath()
        car_path.moveTo(0, -15)  # Front point
        car_path.lineTo(10, 15)  # Right rear
        car_path.lineTo(-10, 15)  # Left rear
        car_path.closeSubpath()
        
        transform = QTransform()
        transform.translate(self.car_x, self.car_y)
        transform.rotate(self.car_angle)
        
        car_path = transform.map(car_path)
        self.scene.addPath(car_path, 
                         QPen(QColor(self.colors['primary']), 2),
                         QBrush(QColor(self.colors['primary'])))

    def update_car_position(self):
        if self.is_moving:
            angle_rad = math.radians(self.car_angle)
            self.car_x += self.car_speed * math.cos(angle_rad)
            self.car_y += self.car_speed * math.sin(angle_rad)
            
            # Wrap around edges with wider view
            self.car_x = self.car_x % 1200
            self.car_y = self.car_y % 600
            
        self.draw_car()
        
    def turn_car(self, direction):
        if direction == "left":
            self.car_angle = (self.car_angle - 90) % 360
        elif direction == "right":
            self.car_angle = (self.car_angle + 90) % 360
        self.draw_car()
        
    def control_movement(self, command):
        if command == "go":
            self.is_moving = True
            self.status_label.setText("Car Status: Moving")
        elif command == "stop":
            self.is_moving = False
            self.status_label.setText("Car Status: Stopped")
        self.draw_car()

    def process_command(self, command, confidence):
        """Process voice commands from speech recognition"""
        # Always update these values first
        self.last_inference_flash = time.time()
        self.last_command = command.lower()
        self.last_confidence = confidence
        
        # Debug print
        print(f"Processing command: {command} (conf: {confidence:.3f})")
        print(f"Silence indicator should be: {'ON' if confidence < 0.23 or command.lower() == 'silence' else 'OFF'}")
        
        if confidence < 0.23:  # Confidence threshold
            return
            
        command = command.lower()
        if command == "go":
            self.control_movement("go")
        elif command == "stop":
            self.control_movement("stop")
        elif command == "left":
            self.turn_car("left")
        elif command == "right":
            self.turn_car("right")
            
    def update_indicators(self):
        current_time = time.time()
        
        # Update inference indicator - only show for high confidence predictions
        if (current_time - self.last_inference_flash < 0.5 and 
            self.last_confidence >= 0.23 and 
            self.last_command not in ['silence', 'unknown']):
            self.inference_indicator.setStyleSheet("""
                background-color: #4CAF50;
                border-radius: 10px;
                margin: 5px;
            """)
        else:
            self.inference_indicator.setStyleSheet("""
                background-color: gray;
                border-radius: 10px;
                margin: 5px;
            """)
            
        # Update silence indicator based on recording state and predictions
        if not self.is_recording:
            # Gray when not recording
            self.silence_indicator.setStyleSheet("""
                background-color: gray;
                border-radius: 10px;
                margin: 5px;
            """)
        else:
            # When recording, check predictions
            if current_time - self.last_inference_flash < 0.5:  # Only for recent predictions
                if (self.last_command == 'silence' or self.last_confidence < 0.23):
                    # Green for silence or no confident prediction
                    self.silence_indicator.setStyleSheet("""
                        background-color: #4CAF50;  /* Green for silence/no word */
                        border-radius: 10px;
                        margin: 5px;
                    """)
                else:
                    # Red for other word predictions
                    self.silence_indicator.setStyleSheet("""
                        background-color: #f44336;  /* Red for words */
                        border-radius: 10px;
                        margin: 5px;
                    """)
            else:
                # Green when no recent prediction (silence)
                self.silence_indicator.setStyleSheet("""
                    background-color: #4CAF50;  /* Green for no prediction */
                    border-radius: 10px;
                    margin: 5px;
                """) 