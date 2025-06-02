import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QStyleFactory
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt
from ui.speech_recognition_ui import SpeechRecognitionUI
from ui.car_control_ui import CarControlUI

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech Recognition System")
        self.setGeometry(100, 100, 1600, 900)
        
        self.set_dark_theme()
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.recognition_ui = SpeechRecognitionUI()
        self.car_control = CarControlUI()
        
        # Share the speech recognition core with car control
        self.car_control.set_core(self.recognition_ui.core)
        
        self.tabs.addTab(self.recognition_ui, "Speech Recognition")
        self.tabs.addTab(self.car_control, "Car Control")
        
        self.recognition_ui.core.word_detected.connect(self.car_control.process_command)
        self.recognition_ui.core.recording_state_changed.connect(self.on_recording_state_changed)
        
    def set_dark_theme(self):
        app = QApplication.instance()
        app.setStyle(QStyleFactory.create('Fusion'))
        
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        
        app.setPalette(dark_palette)

    def on_recording_state_changed(self, is_recording):
        """Update recording state in all UIs"""
        self.car_control.is_recording = is_recording

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
