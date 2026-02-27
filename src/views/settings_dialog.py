import os
from dotenv import set_key, load_dotenv
from PyQt6.QtWidgets import QDialog, QFormLayout, QLineEdit, QComboBox, QPushButton, QHBoxLayout, QLabel

class SettingsDialog(QDialog):
    def __init__(self, parent=None, env_path='.env'):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(450, 250)
        self.env_path = env_path
        self.setStyleSheet("""
            QWidget { background-color: #2b2b2b; color: white; }
            QLineEdit, QComboBox { background-color: #3b3b3b; color: white; border: 1px solid #555; padding: 5px; border-radius: 3px; }
            QPushButton { background-color: #2b5c8f; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #3b7aaf; }
        """)

        layout = QFormLayout(self)
        
        # API Key
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        self.api_key_input.setText(os.getenv("OPENAI_API_KEY", ""))
        layout.addRow("OpenAI API Key:", self.api_key_input)
        
        # Camera Index
        self.camera_combo = QComboBox()
        for i in range(5):
            self.camera_combo.addItem(f"Camera {i}", i)
        
        current_cam = int(os.getenv("CAMERA_INDEX", "0"))
        idx = self.camera_combo.findData(current_cam)
        if idx != -1: self.camera_combo.setCurrentIndex(idx)
        layout.addRow("Camera Device:", self.camera_combo)
        
        # AI Mode (Cloud vs Local)
        self.ai_mode_combo = QComboBox()
        self.ai_mode_combo.addItem("Cloud Mode (OpenAI)", "cloud")
        self.ai_mode_combo.addItem("Local Privacy Mode (Ollama LLaVA)", "local")
        
        current_mode = os.getenv("AI_MODE", "cloud")
        idx = self.ai_mode_combo.findData(current_mode)
        if idx != -1: self.ai_mode_combo.setCurrentIndex(idx)
        layout.addRow("AI Engine Mode:", self.ai_mode_combo)
        
        # Save Button
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_settings)
        layout.addRow("", save_btn)

    def save_settings(self):
        new_api_key = self.api_key_input.text()
        new_cam_idx = str(self.camera_combo.currentData())
        new_mode = str(self.ai_mode_combo.currentData())
        
        set_key(self.env_path, "OPENAI_API_KEY", new_api_key)
        set_key(self.env_path, "CAMERA_INDEX", new_cam_idx)
        set_key(self.env_path, "AI_MODE", new_mode)
        load_dotenv(self.env_path, override=True)  # Reload

        self.accept()
