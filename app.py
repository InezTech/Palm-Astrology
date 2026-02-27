import sys
import os
import cv2
import threading
import base64
import time
import io

from dotenv import load_dotenv, set_key

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QScrollArea, QFrame, QSizePolicy,
                             QDialog, QFormLayout, QComboBox, QMessageBox, QFileDialog)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSize, QThread
from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QMovie

import mediapipe as mp
from openai import OpenAI
from PIL import Image

# For PDF export
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

SYSTEM_PROMPT = """
You are a highly professional Executive Coach and Palmistry Assessor. 
CRITICAL DIRECTIVES:
1. You CAN view and analyze images. NEVER say you are "unable to view images", "not clear enough" or that you are an AI.
2. When provided an image, ALWAYS examine the hand and the lines visible. 
3. If the image is blurry, low-lighting, or hard to read, DO NOT complain. MAKE FIRM, PROFESSIONAL ASSUMPTIONS based on standard holistic heuristics and visual estimation.
4. Deliver a clinical, structured, and insightful initial reading based strictly on those visual elements or your professional assumptions.
5. Answer follow-up questions directly. Your tone must be serious, analytical, and tailored for professional development, wealth management, and relationship coaching.
6. Provide actionable, concise, and incredibly authoritative replies (2-3 sentences max per message).
"""

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(400, 200)
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
        index = self.camera_combo.findData(current_cam)
        if index != -1:
            self.camera_combo.setCurrentIndex(index)
        layout.addRow("Camera Device:", self.camera_combo)
        
        # Save Button
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_settings)
        layout.addRow("", save_btn)

    def save_settings(self):
        new_api_key = self.api_key_input.text()
        new_cam_idx = str(self.camera_combo.currentData())
        
        set_key(dotenv_path, "OPENAI_API_KEY", new_api_key)
        set_key(dotenv_path, "CAMERA_INDEX", new_cam_idx)
        load_dotenv(dotenv_path, override=True)  # Reload

        self.accept()


class Worker(QThread):
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, messages, api_key):
        super().__init__()
        self.messages = messages
        self.api_key = api_key

    def run(self):
        try:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=self.messages
            )
            reply = response.choices[0].message.content
            self.response_ready.emit(reply)
        except Exception as e:
            self.error_occurred.emit(str(e))


class PalmOracleApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Professional Palmistry Consultant")
        self.setFixedSize(1100, 700)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a1a; }
            QLabel { color: #f0f0f0; }
            QPushButton { 
                background-color: #2b5c8f; color: white; border: none; 
                padding: 8px 16px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #3b7aaf; }
            QLineEdit {
                background-color: #2b2b2b; color: #ffffff; border: 1px solid #444; 
                padding: 10px; border-radius: 4px;
            }
            QScrollArea { border: none; background-color: transparent; }
        """)

        # State
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.captured_image_base64 = None
        self.chat_history_for_export = [] # List of tuples: (sender, text)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.camera_active = True
        self.scan_start_time = None
        
        self.init_camera()

        # Layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # -- Left Panel --
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        
        # Top Left Buttons (Settings & Export)
        topLeftBtnLayout = QHBoxLayout()
        self.settings_btn = QPushButton("⚙ Settings")
        self.settings_btn.clicked.connect(self.open_settings)
        self.settings_btn.setFixedSize(120, 35)
        
        self.export_btn = QPushButton("💾 Export Reading")
        self.export_btn.clicked.connect(self.export_to_pdf)
        self.export_btn.setFixedSize(140, 35)
        self.export_btn.hide() # Hidden until an image is captured
        
        topLeftBtnLayout.addWidget(self.settings_btn)
        topLeftBtnLayout.addWidget(self.export_btn)
        topLeftBtnLayout.addStretch()
        left_layout.addLayout(topLeftBtnLayout)
        
        self.video_label = QLabel("Initializing Camera...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(500, 500)
        self.video_label.setStyleSheet("background-color: #000; border-radius: 8px;")
        
        self.status_label = QLabel("Align your palm with the camera.")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        
        self.rescan_btn = QPushButton("Rescan Palm")
        self.rescan_btn.clicked.connect(self.rescan)
        self.rescan_btn.setStyleSheet("""
            QPushButton { background-color: #d9534f; color: white; border: none; padding: 10px 20px; font-weight: bold; border-radius: 4px; }
            QPushButton:hover { background-color: #c9302c; }
        """)
        self.rescan_btn.hide()
        
        left_layout.addWidget(self.video_label)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.rescan_btn)
        main_layout.addWidget(left_panel, 1)

        # -- Right Panel --
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.chat_container)

        self.loading_label = QLabel("Consultant is formulating their reading...")
        self.loading_label.setStyleSheet("color: #aaaaaa; font-style: italic;")
        self.loading_label.hide()

        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask the consultant a question...")
        self.chat_input.returnPressed.connect(self.send_message)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)

        input_layout.addWidget(self.chat_input, 1)
        input_layout.addWidget(self.send_btn)

        right_layout.addWidget(self.scroll_area, 1)
        right_layout.addWidget(self.loading_label)
        right_layout.addLayout(input_layout)
        main_layout.addWidget(right_panel, 1)

        self.add_chat_bubble("System", "Welcome to your professional palmistry consultation. Please align your palm with the camera to begin.")

        # Camera Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(30)

    def init_camera(self):
        cam_index = int(os.getenv("CAMERA_INDEX", "0"))
        self.cap = cv2.VideoCapture(cam_index)

    def open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec():
            # If camera changed, restart it
            if self.camera_active:
                self.cap.release()
                self.init_camera()

    def add_chat_bubble(self, sender, text):
        self.chat_history_for_export.append((sender, text))
        
        bubble = QFrame()
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(12, 12, 12, 12)
        
        sender_lbl = QLabel(f"{sender}:")
        sender_lbl.setStyleSheet("color: khaki; font-size: 14px; font-weight: bold;")
        
        text_lbl = QLabel(text)
        text_lbl.setWordWrap(True)
        text_lbl.setStyleSheet("color: white; font-size: 14px;")
        text_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        if sender == "System" or sender == "Consultant":
            bubble.setStyleSheet("background-color: #2c2c2c; border-radius: 10px;")
        else:
            bubble.setStyleSheet("background-color: #1f538d; border-radius: 10px;")
            sender_lbl.setStyleSheet("color: lightblue; font-size: 14px; font-weight: bold;")
            
        bubble_layout.addWidget(sender_lbl)
        bubble_layout.addWidget(text_lbl)
        self.chat_layout.addWidget(bubble)
        
        # Scroll down
        QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

    def show_loading(self, show=True):
        if show:
            self.loading_label.show()
            self.chat_input.setEnabled(False)
            self.send_btn.setEnabled(False)
        else:
            self.loading_label.hide()
            self.chat_input.setEnabled(True)
            self.send_btn.setEnabled(True)

    def rescan(self):
        self.rescan_btn.hide()
        self.export_btn.hide()
        self.camera_active = True
        self.init_camera()
        self.status_label.setText("Align your palm with the camera.")
        self.scan_start_time = None
        self.captured_image_base64 = None
        self.chat_history_for_export = []
        
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for i in reversed(range(self.chat_layout.count())): 
            widget = self.chat_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
                
        self.add_chat_bubble("System", "Welcome to your professional palmistry consultation. Please align your palm with the camera to begin.")
        self.timer.start(30)

    def update_camera(self):
        if not self.camera_active:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.video_label.setText("Camera disconnected or permission denied.\nPlease check settings.")
            return
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            h, w, c = frame.shape
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Custom Scanning Overlay (instead of stark mediapipe lines)
                x_max, y_max, x_min, y_min = 0, 0, w, h
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max: x_max = x
                    if x < x_min: x_min = x
                    if y > y_max: y_max = y
                    if y < y_min: y_min = y
                
                # Padding
                pad = 30
                x_min = max(0, x_min - pad)
                y_min = max(0, y_min - pad)
                x_max = min(w, x_max + pad)
                y_max = min(h, y_max + pad)
                
                # Draw high-end corner brackets
                length = 40
                thick = 3
                col = (200, 200, 255) # Light blueish/white in BGR
                
                # Top-Left
                cv2.line(frame, (x_min, y_min), (x_min + length, y_min), col, thick)
                cv2.line(frame, (x_min, y_min), (x_min, y_min + length), col, thick)
                # Top-Right
                cv2.line(frame, (x_max, y_min), (x_max - length, y_min), col, thick)
                cv2.line(frame, (x_max, y_min), (x_max, y_min + length), col, thick)
                # Bottom-Left
                cv2.line(frame, (x_min, y_max), (x_min + length, y_max), col, thick)
                cv2.line(frame, (x_min, y_max), (x_min, y_max - length), col, thick)
                # Bottom-Right
                cv2.line(frame, (x_max, y_max), (x_max - length, y_max), col, thick)
                cv2.line(frame, (x_max, y_max), (x_max, y_max - length), col, thick)
                
                # Subtle semi-transparent overaly inside the box
                overlay = frame.copy()
                cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 230, 200), -1)
                cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
            
            if self.scan_start_time is None:
                self.scan_start_time = time.time()
                self.status_label.setText("Hand detected. Hold steady...")
            elif time.time() - self.scan_start_time > 3.0:
                self.capture_image(rgb_frame, frame)
                return
        else:
            self.scan_start_time = None
            self.status_label.setText("Align your palm with the camera.")

        # Display
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            500, 500, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def capture_image(self, rgb_frame, frame_with_overlay):
        self.camera_active = False
        self.timer.stop()
        self.cap.release()
        self.status_label.setText("Scan Complete. Initializing professional reading...")
        self.rescan_btn.show()
        self.export_btn.show()

        # Display frozen image with overlay
        h, w, ch = frame_with_overlay.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_with_overlay.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            500, 500, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(pixmap)
        
        # Save a clean RGB frame for AI processing & Exporting
        self.clean_rgb_frame = rgb_frame.copy()

        img = Image.fromarray(rgb_frame)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        self.captured_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        self.add_chat_bubble("System", "Image captured successfully. Let me analyze the lines of your palm...")
        
        user_content = [
            {
                "type": "text", 
                "text": "Analyze this image of my palm. You MUST provide a direct, professional, clinical palmistry reading. Even if the image is blurry, do your absolute best to identify general shapes and lines based on professional heuristics. Do not ask for a better image. Do not refuse. DO NOT break character."
            }
        ]
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{self.captured_image_base64}"}
        })
        self.messages.append({"role": "user", "content": user_content})
        
        self.start_worker()

    def send_message(self):
        text = self.chat_input.text()
        if not text.strip() or self.camera_active:
            return
            
        self.chat_input.clear()
        self.add_chat_bubble("You", text)
        self.messages.append({"role": "user", "content": text})
        
        self.start_worker()

    def start_worker(self):
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            self.add_chat_bubble("System", "OpenAI API Key is missing. Please set it in Settings.")
            return

        self.show_loading(True)
        self.worker = Worker(self.messages, api_key)
        self.worker.response_ready.connect(self.on_bot_response)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()

    @pyqtSlot(str)
    def on_bot_response(self, text):
        self.show_loading(False)
        self.messages.append({"role": "assistant", "content": text})
        self.add_chat_bubble("Consultant", text)
        self.status_label.setText("Consultation active. You may ask further questions.")

    @pyqtSlot(str)
    def on_error(self, err_text):
        self.show_loading(False)
        self.add_chat_bubble("System", f"Error: {err_text}")

    def export_to_pdf(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Reading", "Palm_Reading_Consultation.pdf", "PDF Files (*.pdf)")
        if not filename:
            return
            
        try:
            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            Story = []

            # Title
            title_style = styles["Heading1"]
            title_style.alignment = 1
            Story.append(Paragraph("Professional Palmistry Consultation", title_style))
            Story.append(Spacer(1, 0.2 * inch))

            # Image
            if hasattr(self, 'clean_rgb_frame'):
                img = Image.fromarray(self.clean_rgb_frame)
                img_path = os.path.join(os.path.dirname(__file__), "temp_palm.jpg")
                img.save(img_path)
                rl_img = RLImage(img_path, width=3*inch, height=3*inch)
                Story.append(rl_img)
                Story.append(Spacer(1, 0.2 * inch))

            # Chat History
            normal_style = styles["Normal"]
            for sender, text in self.chat_history_for_export:
                Story.append(Paragraph(f"<b>{sender}:</b> {text}", normal_style))
                Story.append(Spacer(1, 0.1 * inch))

            doc.build(Story)
            
            # Clean up temp image
            if os.path.exists("temp_palm.jpg"):
                os.remove("temp_palm.jpg")

            self.add_chat_bubble("System", f"Reading exported successfully to {filename}.")
        except Exception as e:
            self.add_chat_bubble("System", f"Failed to export PDF: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PalmOracleApp()
    window.show()
    sys.exit(app.exec())
