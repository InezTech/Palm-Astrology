import sys
import os
import cv2
import base64
import time
import io
import numpy as np

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QScrollArea, QFrame, QFileDialog, QTabWidget)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QThread
from PyQt6.QtGui import QImage, QPixmap, QFont
from PIL import Image
import pyqtgraph.opengl as gl

# Import core components
from src.core.cv_pipeline import CVPipeline
from src.services.llm_service import LLMService, SYSTEM_PROMPT
from src.views.settings_dialog import SettingsDialog

# For PDF export
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

class AIWorker(QThread):
    response_ready = pyqtSignal(str)
    
    def __init__(self, llm_service, prompt, image_b64=None, hand_stats=None):
        super().__init__()
        self.llm = llm_service
        self.prompt = prompt
        self.image_b64 = image_b64
        self.hand_stats = hand_stats

    def run(self):
        try:
            if self.image_b64:
                reply = self.llm.analyze_palm(self.image_b64, self.prompt, self.hand_stats)
            else:
                reply = self.llm.chat([{"role": "user", "content": self.prompt}])
            self.response_ready.emit(reply)
        except Exception as e:
            self.response_ready.emit(f"Error: {e}")

class DebateWorker(QThread):
    response_ready = pyqtSignal(str, str) # sender, text
    finished_debate = pyqtSignal()
    
    def __init__(self, llm_service, image_b64, hand_stats):
        super().__init__()
        self.llm = llm_service
        self.context = {'image': image_b64, 'stats': hand_stats}

    def run(self):
        try:
            self.llm.synthesize_debate(self.context, self.on_update)
            self.finished_debate.emit()
        except Exception as e:
            self.response_ready.emit("System", f"Debate Error: {e}")
            self.finished_debate.emit()

    def on_update(self, sender, text):
        self.response_ready.emit(sender, text)

class PalmOracleApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Professional Palmistry Consultant - Enterprise Edition")
        self.setFixedSize(1300, 800)
        
        self.env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
        
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

        # Components
        self.cv_pipeline = CVPipeline()
        self.llm_service = LLMService()
        
        # State
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.captured_image_base64 = None
        self.current_hand_stats = None
        self.chat_history_for_export = []
        
        self.camera_active = True
        self.scan_start_time = None
        self.cap = None
        
        self.init_ui()
        self.init_camera()

        # Camera Timer Loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(30)

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # -- Left Panel (Vision) --
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        
        # Buttons
        topLeftBtnLayout = QHBoxLayout()
        self.settings_btn = QPushButton("⚙ Settings")
        self.settings_btn.clicked.connect(self.open_settings)
        
        self.export_btn = QPushButton("💾 Export Reading")
        self.export_btn.clicked.connect(self.export_to_pdf)
        self.export_btn.hide()
        
        topLeftBtnLayout.addWidget(self.settings_btn)
        topLeftBtnLayout.addWidget(self.export_btn)
        topLeftBtnLayout.addStretch()
        left_layout.addLayout(topLeftBtnLayout)
        
        # Display Area: Camera + 3D GL
        display_layout = QHBoxLayout()
        
        self.video_label = QLabel("Initializing Camera...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(400, 400)
        self.video_label.setStyleSheet("background-color: #000; border-radius: 8px;")
        
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setFixedSize(400, 400)
        self.gl_widget.opts['distance'] = 2.0
        self.gl_widget.opts['fov'] = 60
        self.gl_widget.opts['elevation'] = -90
        self.gl_widget.opts['azimuth'] = 90
        
        self.scatter = gl.GLScatterPlotItem()
        self.gl_widget.addItem(self.scatter)
        
        # Add wireframe segments (21 nodes -> lines)
        self.connections = self.cv_pipeline.mp_hands.HAND_CONNECTIONS
        
        
        display_layout.addWidget(self.video_label)
        display_layout.addWidget(self.gl_widget)
        
        left_layout.addLayout(display_layout)
        
        self.status_label = QLabel("Align your palm with the camera.")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        
        self.stats_label = QLabel("3D Mesh topology mapping running...")
        self.stats_label.setStyleSheet("color: #00ffff; font-family: monospace;")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.rescan_btn = QPushButton("Rescan Palm")
        self.rescan_btn.clicked.connect(self.rescan)
        self.rescan_btn.hide()
        
        left_layout.addWidget(self.stats_label)
        left_layout.addWidget(self.status_label)
        left_layout.addWidget(self.rescan_btn)
        main_layout.addWidget(left_panel, 1)

        # -- Right Panel (Chat/Debate) --
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.chat_container)

        self.loading_label = QLabel("Initializing Multi-Agent System...")
        self.loading_label.setStyleSheet("color: #aaaaaa; font-style: italic;")
        self.loading_label.hide()

        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask the Synthesizer a question...")
        self.chat_input.returnPressed.connect(self.send_message)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)

        input_layout.addWidget(self.chat_input, 1)
        input_layout.addWidget(self.send_btn)

        right_layout.addWidget(self.scroll_area, 1)
        right_layout.addWidget(self.loading_label)
        right_layout.addLayout(input_layout)
        main_layout.addWidget(right_panel, 1)

        self.add_chat_bubble("System", "Welcome to PalmOracle Enterprise. Featuring Multi-Agent Debating and live 3D Topology scanning. Align hand to capture.")

    def init_camera(self):
        cam_index = int(os.getenv("CAMERA_INDEX", "0"))
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(cam_index)

    def open_settings(self):
        dlg = SettingsDialog(self, env_path=self.env_path)
        if dlg.exec():
            self.llm_service = LLMService() # Reload settings
            if self.camera_active:
                self.init_camera()

    def add_chat_bubble(self, sender, text):
        self.chat_history_for_export.append((sender, text))
        bubble = QFrame()
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(12, 12, 12, 12)
        
        sender_lbl = QLabel(f"[{sender}]")
        sender_lbl.setStyleSheet("color: khaki; font-size: 14px; font-weight: bold;")
        text_lbl = QLabel(text)
        text_lbl.setWordWrap(True)
        text_lbl.setStyleSheet("color: white; font-size: 14px;")
        
        if sender == "System": bubble.setStyleSheet("background-color: #222222; border-radius: 10px;")
        elif sender == "The Traditionalist": bubble.setStyleSheet("background-color: #3b2c1f; border-left: 5px solid orange; border-radius: 10px;")
        elif sender == "The Psychologist": bubble.setStyleSheet("background-color: #1f3b3b; border-left: 5px solid teal; border-radius: 10px;")
        elif sender == "The Skeptic": bubble.setStyleSheet("background-color: #3b1f1f; border-left: 5px solid red; border-radius: 10px;")
        elif sender == "Synthesizer": bubble.setStyleSheet("background-color: #1f538d; border-left: 5px solid lightblue; border-radius: 10px;")
        else: bubble.setStyleSheet("background-color: #555555; border-radius: 10px;")
        
        bubble_layout.addWidget(sender_lbl)
        bubble_layout.addWidget(text_lbl)
        self.chat_layout.addWidget(bubble)
        
        QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()))

    def show_loading(self, show=True):
        if show:
            self.loading_label.setText("Agents are currently formulating readings in parallel...")
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
        self.stats_label.setText("")
        self.scatter.setData(pos=np.empty((0,3)))
        self.camera_active = True
        self.init_camera()
        self.status_label.setText("Align your palm with the camera.")
        self.scan_start_time = None
        self.chat_history_for_export = []
        
        while self.chat_layout.count():
            item = self.chat_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
                
        self.add_chat_bubble("System", "Rescanning... Align your hand.")
        self.timer.start(30)

    def update_camera(self):
        if not self.camera_active: return
        ret, frame = self.cap.read()
        if not ret: return
            
        frame = cv2.flip(frame, 1)
        self.clean_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Keep clean copy
        
        # Pipeline processing
        overlay_frame, hand_stats, hands_present = self.cv_pipeline.process_frame(frame)
        
        if hands_present:
            self.current_hand_stats = hand_stats
            # Update stats UI live
            stat_text = f"Shape: {hand_stats['element']} | Palm W/L: {hand_stats['palm_width']}/{hand_stats['palm_length']}"
            self.stats_label.setText(stat_text)
            
            # Update 3D PyqtGraph
            pts = []
            for lm in hands_present[0].landmark:
                pts.append([lm.x - 0.5, -(lm.y - 0.5), lm.z])
            pts = np.array(pts)
            # Add cyan color to nodes
            colors = np.ones((21, 4))
            colors[:, 0] = 0.0
            colors[:, 1] = 1.0
            colors[:, 2] = 1.0
            self.scatter.setData(pos=pts, color=colors, size=0.05, pxMode=False)
            
            if self.scan_start_time is None:
                self.scan_start_time = time.time()
                self.status_label.setText("Extracting Palm Lines... Hold 3 sec")
            elif time.time() - self.scan_start_time > 3.0:
                self.capture_image(overlay_frame)
                return
        else:
            self.scan_start_time = None
            self.status_label.setText("Align your palm with the camera.")
            self.stats_label.setText("3D Mesh topology mapping running...")
            self.scatter.setData(pos=np.empty((0,3)))

        self.display_frame(overlay_frame)

    def display_frame(self, frame_bgr):
        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_bgr.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image).scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

    def capture_image(self, frozen_frame):
        self.camera_active = False
        self.timer.stop()
        self.cap.release()
        self.status_label.setText(f"Scan Complete. Captured 3D Mesh and Canny Line Map.")
        self.rescan_btn.show()
        self.export_btn.show()
        
        self.display_frame(frozen_frame)

        img = Image.fromarray(self.clean_rgb_frame)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        self.captured_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        self.add_chat_bubble("System", "Initiating Multi-Agent Debate. Awaiting reports from A, B, and C.")
        
        # Dispatch to AI Debate Worker
        self.show_loading(True)
        self.debate_worker = DebateWorker(self.llm_service, self.captured_image_base64, self.current_hand_stats)
        self.debate_worker.response_ready.connect(self.on_agent_response)
        self.debate_worker.finished_debate.connect(self.on_debate_finished)
        self.debate_worker.start()

    def send_message(self):
        text = self.chat_input.text()
        if not text.strip() or self.camera_active: return
        self.chat_input.clear()
        self.add_chat_bubble("You", text)
        
        self.show_loading(True)
        self.worker = AIWorker(self.llm_service, text)
        self.worker.response_ready.connect(lambda txt: self.on_agent_response("Synthesizer", txt))
        self.worker.finished.connect(self.on_debate_finished)
        self.worker.start()

    @pyqtSlot(str, str)
    def on_agent_response(self, sender, text):
        self.add_chat_bubble(sender, text)

    @pyqtSlot()
    def on_debate_finished(self):
        self.show_loading(False)

    def export_to_pdf(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Reading", "Reading.pdf", "PDF (*.pdf)")
        if not filename: return
        try:
            doc = SimpleDocTemplate(filename, pagesize=letter)
            styles = getSampleStyleSheet()
            Story = [Paragraph("Multi-Agent Professional Reading", styles["Heading1"]), Spacer(1, 0.2*inch)]
            
            img = Image.fromarray(self.clean_rgb_frame)
            img.save("temp.jpg")
            Story.append(RLImage("temp.jpg", width=3*inch, height=3*inch))
            Story.append(Spacer(1, 0.2*inch))

            for s, t in self.chat_history_for_export:
                Story.append(Paragraph(f"<b>{s}:</b> {t}", styles["Normal"]))
                Story.append(Spacer(1, 0.1*inch))
            
            doc.build(Story)
            os.remove("temp.jpg")
            self.add_chat_bubble("System", f"Exported: {filename}")
        except Exception as e:
            self.add_chat_bubble("System", f"PDF Error: {e}")
