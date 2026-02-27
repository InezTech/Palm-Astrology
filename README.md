# 🔮 Palm-Astrology: Enterprise Palmistry AI

[![Build and Test CI](https://github.com/InezTech/Palm-Astrology/actions/workflows/build.yml/badge.svg)](https://github.com/InezTech/Palm-Astrology/actions/workflows/build.yml)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, privacy-first computer vision and generative AI pipeline for real-time palmistry assessment. This application bridges **deterministic machine learning calculations** with **multimodal large language models (Vision LLMs)** and a **Multi-Agent Synthesizer System** to generate highly clinical, professional consulting reports.

---

## ⚡ Core Architecture

This project is built using a clean MVC (Model-View-Controller) architecture, decoupling the UI from the heavy CV/AI processing pipelines. 

```mermaid
graph TD;
    Camera[WebCam Feed] --> CVPipeline;
    CVPipeline -->|MediaPipe Landmarks| MLMath[Calculates Palm w/l, element shape]
    CVPipeline -->|OpenCV Canny| LineExtraction[Extracts Palm Lines]
    CVPipeline -->|3D Topology| UI[PyQt6 Interface + PyQtGraph]
    MLMath --> Agents[Multi-Agent System]
    LineExtraction --> Agents
    Agents -->|Traditionalist| Debate
    Agents -->|Psychologist| Debate
    Agents -->|Skeptic| Debate
    Debate --> Synthesizer[Final Reading]
    Synthesizer --> ChatUI
```

## 💻 Tech Stack

- **Frontend / GUI**: PyQt6, PyQtGraph, OpenGL
- **Computer Vision**: OpenCV (cv2), MediaPipe SDK
- **AI / LLMs**: OpenAI (GPT-4o / Realtime API), LangChain concepts
- **CI / CD**: GitHub Actions (Automated multi-os testing & code building)
- **Reporting**: ReportLab (Automated PDF generation)

---

## 🔥 Technical Highlights (The "Wow" Factor)

### 1. Multi-Agent Debate System
Leverages a multi-agent AI architecture where three distinct personas (The Traditionalist, The Psychologist, and The Skeptic) analyze the extracted hand biometrics simultaneously. Their insights are debated and then compiled by a "Synthesizer" AI to provide a highly nuanced, robust, and professional reading.

### 2. Live 3D Topology Mapping
Integrates `pyqtgraph` and OpenGL to render an interactive, real-time 3D wireframe mesh of the user's hand, enhancing the visual experience and demonstrating advanced data visualization capabilities.

### 3. Advanced On-Device Computer Vision
We don't just send images to an API. The `CVPipeline` performs real-time data extraction at 30 FPS:
*   **Dimensional Analysis:** Uses `MediaPipe` landmarks to mathematically calculate the ratio of palm width, palm length, and finger length. It uses deterministic heuristics to identify the user's "elemental hand shape" (Earth, Air, Fire, Water).
*   **Canny Edge Extraction:** Implements an OpenCV pipeline with Gaussian blurring and Canny edge detection, masked dynamically *only* to the region of the palm.

### 4. Professional Export System
Users can export their comprehensive, synthesized multi-agent readings into beautifully formatted PDF reports using `ReportLab`, ideal for enterprise or clinical consulting environments.

### 5. Robust CI/CD Pipeline
Automated testing and compilation are handled via **GitHub Actions**. Every push runs tests on an isolated macOS runner, ensuring that cross-platform UI frameworks like `PyQt6` and dynamic libraries (`libGL`, `XCB`) are functioning correctly, avoiding the classic "it works on my machine" problem.

---

## 🛠️ Technical Challenges Overcome

*   **Jitter in Real-time Processing:** Passing raw `MediaPipe` landmarks into the UI thread caused serious flickering. This was solved by separating the CV pipeline state from the PyQt `QTimer` UI loop, ensuring smooth 30FPS AR overlays and 3D rendering.
*   **Multimodal Synthesis & Prompt Leaking:** Sending a high-res image to an LLM often results in generic responses. By pre-calculating the mathematical ratios and routing them through a multi-agent debate, the system grounds the model in hard deterministic data, eliminating AI hallucinations and ensuring clinical precision.

---

## 🚀 Quick Start Setup

Ensure you have Python 3.11 installed.

1. **Clone the Repository:**
```bash
git clone https://github.com/InezTech/Palm-Astrology.git
cd Palm-Astrology
```

2. **Set up the Virtual Environment & Dependencies:**
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Configure the Environment:**
Create a `.env` file in the root directory and add your keys:
```env
OPENAI_API_KEY=your_api_key_here
CAMERA_INDEX=0
```

4. **Launch the Application:**
```bash
export PYTHONPATH=.
python src/main.py
```
