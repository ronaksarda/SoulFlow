# SoulFlow 🌌  
**Real-Time Multimodal Computer Vision System for Gesture Interaction & Emotion-Aware Rendering**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tracking-00C853)](https://google.github.io/mediapipe/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-F36F21)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Low-latency gesture recognition, emotion inference, and interactive rendering pipeline.**

</div>

---

# Overview

SoulFlow is a computer vision application that combines:

- Real-time hand tracking
- Gesture-driven interaction
- Emotion classification
- Dynamic visual rendering

The system separates rendering and inference into isolated processes to maintain stable frame rates.

---

# Capabilities

## ✨ The Experience
SoulFlow isn't just a tracker; it's a digital mirror that responds to your motion and mood.

### ✋ Gesture-to-Canvas
*   **Neon Trails:** Your fingers leave glowing neon trails that fade and bloom in real-time.
*   **Fluid Motion:** Using OneEuro filters, the drawing feels like physical ink, eliminating digital jitter.
*   **Dynamic Thickness:** Move fast for thick, expressive strokes; slow down for precise, fine lines.
*   **Spatial Eraser:** Simply pinch or change hand pose to clear parts of your creation in 3D space.

### 🧠 Emotion-Aware HUD
*   **Mood Sync:** The entire interface shifts its color palette based on your detected emotion (e.g., Green for Happiness, Red for Anger).
*   **Live Analytics:** A futuristic HUD displays your emotional confidence scores and movement velocity in real-time.
*   **Face Tracking:** The system locks onto your face with a stabilized neon bounding box that adapts to your expressions.

---

# 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| Vision | OpenCV |
| Tracking | MediaPipe |
| Inference | ONNX Runtime |
| Concurrency | multiprocessing |

---

# 📂 Repository Structure

```text
SoulFlow/
│
├── core/
│   ├── emotion.py          # Emotion logic
│   ├── filters.py          # OneEuro filtering
│   ├── gestures.py         # Gesture state machine
│   └── vision.py           # Camera & AI init
│
├── ui/
│   ├── canvas.py           # Neon effects
│   └── hud.py              # Status overlays
│
├── emotion-ferplus-8.onnx   # ONNX Model
├── config.py               # Configuration
├── main.py                 # Entry point
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

```bash
# Clone
git clone https://github.com/yourusername/SoulFlow.git
cd SoulFlow

# Environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/macOS

# Dependencies
pip install -r requirements.txt
```

---

# ▶️ Running
```bash
python main.py
```

---

# 🎮 Controls
| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `S` | Save Canvas |
| `C` | Clear Canvas |

---

# 📜 License
Distributed under the MIT License. See `LICENSE` for details.

---

<div align="center">

## SoulFlow
Real-time AI interaction systems.

</div>