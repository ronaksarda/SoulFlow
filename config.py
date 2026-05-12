import os

# --- MODEL CONFIG ---
MODEL_URL  = ("https://github.com/spmallick/learnopencv/raw/master/"
              "Facial-Emotion-Recognition/emotion-ferplus-8.onnx")
MODEL_PATH = "emotion-ferplus-8.onnx"

EMOTIONS = ['NEUTRAL','HAPPY','SURPRISED','SAD','ANGRY','DISGUSTED','SCARED','CONTEMPT']

# --- PREMIUM DARK PALETTE ---
# Using deep, saturated colors for better contrast against dark backgrounds
EMOTION_COLORS = {
    'NEUTRAL':   (220, 220, 220), # Off-white
    'HAPPY':     (255, 255, 0),   # Neon Cyan
    'SURPRISED': (0, 255, 255),   # Neon Yellow
    'SAD':       (255, 0, 0),     # Vivid Blue (Sad)
    'ANGRY':     (0, 0, 255),     # Deep Red (Angry)
    'DISGUSTED': (0, 200, 0),     # Emerald
    'SCARED':    (180, 0, 180),   # Purple
    'CONTEMPT':  (0, 120, 255),   # Orange/Gold
}

# Neon Accents for Hand Tracking
HAND_COLORS = {
    "Right": (255, 140, 230), # Neon Lavender
    "Left": (230, 230, 100)   # Soft Turquoise
}

# --- TRACKING CONFIG ---
TRAIL_LEN = 30
ERASER_R  = 30
PX_TO_M   = 0.5 / 1280.0
DEPTH_THRESHOLD = -0.08 # Lenient Z depth for easy 'Air Touch'

# --- UI STYLE ---
HUD_BG_COLOR = (8, 8, 12) # Deep Obsidian
HUD_TEXT_COLOR = (240, 240, 240)
ACCENT_LINE_WIDTH = 3
GLASS_ALPHA = 0.75
