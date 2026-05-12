import cv2
import numpy as np
from config import (HUD_BG_COLOR, HUD_TEXT_COLOR, ACCENT_LINE_WIDTH, 
                    GLASS_ALPHA, EMOTION_COLORS)

def draw_hud(frame, w, h, velocities, emotion_label, emo_conf, emotion_color, fps=0, brush_color=(255,255,255)):
    """Renders a clean, high-tech telemetry panel."""
    ui_w = 280; x0 = w - ui_w - 20; y0 = 20
    
    # 1. Glass Background
    roi = frame[y0:y0+200, x0:x0+ui_w]
    dark = np.full_like(roi, (15, 15, 20), dtype=np.uint8)
    cv2.addWeighted(dark, 0.8, roi, 0.2, 0, roi)
    cv2.rectangle(roi, (0, 0), (ui_w, 200), (60, 60, 65), 1) # Subtle border
    frame[y0:y0+200, x0:x0+ui_w] = roi
    
    # Side accent glow
    cv2.line(frame, (x0, y0), (x0, y0+200), emotion_color, 2, cv2.LINE_AA)
    
    # System Title & FPS
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, "SOULFLOW // CORE", (x0+16, y0+30), font, 0.55, (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{int(fps)} FPS", (x0+ui_w-65, y0+30), font, 0.4, (100,255,100), 1, cv2.LINE_AA)
    
    # Velocity Meters
    for i, (label, v) in enumerate(velocities.items()):
        y_off = y0 + 70 + (i * 40)
        bar_w = int(min(1, v / 3.0) * 150)
        color = (255, 150, 150) if label == "Right" else (150, 255, 255)
        
        cv2.putText(frame, f"{label[0]}: {v:>4.2f}m/s", (x0+16, y_off), font, 0.45, (180,180,180), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x0+100, y_off-10), (x0+250, y_off-2), (40,40,40), -1)
        cv2.rectangle(frame, (x0+100, y_off-10), (x0+100+bar_w, y_off-2), color, -1)
    
    # Tracking Status
    tracking_active = any(v > 0.05 for v in velocities.values())
    status_color = (0, 255, 0) if tracking_active else (0, 0, 255)
    status_text = "TRACKING: OK" if tracking_active else "TRACKING: LOST"
    cv2.putText(frame, status_text, (x0+16, y0+150), font, 0.45, status_color, 1, cv2.LINE_AA)
    
    # Brush Preview
    cv2.putText(frame, "INK:", (x0+16, y0+180), font, 0.45, (180,180,180), 1, cv2.LINE_AA)
    cv2.circle(frame, (x0+60, y0+176), 8, brush_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (x0+60, y0+176), 10, (200,200,200), 1, cv2.LINE_AA)

    if not tracking_active:
        cv2.putText(frame, "KEEP PALM IN VIEW", (w//2 - 90, h//2 + 100), font, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    # Emotion
    status = f"MOOD: {emotion_label}" if emotion_label else "SCANNING..."
    cv2.putText(frame, status, (x0+100, y0+180), font, 0.45, emotion_color, 1, cv2.LINE_AA)
    
    # Bottom Controls
    cv2.putText(frame, "[S]AVE   [C]LEAR   [Q]UIT", (w//2 - 120, h-25), font, 0.45, (150,150,150), 1, cv2.LINE_AA)

def draw_face_box(frame, smooth_face, emotion, conf, color):
    """Renders a futuristic targeting-style box around the user's face."""
    if smooth_face is None or emotion is None: return
    sx, sy, sfw, sfh = [int(v) for v in smooth_face]
    
    # Corner brackets (Futuristic UI style)
    l = 20
    # Top Left
    cv2.line(frame, (sx, sy), (sx+l, sy), color, 2, cv2.LINE_AA)
    cv2.line(frame, (sx, sy), (sx, sy+l), color, 2, cv2.LINE_AA)
    # Top Right
    cv2.line(frame, (sx+sfw, sy), (sx+sfw-l, sy), color, 2, cv2.LINE_AA)
    cv2.line(frame, (sx+sfw, sy), (sx+sfw, sy+l), color, 2, cv2.LINE_AA)
    # Bottom Left
    cv2.line(frame, (sx, sy+sfh), (sx+l, sy+sfh), color, 2, cv2.LINE_AA)
    cv2.line(frame, (sx, sy+sfh), (sx, sy+sfh-l), color, 2, cv2.LINE_AA)
    # Bottom Right
    cv2.line(frame, (sx+sfw, sy+sfh), (sx+sfw-l, sy+sfh), color, 2, cv2.LINE_AA)
    cv2.line(frame, (sx+sfw, sy+sfh), (sx+sfw, sy+sfh-l), color, 2, cv2.LINE_AA)
    
    # Floating Label
    label = f"{emotion} {int(conf*100)}%"
    cv2.putText(frame, label, (sx, sy-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)
