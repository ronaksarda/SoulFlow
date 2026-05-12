import math
import numpy as np
from collections import deque

def is_open_palm(lm) -> bool:
    """Detects if a hand is wide open using relative tip-to-wrist ratios."""
    extended = 0
    for tip, mcp in [(8,5), (12,9), (16,13), (20,17)]:
        d_tip = math.hypot(lm[tip].x - lm[0].x, lm[tip].y - lm[0].y)
        d_mcp = math.hypot(lm[mcp].x - lm[0].x, lm[mcp].y - lm[0].y)
        if d_tip > d_mcp * 1.3: extended += 1
    
    d_thumb = math.hypot(lm[4].x - lm[2].x, lm[4].y - lm[2].y)
    d_base = math.hypot(lm[3].x - lm[2].x, lm[3].y - lm[2].y)
    if d_thumb > d_base * 1.1: extended += 1
    
    return extended >= 4

def is_index_only(lm) -> bool:
    """Robust index-extension detection for the 'Pencil' gesture."""
    d_index = math.hypot(lm[8].x - lm[0].x, lm[8].y - lm[0].y)
    d_mcp5 = math.hypot(lm[5].x - lm[0].x, lm[5].y - lm[0].y)
    if d_index < d_mcp5 * 1.4: return False
    
    curled = 0
    for tip, mcp in [(12,9), (16,13), (20,17)]:
        d_tip = math.hypot(lm[tip].x - lm[0].x, lm[tip].y - lm[0].y)
        d_mcp = math.hypot(lm[mcp].x - lm[0].x, lm[mcp].y - lm[0].y)
        if d_tip < d_mcp * 1.1: curled += 1
    return curled >= 3

class GestureManager:
    """Handles time-based state transitions with temporal catch-up buffering."""
    def __init__(self, start_delay=0.08, stop_delay=0.02):
        from core.filters import OneEuroFilter
        self.start_delay = start_delay
        self.stop_delay = stop_delay
        self.start_time = None
        self.state = "NONE"
        self.pending_state = "NONE"
        self.buffer = deque(maxlen=5)
        self.z_filter = OneEuroFilter(min_cutoff=1.0, beta=0.05) # Smoother Z for depth
        self.is_pressed = False # Persistence for hysteresis

    def update(self, lm, now, raw_pos):
        import config
        is_pencil = is_index_only(lm)
        is_eraser = is_open_palm(lm)
        
        # 1. Filtered Z-Depth
        raw_z = lm[8].z
        smooth_z = self.z_filter(raw_z, now)
        
        # 2. Fixed Depth check for 'Air Touch'
        # MediaPipe's Z is already normalized relative to hand size.
        # -0.15 means the finger is extended ~15% of the hand's length towards the camera.
        press_thresh = config.DEPTH_THRESHOLD 
        release_thresh = press_thresh + 0.05 # Hysteresis gap
        
        if not self.is_pressed:
            if smooth_z < press_thresh: self.is_pressed = True
        else:
            if smooth_z > release_thresh: self.is_pressed = False

        if is_eraser:
            new_state = "ERASER"
        elif is_pencil and self.is_pressed:
            new_state = "PENCIL"
        elif is_pencil and not self.is_pressed:
            new_state = "HOVER"
        else:
            new_state = "NONE"

        self.buffer.append(raw_pos)

        if new_state != self.pending_state:
            self.pending_state = new_state
            self.start_time = now
        
        elapsed = now - self.start_time if self.start_time else 0
        req_delay = self.start_delay if new_state != "NONE" else self.stop_delay

        if elapsed >= req_delay:
            just_activated = (self.state == "NONE" and new_state != "NONE")
            self.state = new_state
            return self.state, (list(self.buffer) if just_activated else [])
        
        return self.state, []

    def reset(self):
        self.state = "NONE"
        self.pending_state = "NONE"
        self.start_time = None
        self.buffer.clear()
