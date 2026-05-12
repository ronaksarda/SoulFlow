import cv2
import numpy as np

def init_canvas(h, w):
    """Creates a blank black canvas for drawing."""
    return np.zeros((h, w, 3), dtype=np.uint8)

def apply_neon_bloom(frame, canvas):
    """
    Applies a high-fidelity multi-pass bloom effect for a true neon light glow.
    """
    # 1. Extract the drawing from the canvas
    mask = canvas.any(axis=2)
    drawing = np.zeros_like(canvas)
    drawing[mask] = canvas[mask]
    
    # 2. Multi-pass bloom (small blur for core, large for halo)
    glow1 = cv2.GaussianBlur(drawing, (15, 15), 0)
    glow2 = cv2.GaussianBlur(drawing, (45, 45), 0)
    
    # Additive blending for intensity
    glow = cv2.addWeighted(glow1, 0.6, glow2, 0.4, 0)
    cv2.add(frame, glow, frame) # Additive bloom
    
    # 3. Layer the sharp drawing on top
    frame[mask] = canvas[mask]
    return frame

def draw_smoothed_trail(frame, trail, neon, thickness):
    """Renders the temporary 'ghost' trail with Chaikin smoothing."""
    from core.gestures import is_index_only # Local import to avoid circular dep
    
    def chaikin_smoothing(points, iterations=2):
        if len(points) < 3: return points
        for _ in range(iterations):
            new_points = [points[0]]
            for i in range(len(points) - 1):
                p0, p1 = np.array(points[i]), np.array(points[i+1])
                q = 0.75 * p0 + 0.25 * p1
                r = 0.25 * p0 + 0.75 * p1
                new_points.extend([tuple(q.astype(int)), tuple(r.astype(int))])
            new_points.append(points[-1])
            points = new_points
        return points

    raw_pts = list(trail)
    if len(raw_pts) > 2:
        smooth_pts = np.array(chaikin_smoothing(raw_pts), np.int32)
        cv2.polylines(frame, [smooth_pts], False, (0,0,0), thickness+4, cv2.LINE_AA)
        cv2.polylines(frame, [smooth_pts], False, neon, thickness, cv2.LINE_AA)
def draw_neon_skeleton(frame, lm_list, color, mode="PENCIL"):
    """
    Renders a premium high-tech skeleton using a list of landmarks.
    """
    h, w = frame.shape[:2]
    
    # 1. Base Connections (Simplified manual drawing to avoid mp_draw dependencies on raw objects)
    connections = [
        (0,1), (1,2), (2,3), (3,4), # Thumb
        (0,5), (5,6), (6,7), (7,8), # Index
        (5,9), (9,10), (10,11), (11,12), # Middle
        (9,13), (13,14), (14,15), (15,16), # Ring
        (13,17), (17,18), (18,19), (19,20), (0,17) # Pinky
    ]
    
    for p1_idx, p2_idx in connections:
        p1 = (int(lm_list[p1_idx].x * w), int(lm_list[p1_idx].y * h))
        p2 = (int(lm_list[p2_idx].x * w), int(lm_list[p2_idx].y * h))
        cv2.line(frame, p1, p2, (40, 40, 40), 1, cv2.LINE_AA)

    if mode == "ERASER":
        # Repulsor Core (Palm Center - Landmark 9)
        cx, cy = int(lm_list[9].x * w), int(lm_list[9].y * h)
        cv2.circle(frame, (cx, cy), 12, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 16, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1, cv2.LINE_AA)
        
        for i in [4, 8, 12, 16, 20]:
            tx, ty = int(lm_list[i].x * w), int(lm_list[i].y * h)
            cv2.circle(frame, (tx, ty), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (tx, ty), 6, color, 1, cv2.LINE_AA)

    elif mode in ["PENCIL", "HOVER"]:
        # Highlight index finger only (5-8)
        for i in [5, 6, 7]:
            p1 = (int(lm_list[i].x * w), int(lm_list[i].y * h))
            p2 = (int(lm_list[i+1].x * w), int(lm_list[i+1].y * h))
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
        # Tip glow
        tx, ty = int(lm_list[8].x * w), int(lm_list[8].y * h)
        if mode == "PENCIL":
            cv2.circle(frame, (tx, ty), 8, color, 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (tx, ty), 6, (100, 100, 100), 1, cv2.LINE_AA)
