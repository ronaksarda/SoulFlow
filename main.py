import cv2
import numpy as np
import time
import math
from collections import deque
from datetime import datetime
import sys
import multiprocessing as mp

import config
from core.vision import CameraStream, init_ai_models
from core.filters import OneEuroFilter
from core.gestures import GestureManager
from core.emotion import load_emotion_net, analyze_emotion
from ui.hud import draw_hud, draw_face_box
from ui.canvas import init_canvas, apply_neon_bloom, draw_smoothed_trail, draw_neon_skeleton

def inference_worker(frame_q, result_q):
    """Heavy-lifting AI process: Hands, Face, and Emotion."""
    hands_mp, face_mp = init_ai_models()
    emo_net = load_emotion_net()
    
    while True:
        try:
            # Get latest frame, skip if queue is backing up
            frame = None
            while not frame_q.empty():
                frame = frame_q.get_nowait()
            
            if frame is None:
                time.sleep(0.001)
                continue
            
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. Hand Tracking
            hand_res = hands_mp.process(rgb)
            hands_data = []
            if hand_res.multi_hand_landmarks:
                for i, hlm in enumerate(hand_res.multi_hand_landmarks):
                    score = hand_res.multi_handedness[i].classification[0].score
                    label = hand_res.multi_handedness[i].classification[0].label
                    # Convert landmarks to serializable list for Queue
                    lm_list = [(l.x, l.y, l.z) for l in hlm.landmark]
                    hands_data.append({"label": label, "landmarks": lm_list, "score": score})

            # 2. Face & Emotion
            emo_data = None
            face_res = face_mp.process(rgb)
            if face_res.detections:
                best = max(face_res.detections, key=lambda d: d.location_data.relative_bounding_box.width)
                bb = best.location_data.relative_bounding_box
                # Store normalized coordinates to avoid resolution mismatch
                norm_bbox = (bb.xmin, bb.ymin, bb.width, bb.height)
                
                # ROI extraction for emotion needs pixel coords (relative to 'small' frame)
                fx, fy, fw, fh = int(bb.xmin*w), int(bb.ymin*h), int(bb.width*w), int(bb.height*h)
                pad = int(fw * 0.15)
                x1, y1, x2, y2 = max(0, fx-pad), max(0, fy-pad), min(w, fx+fw+pad), min(h, fy+fh+pad)
                roi = frame[y1:y2, x1:x2]
                
                if roi.size > 0:
                    label, conf, probs = analyze_emotion(emo_net, roi)
                    emo_data = {"label": label, "conf": conf, "probs": probs, "norm_bbox": norm_bbox}

            result_q.put((hands_data, emo_data))
            
        except Exception as e:
            print(f"[!] Inference Error: {e}")
            time.sleep(0.1)

def main():
    # 1. Pipeline Setup
    mp.set_start_method('spawn', force=True) # Required for Windows + AI libs
    frame_q = mp.Queue(maxsize=2)
    result_q = mp.Queue(maxsize=2)
    
    stream = CameraStream(0)
    time.sleep(1.0)
    
    proc = mp.Process(target=inference_worker, args=(frame_q, result_q), daemon=True)
    proc.start()

    cv2.namedWindow('SoulFlow', cv2.WINDOW_NORMAL)
    
    # 2. Persistent State (Render Process)
    trails = {"Right": deque(maxlen=config.TRAIL_LEN), "Left": deque(maxlen=config.TRAIL_LEN)}
    filters_x = {"Right": OneEuroFilter(), "Left": OneEuroFilter()}
    filters_y = {"Right": OneEuroFilter(), "Left": OneEuroFilter()}
    gestures = {"Right": GestureManager(), "Left": GestureManager()}
    
    raw_buffers_x = {"Right": deque(maxlen=3), "Left": deque(maxlen=3)}
    raw_buffers_y = {"Right": deque(maxlen=3), "Left": deque(maxlen=3)}
    
    velocities = {"Right": 0.0, "Left": 0.0}
    smooth_thick = {"Right": 5.0, "Left": 5.0}
    pen_down     = {"Right": False, "Left": False}
    prev_palm    = {"Right": None, "Left": None}
    prev_mid     = {"Right": None, "Left": None}
    
    canvas = None
    smooth_face = None
    smooth_probs = None
    emo_label, emo_conf, emo_color = None, 0.0, (0, 255, 0)
    
    current_ai_data = ([], None)
    prev_t = time.time()

    print("[+] System Live. Multi-Processing Active. Press 'Q' to quit.")

    try:
        # 3. Main Render Loop
        while True:
            now = time.time()
            dt = max(now - prev_t, 1e-6)
            prev_t = now
            fps = 1.0 / dt

            ret, frame = stream.read()
            if not ret or frame is None: continue
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if canvas is None: canvas = init_canvas(h, w)

            # Feed the inference worker (Downscale to 360p for 4x speedup, display stays 720p)
            if not frame_q.full():
                small = cv2.resize(frame, (640, 360))
                frame_q.put(small)

            # Pull latest AI results
            while not result_q.empty():
                current_ai_data = result_q.get_nowait()

            hands_data, emo_data = current_ai_data
            seen_labels = set()

            # --- PART A: HAND RENDERING ---
            for hand in hands_data:
                label, lm_list, score = hand['label'], hand['landmarks'], hand['score']
                if score < 0.4: continue
                seen_labels.add(label)
                
                trail = trails[label]
                neon = config.HAND_COLORS[label]
                
                # Reconstruction for gesture functions (needs .x .y .z attributes)
                class MockLM: 
                    def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z
                mock_lms = [MockLM(*pt) for pt in lm_list]
                
                raw_x, raw_y = int(mock_lms[8].x * w), int(mock_lms[8].y * h)
                raw_buffers_x[label].append(raw_x)
                raw_buffers_y[label].append(raw_y)
                
                clean_x = sorted(list(raw_buffers_x[label]))[len(raw_buffers_x[label])//2]
                clean_y = sorted(list(raw_buffers_y[label]))[len(raw_buffers_y[label])//2]
                clean_pos = (clean_x, clean_y)

                mode, catch_up = gestures[label].update(mock_lms, now, clean_pos)
                px_palm, py_palm = int(mock_lms[9].x * w), int(mock_lms[9].y * h)

                vx, vy = filters_x[label].velocity, filters_y[label].velocity
                v_combined_px = math.hypot(vx, vy)
                velocities[label] = velocities[label] * 0.85 + (v_combined_px * config.PX_TO_M) * 0.15
                
                target_th = max(3, min(12, int(12 - (velocities[label] * 5.0))))
                smooth_thick[label] = smooth_thick[label]*0.7 + target_th*0.3
                thickness = int(smooth_thick[label])

                # 1. ERASER
                if mode == "ERASER":
                    if prev_palm[label] is not None:
                        cv2.line(canvas, prev_palm[label], (px_palm, py_palm), (0,0,0), (config.ERASER_R+20)*2, cv2.LINE_AA)
                    cv2.circle(canvas, (px_palm, py_palm), config.ERASER_R+20, (0,0,0), -1)
                    cv2.circle(frame, (px_palm, py_palm), config.ERASER_R+20, (60,60,60), 2, cv2.LINE_AA)
                    draw_neon_skeleton(frame, mock_lms, neon, mode="ERASER")
                    prev_palm[label] = (px_palm, py_palm); pen_down[label] = False; trail.clear(); continue
                
                prev_palm[label] = None

                # 2. PENCIL / HOVER (Depth-based)
                cx = int(filters_x[label](clean_pos[0], now))
                cy = int(filters_y[label](clean_pos[1], now))

                if mode == "PENCIL":
                    if catch_up:
                        for i in range(len(catch_up)-1):
                            cv2.line(canvas, catch_up[i], catch_up[i+1], neon, thickness, cv2.LINE_AA)
                        for p in catch_up: trail.appendleft(p)
                        pen_down[label] = True

                    trail.appendleft((cx, cy))
                    if pen_down[label] and len(trail) >= 2:
                        p1, p2 = trail[1], trail[0]
                        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                        if math.hypot(p1[0]-p2[0], p1[1]-p2[1]) < (45 + velocities[label]*35):
                            if prev_mid[label] is not None:
                                cv2.line(canvas, prev_mid[label], mid, neon, thickness, cv2.LINE_AA)
                            else:
                                cv2.line(canvas, p1, mid, neon, thickness, cv2.LINE_AA)
                            prev_mid[label] = mid
                        else:
                            trail.clear(); pen_down[label] = False; prev_mid[label] = None
                    pen_down[label] = True
                else:
                    # HOVER Mode: Cursor visual only, no drawing
                    pen_down[label] = False
                    trail.clear()
                    prev_mid[label] = None
                    # Visual feedback for HOVER (Dashed or faint circle)
                    cv2.circle(frame, (cx, cy), 12, neon, 1, cv2.LINE_AA)

                draw_smoothed_trail(frame, trail, neon, thickness)
                draw_neon_skeleton(frame, mock_lms, neon, mode="PENCIL" if mode=="PENCIL" else "HOVER")
                cv2.circle(frame, (cx, cy), 4, (255,255,255), -1, cv2.LINE_AA)

            # Clean inactive hands
            for lbl in ("Right", "Left"):
                if lbl not in seen_labels:
                    pen_down[lbl] = False; prev_mid[lbl] = None; trails[lbl].clear(); gestures[lbl].reset()

            # --- PART B: EMOTION RENDERING ---
            if emo_data:
                nx, ny, nw, nh = emo_data['norm_bbox']
                fx, fy, fw, fh = int(nx*w), int(ny*h), int(nw*w), int(nh*h)
                
                probs = emo_data['probs']
                smooth_probs = probs if smooth_probs is None else smooth_probs*0.7 + probs*0.3
                emo_label = config.EMOTIONS[np.argmax(smooth_probs)]
                emo_conf = float(smooth_probs.max())
                emo_color = config.EMOTION_COLORS.get(emo_label, (0,255,0))
                target_f = np.array([fx, fy, fw, fh], dtype=np.float32)
                smooth_face = target_f if smooth_face is None else smooth_face*0.8 + target_f*0.2
            else:
                smooth_face = None

            # --- PART C: FINAL RENDER ---
            frame = apply_neon_bloom(frame, canvas)
            draw_face_box(frame, smooth_face, emo_label, emo_conf, emo_color)
            
            # Pass FPS and current right hand color (as default ink) to HUD
            current_brush = config.HAND_COLORS.get("Right", (255, 255, 255))
            draw_hud(frame, w, h, velocities, emo_label, emo_conf, emo_color, fps=fps, brush_color=current_brush)

            cv2.imshow('SoulFlow', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')): break
            elif key == ord('s'): cv2.imwrite(f"canvas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", canvas)
            elif key == ord('c'): canvas[:] = 0

    except Exception as e:
        print(f"[!] System Error: {e}")
    finally:
        print("[*] Shutting down SoulFlow...")
        stream.stop()
        proc.terminate() # Explicitly kill the AI worker
        proc.join()
        cv2.destroyAllWindows()
        sys.exit(0)

if __name__ == "__main__":
    main()
