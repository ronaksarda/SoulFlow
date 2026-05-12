import os
import cv2
import requests
import numpy as np
from config import MODEL_URL, MODEL_PATH, EMOTIONS

def download_model():
    """Ensures the ONNX emotion recognition model is available locally."""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:
        print("[*] Downloading emotion model (35MB)...")
        r = requests.get(MODEL_URL, allow_redirects=True)
        if r.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)
            print("[+] Model ready.")
        else:
            print("[!] Download failed. Check internet connection.")

def load_emotion_net():
    """Loads the ONNX network using OpenCV DNN."""
    if not os.path.exists(MODEL_PATH):
        download_model()
    return cv2.dnn.readNetFromONNX(MODEL_PATH)

def analyze_emotion(net, face_roi):
    """Performs deep inference on a face ROI to classify emotion."""
    if face_roi.size == 0: return None, 0.0
    
    # Pre-processing
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blob = cv2.dnn.blobFromImage(gray, 1.0, (64,64), (0), swapRB=False, crop=False)
    
    net.setInput(blob)
    preds = net.forward()
    
    # Softmax
    probs = np.exp(preds[0] - preds[0].max())
    probs /= probs.sum()
    
    label = EMOTIONS[np.argmax(probs)]
    conf = float(probs.max())
    return label, conf, probs
