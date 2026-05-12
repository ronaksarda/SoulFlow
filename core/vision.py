import cv2
import threading
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.face_detection as mp_face

class CameraStream:
    """Threaded camera capture for smooth high-speed video ingestion."""
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = False, None
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

def init_ai_models():
    """Initializes high-fidelity MediaPipe solutions."""
    hands = mp_hands.Hands(
        static_image_mode=False, 
        model_complexity=1, # Increased to 1 for better robustness with partial hand visibility
        max_num_hands=2,
        min_detection_confidence=0.5, # More lenient detection
        min_tracking_confidence=0.5
    )
    face = mp_face.FaceDetection(
        model_selection=0, # 0 for short-range (within 2m), faster
        min_detection_confidence=0.6
    )
    return hands, face
