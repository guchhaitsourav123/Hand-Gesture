import cv2
import numpy as np
import time
import sys
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print("[INFO] Downloading model (~8 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[INFO] Download complete.")

download_model()

WRIST=0; THUMB_MCP=2; THUMB_IP=3; THUMB_TIP=4
INDEX_PIP=6; INDEX_TIP=8
MIDDLE_MCP=9; MIDDLE_PIP=10; MIDDLE_TIP=12
RING_PIP=14; RING_TIP=16
PINKY_PIP=18; PINKY_TIP=20

GESTURE_COLORS = {
    "Thumbs Up":   (50, 205, 50),
    "Thumbs Down": (30,  30, 220),
    "Fist":        (0,  165, 255),
    "Unknown":     (180,180, 180),
}

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def classify_gesture(lms):
    def y(i): return lms[i].y
    all_folded = (
        y(INDEX_TIP)  > y(INDEX_PIP)  and
        y(MIDDLE_TIP) > y(MIDDLE_PIP) and
        y(RING_TIP)   > y(RING_PIP)   and
        y(PINKY_TIP)  > y(PINKY_PIP)
    )
    hand_h    = abs(y(WRIST) - y(MIDDLE_MCP)) + 1e-6
    thresh    = 0.20 * hand_h
    thumb_up  = (y(THUMB_MCP) - y(THUMB_TIP)) > thresh
    thumb_dn  = (y(THUMB_TIP) - y(WRIST))     > thresh
    thumb_fld =  y(THUMB_TIP) > y(THUMB_IP)
    if all_folded and thumb_up:  return "Thumbs Up"
    if all_folded and thumb_dn:  return "Thumbs Down"
    if all_folded and thumb_fld: return "Fist"
    return "Unknown"

def draw_hand(frame, lms, w, h):
    pts = [(int(l.x*w), int(l.y*h)) for l in lms]
    for a,b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0,200,255), 2)
    for x,y in pts:
        cv2.circle(frame, (x,y), 5, (255,255,255), -1)

def draw_label(frame, gesture, lms, w, h):
    color = GESTURE_COLORS.get(gesture, (180,180,180))
    x_min = int(min(l.x*w for l in lms))
    y_min = int(min(l.y*h for l in lms))
    (tw,th),_ = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
    cv2.rectangle(frame, (x_min-8, y_min-th-16), (x_min+tw+8, y_min-8), color, -1)
    cv2.putText(frame, gesture, (x_min, y_min-12),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

def draw_hud(frame, fps, gesture, fidx):
    h,w = frame.shape[:2]
    color = GESTURE_COLORS.get(gesture,(180,180,180))
    ov = frame.copy()
    cv2.rectangle(ov,(0,h-54),(w,h),(20,20,20),-1)
    cv2.addWeighted(ov,0.55,frame,0.45,0,frame)
    cv2.putText(frame,f"FPS:{fps:5.1f}",(14,h-18),cv2.FONT_HERSHEY_DUPLEX,0.65,(200,200,200),1)
    cv2.putText(frame,f"Frame:{fidx}",(14,h-36),cv2.FONT_HERSHEY_DUPLEX,0.55,(140,140,140),1)
    cv2.putText(frame,gesture,(w//2-80,h-14),cv2.FONT_HERSHEY_DUPLEX,0.85,color,2)
    cv2.putText(frame,"Hand Gesture Recognition",(10,30),cv2.FONT_HERSHEY_DUPLEX,0.7,(220,220,220),1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("[ERROR] Cannot open camera.")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0

writer = None
for codec in ("avc1","mp4v","XVID"):
    writer = cv2.VideoWriter("gesture_output.mp4",
                             cv2.VideoWriter_fourcc(*codec), FPS, (W,H))
    if writer.isOpened():
        break

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

fps_buf=[]; fidx=0; last=time.perf_counter(); cur_fps=0.0

print("[INFO] Starting — press q to quit.")
with HandLandmarker.create_from_options(options) as det:
    while True:
        ret, frame = cap.read()
        if not ret: break
        fidx += 1

        now = time.perf_counter()
        fps_buf.append(1.0/max(now-last,1e-6))
        if len(fps_buf)>30: fps_buf.pop(0)
        cur_fps = float(np.mean(fps_buf))
        last = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = det.detect_for_video(img, int(time.time()*1000))

        best = "Unknown"
        if res.hand_landmarks:
            for lms in res.hand_landmarks:
                draw_hand(frame, lms, W, H)
                g = classify_gesture(lms)
                draw_label(frame, g, lms, W, H)
                if g != "Unknown": best = g

        draw_hud(frame, cur_fps, best, fidx)
        writer.write(frame)
        cv2.imshow("Hand Gesture Recognition [q=quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"[INFO] Done. {fidx} frames saved.")
