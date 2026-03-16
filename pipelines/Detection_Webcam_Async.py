import cv2
import torch
import numpy as np
import time
import threading

from collections import deque
from ultralytics import YOLO


##################################################
# Async Camera Class
##################################################

class AsyncCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = frame

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()



##################################################
# Device + Model
##################################################

device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO("yolo26n.pt")
model.to(device)


##################################################
# Buffers
##################################################

batch_size = 4
history_size = 10

frame_buffer = deque(maxlen=batch_size)
history = deque(maxlen=history_size)
fps_tally = deque(maxlen=100)


##################################################
# Camera
##################################################

cap = AsyncCamera(0)

prev_time = time.time()


##################################################
# Main Loop
##################################################

while True:

    ret, frame = cap.read()

    if not ret or frame is None:
        continue

    frame_buffer.append(frame)

    if len(frame_buffer) < batch_size:
        continue


    ##################################################
    # Batch Inference
    ##################################################

    start = time.time()

    results = model(list(frame_buffer), verbose=False)

    infer_time = (time.time() - start) * 1000


    ##################################################
    # Use newest frame
    ##################################################

    result = results[-1]

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    display_frame = frame_buffer[-1].copy()


    ##################################################
    # Draw Bounding Boxes
    ##################################################

    class_scores = {}

    for box, score, cls in zip(boxes, scores, classes):

        if score < 0.4:
            continue

        x1, y1, x2, y2 = map(int, box)

        label = model.names[cls]

        # draw detection
        cv2.rectangle(display_frame, (x1,y1), (x2,y2), (0,255,0), 2)

        cv2.putText(display_frame,
                    f"{label} {score:.2f}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)

        # store score for smoothing
        if label not in class_scores:
            class_scores[label] = []

        class_scores[label].append(score)


    ##################################################
    # Frame Summary (best score per class)
    ##################################################

    frame_summary = {}

    for label, vals in class_scores.items():
        frame_summary[label] = max(vals)

    history.append(frame_summary)


    ##################################################
    # Moving Average Across Frames
    ##################################################

    avg_scores = {}

    for frame_dict in history:
        for label, score in frame_dict.items():

            if label not in avg_scores:
                avg_scores[label] = []

            avg_scores[label].append(score)

    smoothed = {
        label: np.mean(vals)
        for label, vals in avg_scores.items()
    }


    ##################################################
    # Top Scene Classes
    ##################################################

    top_classes = sorted(
        smoothed.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]


    y = 30

    for label, score in top_classes:

        cv2.putText(display_frame,
                    f"{label}: {score:.2f}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,0),
                    2)

        y += 30


    ##################################################
    # FPS
    ##################################################

    curr = time.time()

    fps = 1 / (curr - prev_time)

    fps_tally.append(fps)

    prev_time = curr


    cv2.putText(display_frame,
                f"FPS: {fps:.1f}",
                (20,400),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2)

    cv2.putText(display_frame,
                f"Inference: {infer_time:.1f} ms",
                (20,430),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2)


    ##################################################
    # Display
    ##################################################

    cv2.imshow("YOLO Detection + Scene Smoothing", display_frame)

    if cv2.waitKey(1) == 27:
        break


##################################################
# Cleanup
##################################################

print("Average FPS:", np.mean(fps_tally))

cap.release()
cv2.destroyAllWindows()