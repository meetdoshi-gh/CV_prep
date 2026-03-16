import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
from collections import deque
from utils.AsyncCamera import AsyncCamera

# ###########################################################################################################
# # Simpllest way 
# ###########################################################################################################

# model = YOLO("yolo26n.pt")
# outputs = model.predict(source="0", device="cuda:0", show=True)

# ###########################################################################################################
# ###########################################################################################################

###########################################################################################################
###########################################################################################################
# Typical Way 
###########################################################################################################
###########################################################################################################

# Device & model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selected_model = "yolo26n.pt"
# selected_model = "yolo26n-seg.pt"
# selected_model = "yolo26n.engine"
model = YOLO(selected_model)
# model.to(device=device)

# Camera capture

fps_tally = deque(maxlen=100)

# cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap = AsyncCamera(1)

prev_time = time.time()

while 1:
    flag, frame = cap.read()

    if not flag:
        break

    # YOLO does not require any exclusive preprocessing, 
    # it does that inherently

    # inference
    inf_started = time.time()
    results = model(frame, stream=True, verbose=False, device="cuda:0",
                    conf=0.5, iou=0.5)
    inf_ended = time.time()

    # output = results[-1].plot() #when stream is False
    for result in results:
        output = result.plot()


    inf_time = (inf_ended - inf_started)*1000

    cv2.putText(output,
                f"Inference Time: {inf_time:0.1f}ms",
                (25, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,0,0),
                1)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    fps_tally.append(fps)
    prev_time = curr_time
    cv2.putText(output,
                f"FPS: {fps:0.2f}",
                (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,0,0),
                1)
    cv2.imshow("YOLO Inference", output)

    if cv2.waitKey(2) == ord("q"):
        break

print(f" Average FPS for last 100 frames: {np.mean(np.array(fps_tally))}")
cap.release()
cv2.destroyAllWindows()