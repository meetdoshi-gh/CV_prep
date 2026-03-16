import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
from collections import deque, defaultdict
from utils.AsyncCamera import AsyncCamera

# ###########################################################################################################
# # Simpllest way 
# ###########################################################################################################

# model = YOLO("yolo26n.pt")
# outputs = model.track(source="0", device="cuda:0", show=True, 
#                         tracker="bytetrack.yaml",
#                         persist=True)

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
model = YOLO(selected_model)
model.to(device=device)

# For metrics
fps_tally = deque(maxlen=100)
track_history = defaultdict(list)

# Camera capture
# cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap = AsyncCamera(1)

prev_time = time.time()

while 1: #cap.isOpened():
    flag, frame = cap.read()

    if not flag:
        break

    # YOLO does not require any exclusive preprocessing, 
    # it does that inherently

    # inference
    inf_started = time.time()
    results = model.track(frame, stream=True, verbose=False, tracker="bytetrack.yaml",
                            persist=True, conf=0.5, iou=0.5)
    inf_ended = time.time()

    # annotated_img = results[-1].plot() #when stream is False 
    for result in results:
        annotated_img = result.plot()


    inf_time = (inf_ended - inf_started)*1000000

    cv2.putText(annotated_img,
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

    cv2.putText(annotated_img,
                f"FPS: {fps:0.2f}",
                (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,0,0),
                1)
    
    # # Draw track trails
    # if result.boxes.id is not None:
    #     boxes = result.boxes.xywh.cpu()
    #     track_ids = result.boxes.id.cpu().int().tolist()

    #     for box, track_id in zip(boxes, track_ids):
    #         x, y, w, h = box
    #         center = (float(x), float(y))

    #         track = track_history[track_id]
    #         track.append(center)

    #         points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
    #         cv2.polylines(img=annotated_img, 
    #                         pts=[points],
    #                         isClosed=False,
    #                         color=(230, 230, 230),
    #                         thickness=2)


    cv2.imshow("YOLO Inference", annotated_img)

    if cv2.waitKey(2) == ord("q"):
        break

print(f"Average FPS of last 100 frames: {np.mean(np.array(fps_tally))}")
cap.release()
cv2.destroyAllWindows()