import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
from collections import deque

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device detection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = "cpu"

print(f"""
======> Using {device}
""")

# Model loading
selected_model = "yolo26n.pt"
# selected_model = "yolo26n.engine"
# selected_model = "yolo26n-seg.pt"
model = YOLO(selected_model)

# perfromance measures
fps_tally = deque(maxlen=100)

#camera stream capturing
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# Preprocessing pipeline
from torchvision.transforms import v2
# Define the pipeline once
preprocess = v2.Compose([
    # 1. Resize so the longest edge is 640 (maintains aspect ratio)
    v2.Resize(size=640, max_size=641), 
    
    # 2. Pad or CenterCrop to reach exactly 640x640 with grey background
    v2.CenterCrop(size=(640, 640)), 
    
    # 3. Convert to float and normalize
    v2.ToDtype(torch.float32, scale=True), 

    # # Normalize for ImageNet Statistics
    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



prev_time = time.time()

while True:

    ret, frame0 = cap.read()
    if not ret:
        break

    start = time.time()

    frame = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    # frame = cv2.resize(frame, (640,640))

    
    # numpy -> tensor
    frame_t = torch.from_numpy(frame)
    # PIN MEMORY # async GPU transfer
    frame_t = frame_t.pin_memory().to(device=device, non_blocking=True)

    # HWC → CHW
    frame_t = frame_t.permute(2,0,1)#.float()#/255.0

    #Apply image regarding processing
    frame_t = preprocess(frame_t)

    # add batch dimension
    frame_t = frame_t.unsqueeze(0)

    # # Normalize
    # mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    # std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    # frame_t = (frame_t - mean) / std

    # print(frame_t.shape)



    # inference
    results = model.track(frame_t, verbose=False, tracker="bytetrack.yaml",
                            persist=True, conf=0.4, iou=0.5)
    end = time.time()

    # results[0].orig_img = frame0
    annotated_img = results[0].plot()
    # print(type(annotated_img))

    inf_time = (end - start)*1000

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

    cv2.imshow("YOLO", annotated_img)

    if cv2.waitKey(1) == ord("q"):
        break

print(f"Average FPS of last 100 frames: {np.mean(np.array(fps_tally))}")
cap.release()
cv2.destroyAllWindows()