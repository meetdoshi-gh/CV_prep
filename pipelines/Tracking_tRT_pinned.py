import cv2
import torch
from ultralytics import YOLO
import time
from collections import deque

model = YOLO("yolo26n.engine")

fps_tally = deque(maxlen=100)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

from torchvision.transforms import v2

# Define the pipeline once
preprocess = v2.Compose([
    # 1. Resize so the longest edge is 640 (maintains aspect ratio)
    v2.Resize(size=640, max_size=641), 
    
    # 2. Pad or CenterCrop to reach exactly 640x640 with grey background
    v2.CenterCrop(size=(640, 640)), 
    
    # 3. Convert to float and normalize
    v2.ToDtype(torch.float32, scale=True), 
])

# import torchvision.transforms as T

# preprocess = T.Compose([
#     # 1. Resize so the LONGEST side is 640 (maintains aspect ratio)
#     T.Resize(640, max_size=641), 
    
#     # 2. Add padding to make it exactly 640x640 (CenterCrop pads if image < size)
#     T.CenterCrop(640), 
    
#     # 3. Standard normalization/conversion
#     T.ToTensor(),
# ])


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


    # PIN MEMORY
    frame_t = frame_t.pin_memory()

    # async GPU transfer
    frame_t = frame_t.to("cuda", non_blocking=True)

    # HWC → CHW
    frame_t = frame_t.permute(2,0,1) #.float()/255.0

    frame_t = preprocess(frame_t)

    # add batch dimension
    frame_t = frame_t.unsqueeze(0)



    # inference
    results = model.track(frame_t, verbose=False, tracker="bytetrack.yaml",
                            persist=True, conf=0.5, iou=0.5)
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

cap.release()
cv2.destroyAllWindows()