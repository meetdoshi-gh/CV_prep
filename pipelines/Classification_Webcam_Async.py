import cv2
import torch
import numpy as np
import time
import threading

from collections import deque
from PIL import Image

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


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
# Load Device & Model
##################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights).to(device)
model.eval()

# This is easy but CPU based
# Training-identical preprocessing
transforms_iid = weights.transforms()

# Load ImageNet labels
categories = weights.meta["categories"]


##################################################
# Buffers
##################################################

batch_size = 4
history_size = 10

frame_buffer = deque(maxlen=batch_size)
history = deque(maxlen=history_size)
fps_tally = deque(maxlen=100)


##################################################
# Start Camera
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

    ##################################################
    # Preprocessing
    ##################################################

    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # # This is happening on CPU, as OpenCV returns numpy array 
    # # which is on CPU, then conversion to PIL is also a CPU 
    # # based process. Also here each frame has tro be processed
    # # one by one, as PIL conversion doesn't allow batch
    # pil = Image.fromarray(rgb)

    # tensor_input = transforms_iid(pil)

    # frame_buffer.append(tensor_input)

    # if len(frame_buffer) < batch_size:
    #     print(len(frame_buffer))
    #     print("frame appended!")
    #     continue

    # Preprocessing leveraging GPU

    frame_buffer.append(rgb)

    if len(frame_buffer) < batch_size:
        print(len(frame_buffer))
        print("frame appended!")
        continue

    tensor_input = torch.from_numpy(np.array(frame_buffer)).to(device, non_blocking=True) #<~~~

    # HWC -> CHW & [0, 255] -> [0, 1]
    tensor_input = tensor_input.permute(0, 3, 1, 2).float() / 255.0

    # Resize 
    tensor_input = torch.nn.functional.interpolate(
        tensor_input,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    tensor_input = (tensor_input - mean) / std

    # # if this:
    # # only if using CPU based preprocessing 
    # batch = torch.stack(list(frame_buffer)).to(device, non_blocking=True)

    # else this:
    # if using GPU based preprocessing
    batch = tensor_input


    ##################################################
    # Batch Inference
    ##################################################

    start = time.time()

    with torch.no_grad():
        outputs = model(batch)

    infer_time = (time.time() - start) * 1000


    ##################################################
    # Probabilities
    ##################################################

    probs = torch.nn.functional.softmax(outputs, dim=1)

    # Use newest frame prediction
    last_probs = probs[-1].cpu().numpy()

    history.append(last_probs)

    avg_probs = np.mean(history, axis=0)


    ##################################################
    # Top-3 Predictions
    ##################################################

    top3 = np.argsort(avg_probs)[-3:][::-1]

    for i, idx in enumerate(top3):

        score = avg_probs[idx]
        label = categories[idx]

        y = 40 + i * 40

        cv2.putText(
            frame,
            f"{label}: {score:.2f}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            1
        )

        bar_width = int(score * frame.shape[1] * 0.4)

        cv2.rectangle(
            frame,
            (250, y-15),
            (250 + bar_width, y),
            (0,255,0),
            -1
        )


    ##################################################
    # FPS Calculation
    ##################################################

    curr = time.time()

    fps = 1 / (curr - prev_time)

    fps_tally.append(fps)

    prev_time = curr


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 400),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Inference: {infer_time:.1f} ms",
        (20, 430),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )


    ##################################################
    # Display
    ##################################################

    cv2.imshow("Classification Demo", frame)

    if cv2.waitKey(1) == 27:
        break


##################################################
# Cleanup
##################################################

print(f"Average FPS (last 100 frames): {np.mean(np.array(fps_tally)):.2f}")

cap.release()
cv2.destroyAllWindows()