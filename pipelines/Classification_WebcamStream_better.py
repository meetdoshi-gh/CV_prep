# import cv2
# import torch
# import time
# import torchvision.transforms as transforms
# from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# # device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # load model
# weights = MobileNet_V2_Weights.DEFAULT
# model = mobilenet_v2(weights=weights).to(device)
# model.eval()

# # get preprocessing steps identical to training
# preprocessing = weights.transforms()
# transformation = transforms.Compose([
#     transforms.ToPILImage(),
#     preprocessing])


# # labels
# categories = weights.meta["categories"]

# cap = cv2.VideoCapture(0)

# prev_time = time.time()

# while 1:

#     ret, frame = cap.read()
#     if not ret:
#         break

#     start_infer = time.time()

#     # BGR -> RGB
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # PIL conversion handled internally by transforms
#     input_tensor = transformation(rgb).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = model(input_tensor)

#     probs = torch.nn.functional.softmax(outputs[0], dim=0)
#     top3 = torch.topk(probs, 3)

#     infer_time = (time.time() - start_infer) * 1000

#     # ---------- Draw Predictions ----------
#     for i in range(3):

#         class_id = top3.indices[i].item()
#         score = top3.values[i].item()
#         label = categories[class_id]

#         text = f"{label}: {score:.2f}"

#         y = 40 + i * 40

#         cv2.putText(frame, text, (20, y),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7, (0,255,0), 2)

#         # confidence bar
#         bar_width = int(score * 200)
#         cv2.rectangle(frame,
#                       (250, y-15),
#                       (250 + bar_width, y),
#                       (0,255,0), -1)

#     # ---------- FPS ----------
#     curr_time = time.time()
#     fps = 1 / (curr_time - prev_time)
#     prev_time = curr_time

#     cv2.putText(frame,
#                 f"FPS: {fps:.1f}",
#                 (20, 400),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (0,255,255),
#                 2)

#     cv2.putText(frame,
#                 f"Inference: {infer_time:.1f} ms",
#                 (20, 430),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (0,255,255),
#                 2)

#     cv2.imshow("Classification Demo", frame)

#     if cv2.waitKey(1) == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()









import cv2
import torch
import time
import numpy as np
from collections import deque
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

##################################################
# Load device & Model
##################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights).to(device)
model.eval()

# # Get transformations identical to that of training data
transforms_iid = weights.transforms()

# Load labels 
categories = weights.meta["categories"]

# For Batching & MVA of predictions
batch_size = 4
history_size = 10
frame_buffer = deque(maxlen=batch_size)
history = deque(maxlen=history_size)
fps_tally = deque(maxlen=100)

##################################################
# Capture webcam video stream
##################################################
cap = cv2.VideoCapture(0)

prev_time = time.time()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # # Preprocessing
    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    # This is happening on CPU, as OpenCV returns numpy array 
    # which is on CPU, then conversion to PIL is also a CPU 
    # based process. Also here each frame has tro be processed
    # one by one, as PIL conversion doesn't allow batch
    pil = Image.fromarray(rgb)
    tensor_input = transforms_iid(pil)
    frame_buffer.append(tensor_input)
    if len(frame_buffer) < batch_size:
        print(len(frame_buffer))
        print("frame appended!")
        continue

    # # Preprocessing leveraging GPU
    # frame_buffer.append(rgb)
    # if len(frame_buffer) < batch_size:
    #     print(len(frame_buffer))
    #     print("frame appended!")
    #     continue

    # tensor_input = torch.from_numpy(np.array(frame_buffer)).to(device) #<~~~
    # # HWC -> CHW & [0, 255] -> [0, 1]
    # tensor_input = tensor_input.permute(0, 3, 1, 2).float() / 255.0
    # # # add dimension of batch if not already batched
    # # tensor_input = tensor_input.unsqueeze(0)
    # # Resize 
    # tensor_input = torch.nn.functional.interpolate(
    #     tensor_input,
    #     size=(224, 224),
    #     mode="bilinear",
    #     align_corners=False
    # )
    # # Normalize
    # mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    # std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    # tensor_input = (tensor_input - mean) / std

    # only if using CPU based preprocessing 
    batch = torch.stack(list(frame_buffer)).to(device)
    # # else if using GPU based preprocessing
    # batch = tensor_input

    start = time.time()

    with torch.no_grad():
        outputs = model(batch)

    infer_time = (time.time() - start) * 1000

    probs = torch.nn.functional.softmax(outputs, dim=1)

    # use last frame prediction
    last_probs = probs[-1].cpu().numpy()

    history.append(last_probs)

    avg_probs = np.mean(history, axis=0)

    top3 = np.argsort(avg_probs)[-3:][::-1]

    for i, idx in enumerate(top3):

        score = avg_probs[idx]
        label = categories[idx]

        y = 40 + i * 40

        cv2.putText(frame,
                    f"{label}: {score:.2f}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    1)

        bar_width = int(score * 200)

        cv2.rectangle(frame,
                        (250, y-15),
                        (250 + bar_width, y),
                        (0,255,0),
                        -1)

    curr = time.time()
    fps = 1 / (curr - prev_time)
    fps_tally.append(fps)
    prev_time = curr

    cv2.putText(frame,
                f"FPS: {fps:.1f}",
                (20, 400),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2)

    cv2.putText(frame,
                f"Inference: {infer_time:.1f} ms",
                (20, 430),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2)

    cv2.imshow("Classification Demo", frame)

    if cv2.waitKey(1) == 27:
        break

print(f"Average FPS of last 100 frames = {np.mean(np.array(fps_tally))} ")
cap.release()
cv2.destroyAllWindows()