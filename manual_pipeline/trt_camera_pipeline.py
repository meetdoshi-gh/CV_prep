import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

# -----------------------------
# CONFIG
# -----------------------------

ENGINE_PATH = "yolov8n.engine"
INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# -----------------------------
# LOAD ENGINE
# -----------------------------

with open(ENGINE_PATH, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

input_name = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)

input_shape = engine.get_tensor_shape(input_name)
output_shape = engine.get_tensor_shape(output_name)

input_size = trt.volume(input_shape)
output_size = trt.volume(output_shape)

# GPU buffers
d_input = cuda.mem_alloc(input_size * np.float32().nbytes)
d_output = cuda.mem_alloc(output_size * np.float32().nbytes)

stream = cuda.Stream()

# host buffers
h_output = np.empty(output_shape, dtype=np.float32)

context.set_tensor_address(input_name, int(d_input))
context.set_tensor_address(output_name, int(d_output))

# -----------------------------
# PREPROCESS
# -----------------------------

def preprocess(frame):

    img = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = img.astype(np.float32) / 255.0

    img = np.transpose(img, (2,0,1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    return np.ascontiguousarray(img)

# -----------------------------
# IOU
# -----------------------------

def iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0,x2-x1) * max(0,y2-y1)

    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

    union = area1 + area2 - inter

    return inter / union if union > 0 else 0

# -----------------------------
# NMS
# -----------------------------

def nms(boxes, scores, thresh):

    idxs = np.argsort(scores)[::-1]
    keep = []

    while len(idxs) > 0:

        i = idxs[0]
        keep.append(i)

        rem = []

        for j in idxs[1:]:

            if iou(boxes[i], boxes[j]) < thresh:
                rem.append(j)

        idxs = np.array(rem)

    return keep

# -----------------------------
# YOLO DECODE
# -----------------------------

def decode_yolo(output):

    output = output.reshape(84, -1).T

    boxes = []
    scores = []
    classes = []

    for det in output:

        x,y,w,h = det[:4]
        obj = det[4]
        cls_scores = det[5:]

        cls_id = np.argmax(cls_scores)
        score = obj * cls_scores[cls_id]

        if score < CONF_THRESH:
            continue

        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2

        boxes.append([x1,y1,x2,y2])
        scores.append(score)
        classes.append(cls_id)

    return boxes, scores, classes

# -----------------------------
# CAMERA LOOP
# -----------------------------

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    inp = preprocess(frame)

    cuda.memcpy_htod_async(d_input, inp, stream)

    context.execute_async_v3(stream_handle=stream.handle)

    cuda.memcpy_dtoh_async(h_output, d_output, stream)

    stream.synchronize()

    boxes, scores, classes = decode_yolo(h_output)

    keep = nms(boxes, scores, IOU_THRESH)

    for i in keep:

        x1,y1,x2,y2 = boxes[i]

        cv2.rectangle(frame,
                      (int(x1),int(y1)),
                      (int(x2),int(y2)),
                      (0,255,0),
                      2)

        label = f"{classes[i]} {scores[i]:.2f}"

        cv2.putText(frame,label,
                    (int(x1),int(y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(0,255,0),1)

    cv2.imshow("TensorRT YOLO", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()