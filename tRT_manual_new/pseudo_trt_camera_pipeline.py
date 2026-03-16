"""

START PROGRAM


------------------------------------------------
STEP 1 — Import libraries
------------------------------------------------

Import TensorRT library
Import CUDA driver (PyCUDA)
Automatically initialize CUDA context
Import NumPy for array handling
Import OpenCV for camera capture and drawing


------------------------------------------------
STEP 2 — Define configuration parameters
------------------------------------------------

ENGINE_PATH = location of serialized TensorRT engine file

INPUT_W = width expected by model
INPUT_H = height expected by model

CONF_THRESH = minimum detection confidence
IOU_THRESH = NMS overlap threshold


------------------------------------------------
STEP 3 — Create TensorRT logger
------------------------------------------------

Create logger object for TensorRT
This logger prints warnings or errors during runtime


------------------------------------------------
STEP 4 — Load serialized TensorRT engine
------------------------------------------------

Open engine file in binary mode

Create TensorRT runtime object

Deserialize engine
Meaning:
    convert saved engine binary
    into executable TensorRT engine object

Create execution context from engine
Execution context stores runtime state needed
for running inference


------------------------------------------------
STEP 5 — Discover input and output tensors
------------------------------------------------

Get input tensor name from engine
Get output tensor name from engine

Query input tensor shape
Example: (1,3,640,640)

Query output tensor shape
Example: (1,84,8400)


------------------------------------------------
STEP 6 — Compute number of elements
------------------------------------------------

Compute total elements in input tensor
Compute total elements in output tensor

TensorRT provides helper function "volume"
which multiplies all dimensions together


------------------------------------------------
STEP 7 — Allocate GPU memory (device buffers)
------------------------------------------------

Allocate GPU memory for model input

d_input =
    GPU buffer large enough to hold
    input_size * size_of_float32 bytes

Allocate GPU memory for model output

d_output =
    GPU buffer large enough to hold
    output_size * size_of_float32 bytes


------------------------------------------------
STEP 8 — Create CUDA stream
------------------------------------------------

Create CUDA stream object

A CUDA stream acts as a queue of GPU tasks

Tasks we will enqueue:
    copy input to GPU
    run inference
    copy output back to CPU


------------------------------------------------
STEP 9 — Allocate host output buffer
------------------------------------------------

Create NumPy array with output shape

This lives in CPU memory

It will receive inference results
copied from GPU


------------------------------------------------
STEP 10 — Bind tensors to GPU memory
------------------------------------------------

Tell TensorRT:

Input tensor should read from memory address of d_input

Output tensor should write to memory address of d_output


------------------------------------------------
STEP 11 — Define preprocessing function
------------------------------------------------

Function preprocess(frame):

    Resize frame to model input size

    Convert pixel values to float32

    Normalize pixels to range [0,1]

    Convert image layout
        from HWC (OpenCV format)
        to CHW (deep learning format)

    Add batch dimension

    Return contiguous NumPy array


------------------------------------------------
STEP 12 — Define IoU function
------------------------------------------------

Function iou(box1, box2):

    Compute intersection rectangle

    Compute intersection area

    Compute individual box areas

    Compute union area

    IoU = intersection / union

    Return IoU value


------------------------------------------------
STEP 13 — Define NMS function
------------------------------------------------

Function nms(boxes, scores, threshold):

    Sort detection indices by score (descending)

    Create empty list "keep"

    While there are still boxes left:

        Pick highest scoring box

        Add it to keep list

        Compare it with remaining boxes

        Remove boxes whose IoU exceeds threshold

    Return indices of kept boxes


------------------------------------------------
STEP 14 — Define YOLO decode
------------------------------------------------

Function decode_yolo(output):

    Reshape raw output tensor

    Convert it into list of candidate detections

    For each candidate detection:

        extract box center coordinates
        extract width and height
        extract objectness score
        extract class probabilities

        find class with highest probability

        compute final confidence

        if confidence < threshold:
            skip detection

        convert box format
        from center format
        to corner format

        store box, score, class

    return all candidate detections


------------------------------------------------
STEP 15 — Open camera
------------------------------------------------

Open camera device using OpenCV


------------------------------------------------
STEP 16 — Main inference loop
------------------------------------------------

While camera is running:

    Capture frame

    If capture failed:
        exit loop


    -------------------------
    Preprocess
    -------------------------

    Convert frame to model input tensor


    -------------------------
    Copy input to GPU
    -------------------------

    Asynchronously copy CPU input tensor
    to GPU memory d_input
    using CUDA stream


    -------------------------
    Run TensorRT inference
    -------------------------

    Launch inference kernels
    using execution context


    -------------------------
    Copy result back to CPU
    -------------------------

    Asynchronously copy GPU output
    from d_output
    to CPU array h_output


    -------------------------
    Synchronize GPU
    -------------------------

    Wait until all operations in stream finish


    -------------------------
    Post-processing
    -------------------------

    Decode YOLO detections

    Apply NMS to remove duplicates


    -------------------------
    Draw detections
    -------------------------

    For each detection kept after NMS:

        draw rectangle

        draw label text


    -------------------------
    Display result
    -------------------------

    Show frame in window

    If ESC key pressed:
        exit loop


------------------------------------------------
STEP 17 — Cleanup
------------------------------------------------

Release camera

Destroy OpenCV windows


END PROGRAM

"""