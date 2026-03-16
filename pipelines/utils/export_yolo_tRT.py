from ultralytics import YOLO

selected_model = "yolo26n.pt"
model = YOLO(selected_model)

model.export(
    imgsz=640,
    format="engine",
    int8=True,
    data="coco8.yaml"
)

"""
You’ll demonstrate:

PyTorch → ONNX conversion

TensorRT engine building

manual preprocessing

GPU inference

real-time pipeline


1️⃣ export_to_onnx.py
2️⃣ build_engine.py
3️⃣ trt_camera_pipeline.py
"""