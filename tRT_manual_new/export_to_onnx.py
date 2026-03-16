from ultralytics import YOLO

# load model
model = YOLO("yolov8n.pt")   # or yolo26n.pt

# export to ONNX
model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    nms = True,
    max_det = 10,
    simplify=True
)

print("ONNX model exported.")