import tensorrt as trt

TRT_LOGGER = trt.Logger()

engine_path = "yolov8n.engine"

with open(engine_path, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

print("Number of tensors:", engine.num_io_tensors)

for i in range(engine.num_io_tensors):

    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    dtype = engine.get_tensor_dtype(name)
    mode = engine.get_tensor_mode(name)

    print("----")
    print("Name:", name)
    print("Shape:", shape)
    print("Dtype:", dtype)
    print("Mode:", mode)