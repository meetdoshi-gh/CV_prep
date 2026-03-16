import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

onnx_path = "yolov8n.onnx"
engine_path = "yolov8n.engine"

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)

parser = trt.OnnxParser(network, TRT_LOGGER)

# read ONNX
with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

# enable FP16 if GPU supports it
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

engine = builder.build_engine(network, config)

# serialize engine
with open(engine_path, "wb") as f:
    f.write(engine.serialize())

print("TensorRT engine built successfully.")


"""

trtexec --onnx=yolov8n.onnx \
        --saveEngine=yolov8n.engine \
        --fp16

"""


# import os

# onnx = "yolov8n.onnx"
# engine = "yolov8n.engine"

# cmd = f"""
# trtexec
# --onnx={onnx}
# --saveEngine={engine}
# --fp16
# --workspace=4096
# """

# os.system(cmd)