import tensorrt as trt

model_path = "f1tenth_model.onnx"

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flag)
parser = trt.OnnxParser(network, logger)

# parse onnx model
success = parser.parse_from_file(model_path)
for idx in range(parser.num_errors):
    print(parser.get_error(idx))

# build serialized engine
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)

serialized_engine = builder.build_serialized_network(network, config)

with open("f1tenth_model_fp16.engine", "wb") as f:
    f.write(serialized_engine)