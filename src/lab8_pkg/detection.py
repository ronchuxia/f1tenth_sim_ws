import tensorrt as trt
import numpy as np
from cuda import cudart
import cv2
import time
from utils import DisplayLabel, label_to_box_xyxy, bbox_convert_r, voting_suppression


batch_size = 1
engine_path = "f1tenth_model_fp16.engine"
image_path = "resource/0.jpg"
image_path = "resource/test_car_x60cm.png"

def check(err):
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

with open(engine_path, "rb") as f:
    model_data = f.read()

# deserialize engine
engine = runtime.deserialize_cuda_engine(model_data)

# create execution context
context = engine.create_execution_context()

# allocate device memory for inputs and outputs
for binding in engine:
    shape = engine.get_tensor_shape(binding)
    np_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(binding)))
    mode = engine.get_tensor_mode(binding)
    # print("binding: ", binding, "mode: ", mode, "shape: ", shape, "dtype: ", np_dtype)

    size = trt.volume(shape) * batch_size * np_dtype.itemsize
    err, ptr = cudart.cudaMalloc(size)
    check(err)

    if mode == trt.TensorIOMode.INPUT:
        device_input_ptr = ptr
    else:
        device_output_ptr = ptr

    context.set_tensor_address(binding, ptr)

# create cuda stream
err, stream = cudart.cudaStreamCreate()
check(err)

# preprocess input data
img = cv2.imread(image_path) / 255.0    # shape (360, 640, 3)
img = cv2.resize(img, (320, 180))   # shape (180, 320, 3)
img = np.transpose(img, (2, 0, 1))   # shape (3, 180, 320)
img = np.expand_dims(img, axis=0)   # shape (1, 3, 180, 320)
img = img.astype(np.float32)    # convert from float64 to float32
img = np.ascontiguousarray(img) # ensure data is contiguous in memory

# copy input data to device
err, = cudart.cudaMemcpyAsync(
    device_input_ptr, 
    img.ctypes.data, 
    img.nbytes, 
    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, 
    stream
)
check(err)

# execute model
t_start = time.perf_counter()
context.execute_async_v3(stream)
cudart.cudaStreamSynchronize(stream)
t_end = time.perf_counter()
print(f"Inference time: {(t_end - t_start) * 1000:.2f} ms")

# copy output data to host
output = np.empty(shape=(batch_size, 5, 5, 10), dtype=np.float32)
err, = cudart.cudaMemcpyAsync(
    output.ctypes.data, 
    device_output_ptr, 
    output.nbytes, 
    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, 
    stream
)
check(err)

# synchronize stream before accessing output data
cudart.cudaStreamSynchronize(stream)
# print("output: ", output)

# release resources
cudart.cudaStreamDestroy(stream)
cudart.cudaFree(device_input_ptr)
cudart.cudaFree(device_output_ptr)

# visualize
voting_iou_threshold = 0.5
confi_threshold = 0.4
bboxs, result_prob = label_to_box_xyxy(output[0], confi_threshold)
vote_rank = voting_suppression(bboxs, voting_iou_threshold)
bbox = bboxs[vote_rank[0]]
[c_x, c_y, w, h] = bbox_convert_r(bbox[0], bbox[1], bbox[2], bbox[3])
bboxs_2 = np.array([[c_x, c_y, w, h]])
DisplayLabel(np.transpose(img[0], (1, 2, 0)), bboxs_2)