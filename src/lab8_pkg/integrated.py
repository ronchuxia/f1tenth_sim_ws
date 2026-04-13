import cv2
from lane import detect_lane_markers
import tensorrt as trt
import numpy as np
from cuda import cudart
from utils import label_to_box_xyxy, voting_suppression

K = np.load("K.npy")
f_y, y_0 = K[1, 1], K[1, 2]
H_mount = 0.1379
USE_IMAGE = False

# initialize TensorRT engine and context
batch_size = 1
engine_path = "f1tenth_model_fp16.engine"
img_path = "resource/test_car_x60cm.png"

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

# intialize camera
pipeline = "v4l2src device=/dev/video4 extra-controls=\"c,exposure_auto=3\" ! video/x-raw, width=960, height=540 ! videoconvert ! video/x-raw,format=BGR !appsink"
cap = None if USE_IMAGE else cv2.VideoCapture(pipeline)

if USE_IMAGE or cap.isOpened():
    while True:
        # read image
        if USE_IMAGE:
            img = cv2.imread(img_path)
        else:
            ret_val, img = cap.read()

        # lane detection
        lane_img = detect_lane_markers(img)

        # object detection
        # preprocess input data
        input_img = cv2.resize(img, (320, 180)) / 255.0  # shape (180, 320, 3), normalize to [0, 1]
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = np.expand_dims(input_img, axis=0)   # shape (1, 3, 180, 320)
        input_img = input_img.astype(np.float32)    # convert from float64 to float
        input_img = np.ascontiguousarray(input_img) # ensure data is contiguous

        # copy input data to device
        err, = cudart.cudaMemcpyAsync(
            device_input_ptr,
            input_img.ctypes.data,
            input_img.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream
        )
        check(err)

        # execute model
        context.execute_async_v3(stream)

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

        # post process
        voting_iou_threshold = 0.5
        confi_threshold = 0.4
        bboxs, result_prob = label_to_box_xyxy(output[0], confi_threshold)
        vote_rank = voting_suppression(bboxs, voting_iou_threshold)
        if len(vote_rank) > 0:
            x1, y1, x2, y2 = bboxs[vote_rank[0]]
            scale_x = img.shape[1] / 320
            scale_y = img.shape[0] / 180
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            cv2.rectangle(lane_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            dist = -f_y * H_mount / (y_0 - y2)
            print(f"Distance: {dist:.2f} m")
        else:
            print("No object detected")

        cv2.imshow("Integrated", lane_img)
        cv2.waitKey(1)
else:
    print("Unable to open camera")

cv2.destroyAllWindows()
if cap is not None:
    cap.release()

# release resources
cudart.cudaStreamDestroy(stream)
cudart.cudaFree(device_input_ptr)
cudart.cudaFree(device_output_ptr)
