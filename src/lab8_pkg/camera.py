import cv2
import time

pipeline = "v4l2src device=/dev/video4 extra-controls=\"c,exposure_auto=3\" ! video/x-raw, width=960, height=540 ! videoconvert ! video/x-raw,format=BGR ! appsink"
cap = cv2.VideoCapture(pipeline)

# cap = cv2.VideoCapture("/dev/video4", cv2.CAP_V4L2)

time_old = time.time()
if cap.isOpened():
    cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
    while True:
        ret_val, img = cap.read()
        print(img.shape)
        cv2.imshow("Camera", img)
        cv2.waitKey(1)

        time_new = time.time()
        print("FPS: ", 1/(time_new-time_old))
        time_old = time_new
else:
    print("Unable to open camera")

cv2.destroyAllWindows()
cap.release()