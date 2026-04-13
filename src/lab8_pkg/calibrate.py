'''
Camera calibration with OpenCV.
'''

import cv2
import numpy as np
import glob

image_folder = "calibration"
BOARD_SIZE = (6, 8)
SQUARE_SIZE = 0.25  # m

objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

objpoints, imgpoints = [], []
for path in sorted(glob.glob(f"{image_folder}/*.png")):
    gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

ret, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=cv2.CALIB_FIX_K3)
print(f"RMS error: {ret:.4f}\nK:{K}\ndist_coeffs:{dist}")
np.save("K.npy", K)
np.save("dist.npy", dist)

