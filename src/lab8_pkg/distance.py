import cv2
import numpy as np

# Compute height of the camera
K = np.load("K.npy")
f_x = K[0, 0]
f_y = K[1, 1]
x_0 = K[0, 2]
y_0 = K[1, 2]

X_car = 0.4
y_pixel = 498

H_mount = - X_car * (y_0 - y_pixel) / f_y
print(f"H_mount = {H_mount:.4f} m")

# Compute distance to the cone
x_pixel_unknown = 598
y_pixel_unknown = 420

X_car = -f_y * H_mount / (y_0 - y_pixel_unknown)
Y_car = (x_pixel_unknown - x_0) * X_car / f_x
print(f"X_car = {X_car:.4f} m, Y_car = {Y_car:.4f} m")
