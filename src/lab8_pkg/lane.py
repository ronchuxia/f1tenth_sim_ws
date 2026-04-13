import cv2
import numpy as np


def detect_lane_markers(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([18, 80, 80]), np.array([35, 255, 255]))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue
        cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)

    return result


if __name__ == "__main__":
    img = cv2.imread("resource/lane.png")
    cv2.imwrite("lane.png", detect_lane_markers(img))
