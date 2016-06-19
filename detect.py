import cv2
import math

import numpy as np

cap = cv2.VideoCapture('driving.mov')

fourcc = cv2.cv.CV_FOURCC('a', 'v', 'c', '1')
out = cv2.VideoWriter('output.mp4', fourcc, 29, (1920, 1080))

while cap.isOpened():
    ret, frame = cap.read()

    height, width, channels = frame.shape
    halfHeight = height / 2
    halfWidth = width / 2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi = gray[halfHeight:height, 0:width]
    edges = cv2.Canny(roi, 50, 450, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=30)

    for x1, y1, x2, y2 in lines[0]:
        angle = math.fabs(math.atan2((y2 - y1), (x2 - x1)) * 180 / np.pi)
        if angle > 10:
            cv2.line(frame, (x1, y1 + halfHeight), (x2, y2 + halfHeight), (0, 255, 255, .5), 2)

    cv2.imshow('window-name', frame)
    out.write(frame)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
