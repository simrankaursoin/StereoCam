import numpy as np
import cv2
cap0 = cv2.VideoCapture(1)
cap1 = cv2.VideoCapture(0)


while cap0.isOpened() and cap1.isOpened():
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    frame0 = cv2.flip(frame0, -1)
    frame1 = cv2.flip(frame1, -1)
    cv2.imshow("frame0", frame0)
    cv2.imshow("frame1", frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap1.release()
cap0.release()
cv2.destroyAllWindows()
