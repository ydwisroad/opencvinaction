import cv2       #opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np

videoPath = '../data/tliangsetvideo.mp4'
vc = cv2.VideoCapture(videoPath)

open = False
# 检查是否打开正确
if vc.isOpened():
    open, frame = vc.read()
else:
    open = False

while open:
    ret, frame = vc.read()
    if frame is None:
        break
    if ret == True:
        gray = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)
        cv2.imshow('result', gray)
        if cv2.waitKey(100) & 0xFF == 27:
            break

vc.release()
cv2.destroyAllWindows()