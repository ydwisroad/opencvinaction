import cv2       #opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np

import CommonUtil

imgPath = '../data/tliangtrans.jpg'
img=cv2.imread(imgPath)

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print("shape", img_gray.shape)

CommonUtil.cv_show("gray image ",img_gray)

hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
CommonUtil.cv_show("gray image ",hsv)



