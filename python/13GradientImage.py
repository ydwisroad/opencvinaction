import cv2       #opencv读取的格式是BGR
import numpy as np
import CommonUtil

import matplotlib.pyplot as plt

imgPath = '../data/tliangtrans.jpg'
img=cv2.imread(imgPath)

# 梯度=膨胀-腐蚀
kernel = np.ones((7,7),np.uint8)
dilate = cv2.dilate(img,kernel,iterations = 5)
erosion = cv2.erode(img,kernel,iterations = 5)

res = np.hstack((dilate,erosion))

CommonUtil.cv_show("res",res)

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

CommonUtil.cv_show("gradient",gradient)




