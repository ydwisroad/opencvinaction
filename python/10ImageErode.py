import cv2       #opencv读取的格式是BGR
import numpy as np
import CommonUtil

import matplotlib.pyplot as plt

imgPath = '../data/tliangtrans.jpg'
img=cv2.imread(imgPath)

kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)

CommonUtil.cv_show("erosion", erosion)

kernel = np.ones((30,30),np.uint8)
erosion_1 = cv2.erode(img,kernel,iterations = 1)
erosion_2 = cv2.erode(img,kernel,iterations = 2)
erosion_3 = cv2.erode(img,kernel,iterations = 3)
res = np.hstack((erosion_1,erosion_2,erosion_3))
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
