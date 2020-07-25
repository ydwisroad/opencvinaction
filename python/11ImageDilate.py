import cv2       #opencv读取的格式是BGR
import numpy as np
import CommonUtil

import matplotlib.pyplot as plt

imgPath = '../data/tliangtrans.jpg'
img=cv2.imread(imgPath)

kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)

CommonUtil.cv_show("erosion", erosion)

kernel = np.ones((3,3),np.uint8)
dilate = cv2.dilate(erosion,kernel,iterations = 1)

CommonUtil.cv_show("dilate", dilate)

kernel = np.ones((30,30),np.uint8)
dilate_1 = cv2.dilate(img,kernel,iterations = 1)
dilate_2 = cv2.dilate(img,kernel,iterations = 2)
dilate_3 = cv2.dilate(img,kernel,iterations = 3)
res = np.hstack((dilate_1,dilate_2,dilate_3))

CommonUtil.cv_show("dilate3", res)




