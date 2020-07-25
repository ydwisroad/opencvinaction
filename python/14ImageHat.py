import cv2       #opencv读取的格式是BGR
import numpy as np
import CommonUtil

import matplotlib.pyplot as plt

imgPath = '../data/tliangtrans.jpg'
img=cv2.imread(imgPath)

kernel = np.ones((3,3),np.uint8)

#礼帽
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
CommonUtil.cv_show("tophat",tophat)

#黑帽
blackhat  = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT, kernel)
CommonUtil.cv_show("blackhat",blackhat)