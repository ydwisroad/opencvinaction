import cv2       #opencv读取的格式是BGR
import numpy as np
import CommonUtil

import matplotlib.pyplot as plt

imgPath = '../data/tliangtrans.jpg'
img=cv2.imread(imgPath)

res = cv2.resize(img, (0, 0), fx=4, fy=4)
CommonUtil.cv_show("resized_4_4",res)

res = cv2.resize(img, (0, 0), fx=1, fy=3)
CommonUtil.cv_show("resized_1_3",res)

