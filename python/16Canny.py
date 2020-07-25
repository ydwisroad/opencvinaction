import cv2       #opencv读取的格式是BGR
import numpy as np
import CommonUtil

import matplotlib.pyplot as plt

imgPath = '../data/tliangtrans.jpg'
img=cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)

#1) 使用高斯滤波器，以平滑图像，滤除噪声。
#2) 计算图像中每个像素点的梯度强度和方向。
#3) 应用非极大值（Non-Maximum Suppression）抑制，以消除边缘检测带来的杂散响应。
#4) 应用双阈值（Double-Threshold）检测来确定真实的和潜在的边缘。
#5) 通过抑制孤立的弱边缘最终完成边缘检测。

v1=cv2.Canny(img,80,150)
v2=cv2.Canny(img,50,100)

res = np.hstack((v1,v2))
CommonUtil.cv_show("canny1",res)

v1=cv2.Canny(img,120,250)
v2=cv2.Canny(img,50,100)
res = np.hstack((v1,v2))
CommonUtil.cv_show("canny2",res)
