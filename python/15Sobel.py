import cv2       #opencv读取的格式是BGR
import numpy as np
import CommonUtil

import matplotlib.pyplot as plt

imgPath = '../data/tliangtrans.jpg'
img=cv2.imread(imgPath)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)

CommonUtil.cv_show("sobelx",sobelx)

sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobely = cv2.convertScaleAbs(sobely)
CommonUtil.cv_show("sobely",sobely)

sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
CommonUtil.cv_show("sobelxy",sobelxy)

scharrx = cv2.Scharr(img,cv2.CV_64F,1,0)
scharry = cv2.Scharr(img,cv2.CV_64F,0,1)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
scharrxy =  cv2.addWeighted(scharrx,0.5,scharry,0.5,0)
CommonUtil.cv_show("scharrxy",scharrxy)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

res = np.hstack((sobelxy,scharrxy,laplacian))
CommonUtil.cv_show("scharrxy laplacian ",res)



