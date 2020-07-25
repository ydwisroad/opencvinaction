import cv2       #opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np

import CommonUtil

imgPath = '../data/tliangtrans.jpg'
img=cv2.imread(imgPath)

print("image read img:", img)

print("image shape:" , img.shape)

print("type of image:" , type(img))


CommonUtil.cv_show("image", img)

img=cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)
CommonUtil.cv_show("imageGray", img)

imgSaveGrayPath = '../data/tliangtransGray.jpg'
CommonUtil.cv_save(imgSaveGrayPath, img)





