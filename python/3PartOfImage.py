import cv2       #opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np

import CommonUtil

imgPath = '../data/tliangtrans.jpg'

img=cv2.imread(imgPath)
cutTBeam=img[0:400,0:300]
CommonUtil.cv_show('tBeam',cutTBeam)

b,g,r=cv2.split(img)

img=cv2.merge((b,g,r))
print("merged shape", img.shape)

# 只保留R
cur_img = img.copy()
cur_img[:,:,0] = 0
cur_img[:,:,1] = 0
CommonUtil.cv_show('R',cur_img)

# 只保留G
cur_img = img.copy()
cur_img[:,:,0] = 0
cur_img[:,:,2] = 0
CommonUtil.cv_show('G',cur_img)

# 只保留B
cur_img = img.copy()
cur_img[:,:,1] = 0
cur_img[:,:,2] = 0
CommonUtil.cv_show('B',cur_img)