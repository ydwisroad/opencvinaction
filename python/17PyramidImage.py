import cv2       #opencv读取的格式是BGR
import numpy as np
import CommonUtil

import matplotlib.pyplot as plt

imgPath = '../data/tliangtrans.jpg'
img=cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)

up = cv2.pyrUp(img)
CommonUtil.cv_show('up', up)
print("up shape", up.shape)

down=cv2.pyrDown(img)
CommonUtil.cv_show('down', down)

up2=cv2.pyrUp(up)
CommonUtil.cv_show('up2', up2)

up=cv2.pyrUp(img)
up_down=cv2.pyrDown(up)
CommonUtil.cv_show('up_down', up_down)

CommonUtil.cv_show('up_down',np.hstack((img,up_down)))

up=cv2.pyrUp(img)
up_down=cv2.pyrDown(up)
CommonUtil.cv_show('img-up_down',img-up_down)

down=cv2.pyrDown(img)
down_up=cv2.pyrUp(down)
l_1=img-down_up
CommonUtil.cv_show('l_1',l_1)

