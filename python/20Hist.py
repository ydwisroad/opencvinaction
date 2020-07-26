import cv2       #opencv读取的格式是BGR
import numpy as np
import CommonUtil

import matplotlib.pyplot as plt

imgPath = '../data/tliangtrans.jpg'
img=cv2.imread(imgPath)

hist = cv2.calcHist([img],[0],None,[256],[0,256])
print("hist Shape", hist.shape)

plt.hist(img.ravel(),256);
plt.show()

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

# 创建mast
mask = np.zeros(img.shape[:2], np.uint8)
print(mask.shape)
mask[100:300, 100:400] = 255
CommonUtil.cv_show('mask',mask)

masked_img = cv2.bitwise_and(img, img, mask=mask)#与操作
CommonUtil.cv_show('masked_img',masked_img)

hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0, 256])
plt.show()

plt.hist(img.ravel(),256);
plt.show()

img=cv2.imread(imgPath, 0)
equ = cv2.equalizeHist(img)
plt.hist(equ.ravel(),256)
plt.show()

res = np.hstack((img,equ))
CommonUtil.cv_show('res', res)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
res_clahe = clahe.apply(img)
res = np.hstack((img,equ,res_clahe))
CommonUtil.cv_show('res', res)







