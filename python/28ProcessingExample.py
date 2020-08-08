import cv2
import matplotlib.pyplot as plt
import numpy as np


imgPath = '../data/jialiang.png'
image=cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

cv2.imwrite('../data/1grey1.jpg',image,[cv2.IMWRITE_JPEG_QUALITY, 50])

image = cv2.GaussianBlur(image, (5, 5), 1)
cv2.imwrite('../data/2gaussBlur1.jpg',image,[cv2.IMWRITE_JPEG_QUALITY, 50])

kernel = np.array([[0, -1, 0], [0, 3, 0], [0, -1, 0]]) #定义卷积核
image = cv2.filter2D(image,-1, kernel) #进行卷积运算

cv2.imwrite('../data/3enhance1.jpg',image)





