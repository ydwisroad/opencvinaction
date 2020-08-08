import cv2
import matplotlib.pyplot as plt
import numpy as np


imgPath = '../data/tliangtrans.jpg'
image=cv2.imread(imgPath,cv2.IMREAD_COLOR)

cv2.imwrite('./tliangnewimage.jpg',image,[cv2.IMWRITE_JPEG_QUALITY, 50])
b,g,r = cv2.split(image)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
b = clahe.apply(b)
g = clahe.apply(g)
r = clahe.apply(r)
image = cv2.merge([b,g,r])

cv2.imwrite('./tliangclahe.jpg',image,[cv2.IMWRITE_JPEG_QUALITY, 50])