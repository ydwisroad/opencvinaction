import cv2       #opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np

import CommonUtil

imgPath = '../data/tliangtrans.jpg'
img=cv2.imread(imgPath)

img2= img +10
print(img2[:5,:,0])

print((img+ img2)[:5,:,0])
