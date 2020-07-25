import cv2       #opencv读取的格式是BGR
import numpy as np
import CommonUtil

import matplotlib.pyplot as plt

imgPath = '../data/tliangtrans.jpg'
img=cv2.imread(imgPath)

# 均值滤波
# 简单的平均卷积操作
blur = cv2.blur(img, (3, 3))

# 方框滤波
# 基本和均值一样，可以选择归一化
box1 = cv2.boxFilter(img,-1,(3,3), normalize=True)

# 方框滤波
# 基本和均值一样，可以选择归一化,容易越界
box2 = cv2.boxFilter(img,-1,(3,3), normalize=False)

# 高斯滤波
# 高斯模糊的卷积核里的数值是满足高斯分布，相当于更重视中间的
aussian = cv2.GaussianBlur(img, (5, 5), 1)

# 中值滤波
# 相当于用中值代替
median = cv2.medianBlur(img, 5)  # 中值滤波

titles = ['Original Image', 'Mean', 'Box normalize', 'Box FalseNorm', 'Gaussian', 'Median']
images = [img, blur, box1, box2, aussian, median]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()