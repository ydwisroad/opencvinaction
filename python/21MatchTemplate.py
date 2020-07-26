import cv2       #opencv读取的格式是BGR
import numpy as np
import CommonUtil

import matplotlib.pyplot as plt

imgPath = '../data/tliangtrans.jpg'
templatePath = '../data/tliangonly.jpg'

img=cv2.imread(imgPath, 0)

template = cv2.imread(templatePath, 0)
h, w = template.shape[:2]

print("image shape ", img.shape)
print("template shape ", template.shape)

#TM_SQDIFF：计算平方不同，计算出来的值越小，越相关
#TM_CCORR：计算相关性，计算出来的值越大，越相关
#TM_CCOEFF：计算相关系数，计算出来的值越大，越相关
#TM_SQDIFF_NORMED：计算归一化平方不同，计算出来的值越接近0，越相关
#TM_CCORR_NORMED：计算归一化相关性，计算出来的值越接近1，越相关
#TM_CCOEFF_NORMED：计算归一化相关系数，计算出来的值越接近1，越相关

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
print("match result shape" , res.shape)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

for meth in methods:
    img2 = img.copy()

    # 匹配方法的真值
    method = eval(meth)
    print (method)
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 如果是平方差匹配TM_SQDIFF或归一化平方差匹配TM_SQDIFF_NORMED，取最小值
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 画矩形
    cv2.rectangle(img2, top_left, bottom_right, 255, 2)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
    plt.subplot(122), plt.imshow(img2, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()

img_rgb = cv2.imread(imgPath,)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread(templatePath, 0)
h, w = template.shape[:2]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
# 取匹配程度大于%80的坐标
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):  # *号表示可选参数
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)

CommonUtil.cv_show("img_rgb", img_rgb)



