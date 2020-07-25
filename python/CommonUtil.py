import cv2       #opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cv_save(path,img):
    cv2.imwrite(path, img)





