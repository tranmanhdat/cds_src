import config as cf
import cv2
import numpy as np

cf.img_gray = cv2.imread('../output/1__8_0.jpg')

cf.img_gray = cv2.cvtColor(cf.img_gray, cv2.COLOR_BGR2GRAY)
cf.img_gray_equalized = cv2.equalizeHist(cf.img_gray)

# cv2.imshow('img_gray_equalized', cf.img_gray_equalized)


kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27,27))
close = cv2.morphologyEx(cf.img_gray_equalized, cv2.MORPH_CLOSE, kernel1)
cv2.imshow('close', close)
div = np.float32(cf.img_gray_equalized)/(close)
normed = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))


th2 = cv2.adaptiveThreshold(normed, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY, 43, 22)
th3 = cv2.adaptiveThreshold(normed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 43, 22)



# stacking images side-by-side
out = np.hstack((cf.img_gray, cf.img_gray_equalized, normed))
out2 = np.hstack((th2, th3))
cv2.imshow('out', out)
cv2.imshow('out2', out2)

cv2.waitKey(0)
