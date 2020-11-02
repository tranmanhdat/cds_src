import cv2
import numpy as np
from sklearn import preprocessing

cropped = cv2.imread('/home/ubuntu/catkin_ws/src/mtapos/output/stop__3.jpg')
template = cv2.imread('/home/ubuntu/catkin_ws/src/mtapos/template/stop__0.jpg')

cropped = cv2.resize(cropped, (40,40))
template = cv2.resize(template, (40,40))

cropped_norm = preprocessing.normalize(cropped.reshape(1,-1))
template_norm = preprocessing.normalize(template.reshape(1,-1))
s = np.dot(cropped_norm, template_norm.T)
print('**** s', s)
# if s > 0.8:
#     print('True STOP !!')

#     cv2.imshow('stop', sign_region)