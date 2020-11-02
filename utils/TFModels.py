import tensorflow as tf
from keras.models import load_model
from tensorflow import keras
import keras.backend as K
import cv2
import config as cf
import sys
import math

class TFModels():
    def __init__(self, model_path):
        # super(TFModels, self).__init__()

        self.config = tf.ConfigProto(
            device_count={'GPU': 1},
            intra_op_parallelism_threads=1,
            allow_soft_placement=True
        )
        self.config.gpu_options.allow_growth = True
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.6
        K.clear_session()
        self.session = tf.Session(config=self.config)
        K.set_session(self.session)

        self.model = load_model(sys.path[1]+model_path)


    def caculateAngle(self, center, width):
        temp = math.atan(float(abs(float(center)-float(width/2))/float(cf.line_from_bottom)))
        angle = math.degrees(float(temp))
        if center > width/2:
            return -angle
        else:
            return angle

    def predictCenter(self, img):
        # img = cv2.resize(image, (320, 160))
        with self.session.as_default():
            with self.session.graph.as_default():
                predict = self.model.predict(np.array([img])/255.0)[0]
        cf.predict = predict[0]
        center = int(predict[0]*img.shape[1])
        angle = self.caculateAngle(center, img.shape[1])
        # cv2.imshow('img', img)
        # print('img.shape', img.shape, 'image.shape', image.shape)
        return angle, center

    def preprocess(self, image):
        image = cv2.resize(image, (25, 25))
        # image = imutils.resize(image, 25)
        blur = cv2.GaussianBlur(image, (3, 3), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
        image = gray
        image = image.reshape((25, 25, 1))
        return image


