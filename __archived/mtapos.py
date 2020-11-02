#!/usr/bin/env python
from __future__ import print_function
import rospy
# import cv2
from std_msgs.msg import String, Float32
from sensor_msgs.msg import CompressedImage
# from cv_bridge import CvBridge, CvBridgeError
# import numpy as np
# from modules.function import processImage


class image_processing:
    def __init__(self):
        self.k = 1
        self.sign = []
        self.signTrack = -1
        self.line = 170
        self.speed = 10
        self.angle = 15
        # self.rgbImageSub = rospy.Subscriber(
        #     "/MTA_PoS/camera/rgb/compressed", CompressedImage, self.getRgbImage, queue_size=1)
        self.AnglePub = rospy.Publisher(
            "/set_angle", Float32, queue_size=1)  # angle of car
        # self.CameraAnglePub = rospy.Publisher(
        #     "/set_camera_angle", Float32, queue_size=1)  # angle of camera
        self.SpeedPub = rospy.Publisher(
            "/set_speed", Float32, queue_size=1)  # speed of car
        self.signMode = 0
        self.signstep = 0
        self.maxspeed = 60
        self.signCount = 0
    # call back function

    def getRgbImage(self, rosdata):
        global angleRight, speedTurn
        # np_arr = np.fromstring(rosdata.data, np.uint8)
        # image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # processImage(self, image_np)
        self.setAngle()
        self.setSpeed()
    # public function

    def setAngle(self):
        self.AnglePub.publish(self.angle)

    # def setCameraAngle(self):
    #     self.CameraAnglePub.publish(self.angle)

    def setSpeed(self):
        self.SpeedPub.publish(self.speed)


def main():
    rospy.init_node('image_processing', anonymous=True)
    temp = image_processing()
    print("connected...")
    temp.setAngle()
    temp.setSpeed()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()