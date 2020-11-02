#!/usr/bin/env python3

import time
import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
import numpy as np
# import rospkg
import cv2
import config as cf
# from utils.ros_publish import set_speed, set_steer, print_speed_lcd
import utils.ros_publish as ros_pub
import utils.ros_subscribe as ros_sub
from utils.camera import Camera
from utils.hand_control import HandControl
from utils.auto_control import AutoControl
# from utils.lidar import Lidar
from utils.object_detection import Detector

import threading


# get and set path
# rospack = rospkg.RosPack()
# path = rospack.get_path('mtapos')

# init node
rospy.init_node('run_auto', anonymous=True, disable_signals=True)

cf.model_path = '/models/08.02_newcam/model_o3456_320.h5'
cf.gray_model = False
cf.model_crop_top = 130
cf.model_crop_bottom = 55

# set control variables
cf.running = True
cf.pause = True
cf.ready = False

cf.collect_data = False
cf.init_speed = 17 # init speed
cf.angle = 0 # init steer
cf.imu_angle = 0 # imu_angle subscribe from topic
cf.first_imu_angle = None

# set img / video variables
cf.HEIGHT = 240 # 480
cf.WIDTH = 320 # 640
# cf.HEIGHT_D = 240 # height of depth image
# cf.WIDTH_D = 320 # width of depth image
cf.img_rgb = np.zeros((cf.WIDTH, cf.HEIGHT, 3), np.uint8)
cf.img_depth = np.zeros((cf.WIDTH, cf.HEIGHT), np.uint8)
cf.rgb_processed = np.zeros((cf.WIDTH, cf.HEIGHT), np.uint8)
cf.depth_processed = np.zeros((cf.WIDTH, cf.HEIGHT), np.uint8)

cf.change_steer = False


cf.is_record = False
cf.save_image = False

cf.sub_lidar = None
cf.center = 160



#### init object
m_Cam = Camera()
m_HandControl = HandControl()
m_AutoControl = AutoControl()
# m_Lidar = Lidar()
m_Detector = Detector()


def listenner():
    while cf.running:
        if not cf.pause:
            # cf.sub_lidar = rospy.Subscriber('/scan', LaserScan, m_Lidar.on_receive, queue_size=1)
            cf.sub_getIMUAngle = rospy.Subscriber('/imu_angle', Float32, ros_sub.on_get_imu_angle, queue_size=1)
            # rospy.Subscriber("/angle", Float32, get_angle, queue_size=1)
            rospy.spin()



if __name__ == "__main__":
    # Reset everything
    ros_pub.set_steer(0)
    ros_pub.set_speed(0)

    ### start thread
    auto_process_thread = threading.Thread(name="auto_process_thread", target=m_AutoControl.run)
    auto_process_thread.start()

    get_rbg_thread = threading.Thread(name="get_rbg_thread", target=m_Cam.get_rgb)
    get_rbg_thread.start()

    get_depth_thread = threading.Thread(name="get_depth_thread", target=m_Cam.get_depth)
    get_depth_thread.start()

    # detector_thread = threading.Thread(name="detector_thread", target=m_Detector.process)
    # detector_thread.start()

    show_thread = threading.Thread(name="show_thread", target=m_Cam.show2)
    show_thread.start() # save data thread

    control_thread = threading.Thread(name="control_thread", target=m_HandControl.hand_control)
    control_thread.start()
    
    get_rbg_thread.join()
    get_depth_thread.join()
    # detector_thread.join()
    # show_thread.join()
    auto_process_thread.join()
    control_thread.join()


    # listenner()
