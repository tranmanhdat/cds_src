#!/usr/bin/env python3

# import tensorflow as tf
# from keras.models import load_model
# from tensorflow import keras
# import keras.backend as K
# import math
# import time
# from primesense import openni2
# from primesense import _openni2 as c_api
# import cv2
# from yolo import YOLO
# from PIL import Image
import sys
import rospy
from std_msgs.msg import String, Float32, Bool
from sensor_msgs.msg import LaserScan
import numpy as np
import rospkg
import config as cf
import os
import threading

from utils.Camera import Camera
from utils.ImageProcessing import ImageProcessing
from utils.Control import AutoControl
from utils.Subscribe import Subscribe


red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
yellow = (52, 235, 232)
cf.listColor = [red, green, blue, yellow]
cf.detect_sign_region = {'top':0, 'bottom':240, 'left':320, 'right':640}

cf.turnSignal = False # do I see turn 

route = {
    'thang': ['left', 'left', 'left', 'right', 'right', 'right', 'right', 'left'],
    'phai': ['left', 'left', 'left', 'right', 'right', 'left', 'left', 'left']
}
cf.turned = [] # what did I turn
cf.current_route = None
cf.start_tick_count = 0 # If this is set to 0, it will be in running straight mode (with fixed angle=0 and speed=MAX_SPEED) and ignore all other calculation. If set to None, it uses predictCenter to calculate angle and set speed.

# get and set path
rospack = rospkg.RosPack()
path = rospack.get_path('mtapos')
sys.path.insert(1, path)

# init node
rospy.init_node('AllInOne', anonymous=True, disable_signals=True)


# Model
cf.sign_weight_h5_path = "/models/yolo/yolov3-tiny-sign__phai_thang.h5"
cf.sign_anchor_path = "/models/yolo/yolov3-tiny-sign__phai_thang-anchors.txt"
cf.sign_class_name_path = "/models/yolo/sign__phai_thang.names"
# cf.sign_weight_path = "/models/yolo/yolov3-tiny-sign__phai_thang.weights"
# cf.sign_data_path = "/models/yolo/sign__phai_thang.data"
# cf.sign_config_path = "/models/yolo/yolov3-tiny-sign__phai_thang.cfg"
# cf.lane_model_path = "/models/lane/model_full+den_2000_320x140_100.h5"
# cf.lane_model_path = "/models/lane/dem_full_320x140.h5"
# cf.lane_model_path = "/models/lane/dem_full+cuasang_320x140.h5"
cf.lane_model_path = "/models/lane/rephai_320x140.h5"
# cf.lane_model_path = "/models/lane/dem_320x140.h5"
cf.crop_top_detect_center = 100
# cf.sign_model_path = "/models/signgray.h5"

cf.line_from_bottom = 90
cf.predict = 0
cf.reduce_speed = False
cf.speed_reduced = 10
cf.imu_early = 10

# set control variables
cf.running = True # Must set to True. Scripts executed only when cf.running == True
cf.do_detect_barrier = False # = True when button 1 is clicked. = False when sensor 2 detects barrier open. Used everytime bring the car to the start position to redetect barrier open and restart everything
cf.pause = True # set speed = 0 and stop publishing stuff
cf.ready = False # make sure everything is loaded and image is successfully retrieved before car runs or speed/angle is published
cf.change_steer = False
cf.got_rgb_image = False
cf.got_depth_image = False

# Speed and angle
cf.center = 0 # center to calculate angle
cf.init_speed = 10  # init speed
cf.angle = 0
cf.speed = 0
cf.end_tick_from_start = 5
 #20(for 17,14) # 22(for 14,12)
cf.MAX_SPEED = 13
cf.FIXED_SPEED_TURN = 13
# 10 * maxLeft angle(20 degree) = -200, mul 10 to smooth control
cf.MIN_ANGLE = -60
cf.MAX_ANGLE = 60  # 10 * maxRight angle
cf.angle_increasement = 11
cf.speed_increasement = 1.0

# data collection
cf.is_record = True
cf.save_image = True
cf.save_log_angles = False

# set img / video variables
cf.HEIGHT = 240  # 480
cf.WIDTH = 320  # 640
# cf.HEIGHT_D = 240 # height of depth image
# cf.WIDTH_D = 320 # width of depth image
cf.img_rgb_raw = np.zeros((640, 480, 3), np.uint8)
cf.img_rgb_resized = np.zeros((cf.WIDTH, cf.HEIGHT, 3), np.uint8)
cf.img_depth = np.zeros((cf.WIDTH, cf.HEIGHT), np.uint8)
cf.depth_processed = np.zeros((cf.WIDTH, cf.HEIGHT), np.uint8)
cf.rgb_viz = np.zeros((cf.WIDTH, cf.HEIGHT), np.uint8)
# cf.img_detect_sign = np.zeros((cf.detect_sign_region['right']-cf.detect_sign_region['left'], cf.detect_sign_region['bottom']-cf.detect_sign_region['top']), np.uint8)
# cf.rgb_viz = np.zeros((cf.detect_sign_region['right']-cf.detect_sign_region['left'], cf.detect_sign_region['bottom']-cf.detect_sign_region['top']), np.uint8)

# variables for sign detection
cf.signMode = None
cf.signstep = 0
cf.signCount = 0
cf.k = 1
cf.sign = []
cf.signTrack = -1
cf.sign_detect_step = 3
cf.specialCorner = 1
cf.signSignal = None
cf.sign_bbbox = None

# subscribe stuff
cf.sub_btn1 = None
cf.sub_btn2 = None
cf.sub_btn3 = None
cf.sub_sensor2 = None
cf.sub_lidar = None
cf.sub_getIMUAngle = None
cf.imu_angle = 0  # imu_angle subscribe from topic
cf.first_imu_angle = None

cf.speed_pub = rospy.Publisher('/set_speed', Float32, queue_size=1)
cf.steer_pub = rospy.Publisher('/set_angle', Float32, queue_size=1)
cf.lcd_pub = rospy.Publisher('/lcd_print', String, queue_size=1)
cf.rate = rospy.Rate(10)  # 10hz



def listenner():
    ros_sub = Subscribe()
    # If button 1 is clicked, set running = True
    h = rospy.Subscriber(
        '/bt1_status', Bool, ros_sub.on_get_btn_1, queue_size=1)
    cf.sub_btn2 = rospy.Subscriber(
        '/bt2_status', Bool, ros_sub.on_get_btn_2, queue_size=1)
    cf.sub_btn3 = rospy.Subscriber(
        '/bt3_status', Bool, ros_sub.on_get_btn_3, queue_size=1)
    cf.sub_sensor2 = rospy.Subscriber(
        '/ss2_status', Bool, ros_sub.on_get_sensor2, queue_size=1)
    cf.sub_getIMUAngle = rospy.Subscriber(
        '/imu_angle', Float32, ros_sub.on_get_imu_angle, queue_size=1)
    cf.sub_lidar = rospy.Subscriber(
        '/scan', LaserScan, ros_sub.on_get_lidar, queue_size=1)
    # while cf.running:
    #     if not cf.pause:
    #         # cf.sub_lidar = rospy.Subscriber('/scan', LaserScan, ros_sub.on_get_lidar, queue_size=1)
    #         cf.sub_getIMUAngle = rospy.Subscriber(
    #             '/imu_angle', Float32, ros_sub.on_get_imu_angle, queue_size=1)
    rospy.spin()



class App(Camera, ImageProcessing, AutoControl):

    def __init__(self):
        super(App, self).__init__()
        return

    def run(self):
        self.clear_lcd()

        # start thread
        # auto_control_thread = threading.Thread(
        #     name="auto_control_thread", target=self.auto_control)
        # auto_control_thread.start()

        # cf.running = True
        # cf.do_detect_barrier = False
        # cf.ready = True
        # self.run_car_by_signal()
        
        get_rgb_thread = threading.Thread(
            name="get_rbg_thread", target=self.get_rgb)
        get_rgb_thread.start()


        # process_rgb_thread = threading.Thread(
        #     name="process_rgb_thread", target=self.process_rgb_image)
        # process_rgb_thread.start()

        get_sign_thread = threading.Thread(
            name="get_sign_thread", target=self.getSign)
        get_sign_thread.start()

        get_turn_thread = threading.Thread(
            name="get_turn_thread", target=self.getTurn)
        get_turn_thread.start()

        get_center_thread = threading.Thread(
            name="get_center_thread", target=self.getCenter) # Drive control in this function
        get_center_thread.start()

        # get_depth_thread = threading.Thread(
        #     name="get_depth_thread", target=self.get_depth)
        # get_depth_thread.start()

        # show_thread = threading.Thread(name="show_thread", target=self.visualize)
        # show_thread.start() # save data thread

        # control_thread = threading.Thread(
        #     name="control", target=self.hand_control)
        # control_thread.start()
        # self.hand_control()

        # listen_thread = threading.Thread(name="listen_thread", target=listenner)
        # listen_thread.start() # save data thread
        listenner()

        get_rgb_thread.join()
        get_sign_thread.join()
        get_turn_thread.join()
        get_center_thread.join()
        # get_depth_thread.join()
        # show_thread.join()
        # auto_control_thread.join()
        # control_thread.join()


if __name__ == "__main__":
    app = App()
    app.run()
