#!/usr/bin/env python3

import sys
import time
import rospy
from std_msgs.msg import String, Float32, Bool
from sensor_msgs.msg import LaserScan
import config as cf
import rospkg
from pynput import keyboard

# import tensorflow as tf
# from keras.models import load_model
# from tensorflow import keras
# import keras.backend as K
# import math
# from primesense import openni2
# from primesense import _openni2 as c_api
# import numpy as np
# import cv2
# from pynput import keyboard
# import os
import threading
# from yolo import YOLO
# from PIL import Image


# get and set path
rospack = rospkg.RosPack()
path = rospack.get_path('mtapos')
sys.path.insert(1, path)

# init node
rospy.init_node('test_motor', anonymous=True, disable_signals=True)



cf.wait_tick_count = None
cf.go_tick_count = 0
cf.speed = 10
cf.angle = 0

rate = rospy.Rate(10)  # 10hz

class Publish(object):
    def __init__(self):
        self.speed_pub = rospy.Publisher('/set_speed', Float32, queue_size=1)
        self.steer_pub = rospy.Publisher('/set_angle', Float32, queue_size=1)
        return

    def set_speed(self, speed):
        if speed > 30:
            speed = 30
        elif speed < -30:
            speed = -30
        

        connections = self.speed_pub.get_num_connections()
        rospy.loginfo('Connections: %d', connections)
        if connections > 0:
            print('\t set_speed', speed)
            self.speed_pub.publish(speed)
            rospy.loginfo('Published')
        rate.sleep()


    def set_steer(self, steer):
        print('\t set_angle', steer)
        self.steer_pub.publish(steer)


class Control(Publish):
    def __init__(self):
        super(Control, self).__init__()
        return

    def pause(self):
        print('Pause!')
        time_to_sleep = cf.speed*0.5/15
        # before pause set speed to its negative value to go backwards
        cf.speed = -cf.speed
        self.set_speed(cf.speed)
        cf.pause = True
        time.sleep(time_to_sleep)
        # and = 0 to stop
        cf.speed = 0
        self.set_speed(cf.speed)
        # and reset angle = 0 for next time
        cf.steer = 0
        self.set_steer(cf.steer)
    
    def quit(self):
        self.pause()

        time.sleep(1)
        print('QUit')
        os._exit(0)


class App(Control):

    def __init__(self):
        super(App, self).__init__()
        return
    
    def on_key_press(self, key):
        self.set_speed(10)
    
    def srun(self):
        with keyboard.Listener(
                on_press=self.on_key_press) as listener:
            listener.join()

    def run(self):
        while True:
            print('cf.wait_tick_count, cf.go_tick_count', cf.wait_tick_count, cf.go_tick_count)
            if cf.wait_tick_count is None and cf.go_tick_count is not None:
                if cf.go_tick_count < 10:
                    cf.go_tick_count += 1
                    self.set_speed(12)
                    # time.sleep(0.1)

                    # if cf.go_tick_count > 0 and cf.go_tick_count % 1000 == 0:
                    #     cf.angle += 2
                    #     if cf.angle > 60:
                    #         cf.angle = 60
                    #     elif cf.angle < -60:
                    #         cf.angle = -60
                    #     self.set_steer(cf.angle)
                else:
                    cf.go_tick_count = None
                    cf.wait_tick_count = 0

            if cf.go_tick_count is None and cf.wait_tick_count is not None:
                if cf.wait_tick_count < 5:
                    cf.wait_tick_count += 1
                    self.set_speed(8)
                    time.sleep(0.1)
                    # self.set_steer(0)
                else:
                    cf.wait_tick_count = None
                    cf.go_tick_count = 0


if __name__ == "__main__":
    app = App()
    app.run()

