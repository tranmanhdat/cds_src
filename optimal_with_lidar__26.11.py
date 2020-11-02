#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from sklearn import preprocessing

import math
import sys
import time
import rospy
from std_msgs.msg import String, Float32, Bool
from sensor_msgs.msg import LaserScan, Imu
from primesense import openni2
from primesense import _openni2 as c_api
import numpy as np
import rospkg
import cv2
import config as cf
from pynput import keyboard
import os
import threading
import glob
# from models.darknet_video import load_model, sign_detect
import darknet

cf.netMain = None
cf.metaMain = None
cf.altNames = None

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
# cf.detect_sign_region = {'top':0, 'bottom':240, 'left':320, 'right':640}
cf.detect_sign_region = {'top':0, 'bottom':120, 'left':160, 'right':320}
cf.fps_count = 0

cf.turned = [] # what did I turn
cf.current_route = None
cf.start_tick_count = 0 # If this is set to 0, it will be in running straight mode (with fixed angle=0 and speed=MAX_SPEED) and ignore all other calculation. If set to None, it uses predictCenter to calculate angle and set speed.


# get and set path
rospack = rospkg.RosPack()
path = rospack.get_path('mtapos')
sys.path.insert(1, path)

# init node
rospy.init_node('optimal', anonymous=True, disable_signals=True)


# set img / video variables
cf.HEIGHT = 240  # 480
cf.WIDTH = 320  # 640
# cf.HEIGHT_D = 240 # height of depth image
# cf.WIDTH_D = 320 # width of depth image
# cf.img_rgb_raw = np.zeros((640, 480, 3), np.uint8)
cf.img_rgb_raw = np.zeros((cf.WIDTH, cf.HEIGHT, 3), np.uint8)
cf.img_depth = np.zeros((cf.WIDTH, cf.HEIGHT), np.uint8)
cf.sign_depth = None
cf.depth_processed = np.zeros((cf.WIDTH, cf.HEIGHT), np.uint8)
cf.rgb_viz = np.zeros((cf.WIDTH, cf.HEIGHT), np.uint8)


# For center detection
h_center_detect, w_center_detect = 135, 320
cf.line_from_bottom = 70
cf.crop_top_detect_center = cf.HEIGHT - h_center_detect # crop from top
append_more = True
if append_more:
    org_thang = (160, 20) # draw "route thang" after detecting sign
    org_phai = (295, 20) # draw "route phai" after detecting sign
    org_trai = (25, 20) # draw "route trai" after detecting sign
    pts_lanephai = [(w_center_detect//2+30, 0), (w_center_detect, 40)] # draw "bam lane phai"
    pts_lanetrai = [(0, 0), (w_center_detect//2-30, 40)] # draw "bam lane trai" (after detecting moving objects)
else:
    org_thang = (160, h_center_detect-50) # draw "route thang" after detecting sign
    org_phai = (295, h_center_detect-50) # draw "route phai" after detecting sign
    org_trai = (25, h_center_detect-50) # draw "route trai" after detecting sign
    pts_lanephai = [(w_center_detect//2+30, h_center_detect-40), (w_center_detect, h_center_detect)] # draw "bam lane phai"
    pts_lanetrai = [(0, h_center_detect-40), (w_center_detect//2-30, h_center_detect)] # draw "bam lane trai" (after detecting moving objects)

# set control variables
cf.running = True # Must set to True. Scripts executed only when cf.running == True
cf.do_detect_barrier = False # = True when button 1 is clicked. = False when sensor 2 detects barrier open. Used everytime bring the car to the start position to redetect barrier open and restart everything
cf.pause = True # set speed = 0 and stop publishing stuff
cf.ready = False # make sure everything is loaded and image is successfully retrieved before car runs or speed/angle is published
cf.got_rgb_image = False
cf.got_depth_image = False

# Speed and angle
cf.angle = 0
cf.speed = 0
cf.FIXED_SPEED_NORMAL = 20
# cf.FIXED_SPEED_NORMAL = 17
cf.FIXED_SPEED_STRAIGHT = cf.FIXED_SPEED_NORMAL
cf.FIXED_SPEED_route_difficult = 19 # route kho di toc do nay
cf.SPEED_PASS_OBJ = 21 # toc do chay de vuot vat can
cf.FIXED_SPEED_TURN = 17 # toc do khi re
cf.end_tick_from_start = 42 # tick de di thang doan dau #20(for 17,14) # 22(for 14,12)
cf.MAX_SPEED = 26
# cf.MAX_SPEED = 16
cf.MIN_SPEED = 15
cf.reduce_speed = 0 # 0: normal | 1: reduce speed | -1: speed up
cf.speed_reduced = 15 # toc do thap
cf.imu_early = 15
cf.speed_increasement = 1
cf.init_speed = 14 # init speed to increase by time
cf.speed_chuyen_lane = 17 # 13 # toc do de chuyen lane
cf.speed_detect_sign = 16 # toc do di de bat bien bao

# detect chuyen lane
cf.lane = None
# Tin hieu chuyen lane:
#    0: chua chuyen, ko chuyen 
#    1: nhan tin hieu chuyen tu PHAI -> TRAI
#    2: nhan tin hieu chuyen tu TRAI -> PHAI
#   -1: da chuyen tu PHAI -> TRAI, ko bao h chuyen P->T nua
#   -2: da chuyen tu TRAI -> PHAI, ko bao h chuyen T->P nua
cf.do_chuyen_lane = 0
cf.tick_stop_chuyen_lane = 10
cf.goc_chuyen_lane = 40
cf.tick_stop_giu_lane = 7
# cf.max_angle_giu_lane = 30
cf.run_lidar = False
cf.distance_to_detect_lidar = 3.0 # 3.5
cf.distance_behind_lidar = 1.0
cf.max_angle_to_detect_lidar = 2.3
cf.segment_chuyen_lane = 0
cf.pass_object = False

# variables for sign detection
cf.sign = []
cf.sign_detect_step = 3
cf.do_detect_sign = False
cf.signSignal = None
cf.count_sign_step = 0
cf.tick_to_pass_turn = None
cf.max_tick_to_pass_turn = 21
cf.sign_bbox = None

# For stop sign detection
cf.do_detect_stop_sign = False
cf.turn_on_depth = False
cf.low_thresh = 2000 # min distance to stop
cf.up_thresh = 25000 # max distance to stop
cf.do_stop = False
cf.stop_detected = False
cf.begin_stop_time = None
cf.stopSignal = None
cf.sec_to_stop = 0.38
cf.end_tick_detect_stop = 25
cf.MAX_DEPTH_DISTANCE = 12000

# subscribe stuff
cf.sub_btn1 = None
cf.sub_btn2 = None
cf.sub_btn3 = None
cf.sub_sensor2 = None
cf.sub_lidar = None
cf.sub_getIMUAngle = None
cf.imu_angle = 0  # imu_angle subscribe from topic
cf.first_imu_angle = None

# imu var
cf.t_imu = time.time()
cf.gyro_z = 0
cf.imu_angle = 0
cf.wz = 0.0
cf.last_gyro_z = 0.0

def reset_imu():
    cf.gyro_z = 0
    cf.imu_angle = 0
    cf.wz = 0.0
    cf.last_gyro_z = 0.0
    cf.gyro_y = 0
    cf.angle_y = 0
    cf.wy = 0.0
    cf.last_gyro_y = 0.0


cf.speed_pub = rospy.Publisher('/set_speed', Float32, queue_size=1)
cf.steer_pub = rospy.Publisher('/set_angle', Float32, queue_size=1)
cf.lcd_pub = rospy.Publisher('/lcd_print', String, queue_size=1)


class Publish(object):
    def __init__(self):
        return

    def set_speed(self, speed):
        if cf.running:
            if cf.pause or not cf.ready:
                speed = 0
            elif speed > 30:
                speed = 30
            elif speed < -30:
                speed = -30
            
            if (not cf.pause and cf.ready) or speed == 0:
                cf.speed_pub.publish(speed)

    def set_steer(self, steer):
        if steer == 0 or (cf.running and not cf.pause and cf.ready):
            cf.steer_pub.publish(steer)

    def set_lcd(self, text):
        cf.lcd_pub.publish(text)

    def clear_lcd(self):
        # clear lcd
        for row in range(4):
            self.set_lcd("0:{}:{}".format(row, ' '*20))

    def print_lcd(self, text):
        print(text)
        self.clear_lcd()
        time.sleep(0.1)
        self.set_lcd("0:0:"+text) #col:row:content
        time.sleep(0.1)


class Control(Publish):
    error_arr = np.zeros(5)
    t = time.time()

    def __init__(self):
        super(Control, self).__init__()
        return

    def PID(self, error, p=0.43, i=0, d=0.02):
        self.error_arr[1:] = self.error_arr[0:-1]
        self.error_arr[0] = error
        P = error*p
        delta_t = time.time() - self.t
        self.t = time.time()
        D = (error-self.error_arr[1])/delta_t*d
        I = np.sum(self.error_arr)*delta_t*i
        angle = P + I + D
        if abs(angle) > 60:
            angle = np.sign(angle)*60
        return float(angle)

    def reset_all(self):
        cf.signSignal = None
        cf.do_detect_sign = False
        cf.count_sign_step = 0
        cf.sign = []
        cf.sign_bbox = None
        reset_imu()
        cf.first_imu_angle = None
        cf.FIXED_SPEED_STRAIGHT = cf.FIXED_SPEED_NORMAL
        cf.do_chuyen_lane = 0
        cf.segment_chuyen_lane = 0
        cf.pass_object = False
        cf.reduce_speed = 0
        cf.run_lidar = False
        cf.turn_on_depth = False
        cf.stop_detected = False
        cf.begin_stop_time = None
        cf.stopSignal = None

        # cf.current_route = None
        # cf.lane = None
        # # cf.tick_to_pass_turn = None
        # cf.turned = []
        # cf.start_tick_count = 0
        # cf.fixed_speed = None
        # cf.signSignal = None
        # cf.value_to_start_at_section2 = 0

        # start from section 2
        cf.current_route = 'trai'
        cf.lane = 'trai'
        cf.tick_to_pass_turn = cf.max_tick_to_pass_turn
        cf.turned = ['right', 'right', 'right', 'left', 'left', 'right']
        cf.start_tick_count = None
        cf.fixed_speed = cf.FIXED_SPEED_route_difficult
        cf.signSignal = 'trai_certain'
        cf.value_to_start_at_section2 = 180

        self.run_car_by_signal()


    def run_car_by_signal(self):
        '''
        Called when a signal is sent to start run the car.
        But car is run only when everything is loaded.
        cf.ready makes sure of that
        '''
        if cf.pause is True:
            cf.pause = False
            self.print_lcd('Send signal to run')
            while cf.ready is False:
                self.print_lcd("Wait a moment...")
                if cf.ready is True:
                    break

    def pause(self):
        cf.pause = True
        cf.speed = 0
        self.set_speed(cf.speed)
        # and reset angle = 0 for next time
        cf.steer = 0
        self.set_steer(cf.steer)

        self.print_lcd('Pause!')
    
    def quit(self):
        self.pause()
        
        # Unscribe all sensors
        if cf.sub_btn1 is not None:
            cf.sub_btn1.unregister()
        if cf.sub_btn2 is not None:
            cf.sub_btn2.unregister()
        if cf.sub_btn3 is not None:
            cf.sub_btn3.unregister()
        if cf.sub_sensor2 is not None:
            cf.sub_sensor2.unregister()

        if cf.sub_lidar is not None:
            cf.sub_lidar.unregister()
        if cf.sub_getIMUAngle is not None:
            cf.sub_getIMUAngle.unregister()

        cv2.destroyAllWindows()
        self.clear_lcd()
        time.sleep(1)
        os._exit(0)



class HandControl(Publish):
    def __init__(self):
        super(HandControl, self).__init__()
        return

    def hand_control(self):
        self.t_start = time.time()
        with keyboard.Listener(
                on_press=self.on_key_press) as listener:
            listener.join()

    def on_key_press(self, key):
        try:
            if key.char == 's':  # start
                if cf.running is False:
                    cf.running = True
                self.print_lcd('Start!')
                if cf.pause is True:
                    self.run_car_by_signal()
            if key.char == 'r':  # reset
                if cf.running is False:
                    cf.running = True
                self.print_lcd('Reset! Start!')
                self.reset_all()
                # if cf.pause is True:
                #     self.reset_all()
            if key.char == 'q':  # quit
                print('Quit pressed')
                self.pause()
            if key.char == 'z':  # chuyen lane
                if cf.lane == 'trai':
                    cf.do_chuyen_lane = 2 # chuyen tu trai sang phai
                elif cf.lane == 'phai':
                    cf.do_chuyen_lane = 1 # chuyen tu phai sang trai

        except AttributeError:
            ''' Control speed '''
            if key == keyboard.Key.up:
                if cf.running and not cf.pause:
                    cf.FIXED_SPEED_STRAIGHT += cf.speed_increasement
                    cf.FIXED_SPEED_STRAIGHT = min(cf.FIXED_SPEED_STRAIGHT, cf.MAX_SPEED)
            if key == keyboard.Key.down:
                if cf.running and not cf.pause:
                    cf.FIXED_SPEED_STRAIGHT -= cf.speed_increasement
                    cf.FIXED_SPEED_STRAIGHT = max(cf.FIXED_SPEED_STRAIGHT, cf.MIN_SPEED)

            if key == keyboard.Key.esc:
                print('esc pressed')
                self.quit()

class AutoControl(Control):
    arr_speed = np.zeros(5)
    timespeed = time.time()
    tick_to_finish__route_thang = None
    fixed_speed = cf.init_speed # init speed to increase by time
    tick_chuyen_lane = None
    tick_giu_lane = None
    angle_giu_lane = None
    last_angles_chuyen_lane = []
    tick_start_save_angle = cf.tick_stop_chuyen_lane - cf.tick_stop_giu_lane
    do_not_chuyen_lane = False
    wait_pass_2_turn_to_chuyen_lane = False
    start_tick_detect_stop = None
    
    def __init__(self):
        super(AutoControl, self).__init__()
        
        return

    def speedControl(self, angle, fixed_speed=None):
        tempangle = abs(angle)
        if fixed_speed is not None:
            speed = fixed_speed # fixed_speed
        elif cf.reduce_speed == 1:
            speed = cf.speed_reduced # Reduce speed
        elif cf.reduce_speed == -1:
            speed = cf.MAX_SPEED # Speed up
        else: # cf.reduce_speed == 0
            if abs(angle) > 18:
                speed = cf.FIXED_SPEED_TURN
            else:
                speed = cf.FIXED_SPEED_STRAIGHT

        cf.speed = speed
        self.set_speed(cf.speed)
        # time.sleep(0.02)

    def whereAmI__phai(self):
        if cf.imu_angle > 90-cf.imu_early-20 and len(cf.turned) == 0: # turn first left
            cf.turned.append('left')
            cf.reduce_speed = 0 # go at normal speed
        if 180+cf.imu_early > cf.imu_angle > 180-cf.imu_early and len(cf.turned) == 1: # turn second left
            cf.turned.append('left')
        if 270+cf.imu_early > cf.imu_angle > 270-cf.imu_early and len(cf.turned) == 2: # turn third left
            cf.turned.append('left')
        if 180+cf.imu_early > cf.imu_angle > 180-cf.imu_early and len(cf.turned) == 3: # turn right at finish of segment 1
            cf.turned.append('right')
            cf.do_detect_sign = True # now need to detect sign
            cf.count_sign_step = 0
            cf.fixed_speed = cf.speed_detect_sign # go really slow to detect sign !!!

        ''' Get to second segment '''
        # route phai
        if cf.current_route == 'phai':
            # After detecting sign and turn successfully, set reduce_speed = 0 to go at normal speed (set in signRecognize_)
            
            if 90+cf.imu_early > cf.imu_angle > 90-cf.imu_early and len(cf.turned) == 4: # turn right
                cf.turned.append('right')

            if 180+cf.imu_early > cf.imu_angle > 180-cf.imu_early and len(cf.turned) == 5: # turn left
                cf.turned.append('left')
            if 270+cf.imu_early > cf.imu_angle > 270-cf.imu_early and len(cf.turned) == 6: # turn left
                cf.turned.append('left')
            if 360+cf.imu_early > cf.imu_angle > 360-cf.imu_early and len(cf.turned) == 7: # turn left
                cf.turned.append('left')
                # we pass all the route. Now speed up to finish line!
                cf.reduce_speed = -1
                self.tick_to_finish__route_thang = 0
                self.start_tick_detect_stop = 0

            if len(cf.turned) == 8: # after some tick_to_finish__route_thang we would need to slow down to detect stop sign
                self.tick_to_finish__route_thang += 1
                if self.tick_to_finish__route_thang > 50:
                    cf.reduce_speed = 0

        # route thang
        if cf.current_route == 'thang':
            # After detecting sign successfully, set reduce_speed = 0 to go at normal speed AND set cf.run_lidar = True to detect moving object (set in signRecognize_)

            if 90+cf.imu_early > cf.imu_angle > 90-cf.imu_early and len(cf.turned) == 4: # turn right
                cf.turned.append('right')

            if 0+cf.imu_early > cf.imu_angle > 0-cf.imu_early and len(cf.turned) == 5: # turn right
                cf.turned.append('right')
            if -90+cf.imu_early > cf.imu_angle > -90-cf.imu_early and len(cf.turned) == 6: # turn right
                cf.turned.append('right')
            if 0+cf.imu_early > cf.imu_angle > 0-cf.imu_early and len(cf.turned) == 7: # turn left
                cf.turned.append('left')


    def auto_control__phai(self, angle_from_center):
            #############################
            # Di thang doan dau
            #############################
            # just go straight first few ticks
            if cf.start_tick_count is not None:
                if cf.start_tick_count < cf.end_tick_from_start:
                    if cf.start_tick_count == 0:
                        cf.fixed_speed = cf.init_speed
                    cf.start_tick_count += 1
                    cf.angle = 0
                    if cf.start_tick_count % 5 == 1 and cf.fixed_speed < cf.MAX_SPEED:
                        cf.fixed_speed += 1
                else:
                    cf.start_tick_count = None
                    cf.reduce_speed = 1
                    cf.fixed_speed = None
            
            #############################
            # Chuyen lane
            #############################
            elif cf.do_chuyen_lane > 0: # khong chuyen lane cho re
                cf.fixed_speed = cf.speed_chuyen_lane
                if cf.do_chuyen_lane == 1: # chuyen trai
                    cf.angle = cf.goc_chuyen_lane
                    if cf.lane == 'phai':
                        cf.lane = 'trai'
                        # check xem co dang o giua duong khong, giua duong thi giam cf.tick_stop_chuyen_lane xuong
                        # cf.tick_stop_chuyen_lane = 14
                    if len(cf.turned) == 4: # chuyen o doan dau
                        cf.segment_chuyen_lane = 1
                    elif len(cf.turned) == 5:
                        cf.segment_chuyen_lane = 2
                    elif len(cf.turned) == 6:
                        cf.segment_chuyen_lane = 3
                    
                elif cf.do_chuyen_lane == 2: # chuyen phai
                    cf.FIXED_SPEED_STRAIGHT = cf.FIXED_SPEED_route_difficult
                    cf.angle = -cf.goc_chuyen_lane
                    cf.lane = 'phai'

                # if self.tick_chuyen_lane is None and abs(angle_from_center) < 16:
                if self.tick_chuyen_lane is None and self.do_not_chuyen_lane is False:
                    self.tick_chuyen_lane = 0
                if self.tick_chuyen_lane is not None:
                    self.tick_chuyen_lane += 1
                
                    # if self.tick_chuyen_lane >= self.tick_start_save_angle:
                    if self.tick_chuyen_lane <= cf.tick_stop_giu_lane:
                        self.angle_giu_lane = -cf.angle*1.05

                    if self.tick_chuyen_lane > cf.tick_stop_chuyen_lane:
                        cf.do_chuyen_lane = -cf.do_chuyen_lane
                        cf.fixed_speed = None
                        self.tick_chuyen_lane = None

                        self.tick_giu_lane = 0
                print('\t\t ----> cf.do_chuyen_lane', cf.do_chuyen_lane, ' | self.tick_chuyen_lane', self.tick_chuyen_lane, cf.lane)

            #############################
            # Giu lane
            #############################
            # elif len(cf.turned) > 6 and cf.do_chuyen_lane == 2 and self.tick_giu_lane is not None: # ket thuc chuyen lane phai o vi tri nhay cam!
            elif self.tick_giu_lane is not None: # ket thuc chuyen lane phai o vi tri nhay cam!
                print('\t\t\t ~~~~~~ self.tick_giu_lane', self.tick_giu_lane, self.angle_giu_lane)
                cf.angle = self.angle_giu_lane

                self.tick_giu_lane += 1
                if self.tick_giu_lane > cf.tick_stop_giu_lane:
                    self.tick_giu_lane = None

                    if cf.do_chuyen_lane == -1:
                        # tang toc de vuot
                        cf.FIXED_SPEED_STRAIGHT = cf.SPEED_PASS_OBJ

            else:
                if cf.current_route == 'thang':
                    if cf.tick_to_pass_turn is not None:
                        print('\t\t ~~~~~~ cf.tick_to_pass_turn', cf.tick_to_pass_turn)
                        # cho nay khong the re phai or re trai, neu phat hien goc > 17, ep goc ve 0
                        if abs(angle_from_center) > 16:
                            angle_from_center = 0
                        
                        cf.tick_to_pass_turn += 1
                        if cf.tick_to_pass_turn > cf.max_tick_to_pass_turn:
                            cf.tick_to_pass_turn = None
                            # do lidar scan to detect moving object
                            cf.run_lidar = True
                
                # if cf.current_route is not None:
                if len(cf.turned) > 4:
                    if cf.angle < -25: # turn right
                        cf.angle -= 15
                    elif cf.angle > 25:
                        cf.angle += 15

                cf.angle = angle_from_center
            
            
            if cf.do_chuyen_lane == -1: # da chuyen sang lane trai (cf.lane = 'trai')
                if cf.segment_chuyen_lane == 1:
                    if len(cf.turned) == 5:
                        if cf.pass_object is True:
                            cf.do_chuyen_lane = 2 # chuyen phai
                        else: # chua vuot duoc vat
                            self.wait_pass_2_turn_to_chuyen_lane = True
                    elif len(cf.turned) == 6: # chuyen ve di thoi
                        if cf.pass_object is True:
                            cf.do_chuyen_lane = 2 # chuyen phai
                            self.wait_pass_2_turn_to_chuyen_lane = False # reset
                elif cf.segment_chuyen_lane == 2:
                    if len(cf.turned) == 6 and cf.pass_object is True: # qua doa va vuot vat
                        cf.do_chuyen_lane = 2 # chuyen phai
                elif cf.segment_chuyen_lane == 3:
                    if cf.pass_object is True:
                        cf.do_chuyen_lane = 2 # chuyen phai



    def whereAmI__trai(self):
        cf.imu_angle = -cf.imu_angle # for simplicity
        
        if cf.imu_angle > 90-cf.imu_early-20 and len(cf.turned) == 0: # turn first left
            cf.turned.append('right')
            cf.reduce_speed = 0 # go at normal speed
        if 180+cf.imu_early > cf.imu_angle > 180-cf.imu_early and len(cf.turned) == 1: # turn second left
            cf.turned.append('right')
        if 270+cf.imu_early > cf.imu_angle > 270-cf.imu_early and len(cf.turned) == 2: # turn third left
            cf.turned.append('right')
        if 180+cf.imu_early > cf.imu_angle > 180-cf.imu_early and len(cf.turned) == 3: # turn right at finish of segment 1
            cf.turned.append('left')
            cf.do_detect_sign = True # now need to detect sign
            cf.count_sign_step = 0
            cf.fixed_speed = cf.speed_detect_sign # go really slow to detect sign !!!

        ''' Get to second segment '''
        # route trai
        if cf.current_route == 'trai':
            # After detecting sign successfully, set reduce_speed = 0 to go at normal speed AND set cf.run_lidar = True to detect moving object (set in signRecognize_)
            
            if 90+cf.imu_early > cf.imu_angle > 90-cf.imu_early and len(cf.turned) == 4: # turn right
                cf.turned.append('left')

            if 180+cf.imu_early > cf.imu_angle > 180-cf.imu_early and len(cf.turned) == 5: # turn left
                cf.turned.append('right')
                cf.tick_to_pass_turn = 0

            if 270+cf.imu_early - cf.value_to_start_at_section2 > cf.imu_angle > 270-cf.imu_early - cf.value_to_start_at_section2 and len(cf.turned) == 6: # turn left
                cf.turned.append('right')
            if 360+cf.imu_early - cf.value_to_start_at_section2 > cf.imu_angle > 360-cf.imu_early - cf.value_to_start_at_section2 and len(cf.turned) == 7: # turn left
                cf.turned.append('right')
                self.start_tick_detect_stop = 0

        # route thang
        if cf.current_route == 'thang':
            # After detecting sign and turn successfully, set reduce_speed = 0 to go at normal speed (set in signRecognize_)

            if 90+cf.imu_early > cf.imu_angle > 90-cf.imu_early and len(cf.turned) == 4: # turn right
                cf.turned.append('left')

            if 0+cf.imu_early > cf.imu_angle > 0-cf.imu_early and len(cf.turned) == 5: # turn right
                cf.turned.append('left')
            if -90+cf.imu_early > cf.imu_angle > -90-cf.imu_early and len(cf.turned) == 6: # turn right
                cf.turned.append('left')
            if 0+cf.imu_early > cf.imu_angle > 0-cf.imu_early and len(cf.turned) == 7: # turn left
                cf.turned.append('right')


    def auto_control__trai(self, angle_from_center):
            #############################
            # Di thang doan dau
            #############################
            # just go straight first few ticks
            if cf.start_tick_count is not None:
                if cf.start_tick_count < cf.end_tick_from_start:
                    if cf.start_tick_count == 0:
                        cf.fixed_speed = cf.init_speed
                    cf.start_tick_count += 1
                    cf.angle = 0
                    if cf.start_tick_count % 5 == 1 and cf.fixed_speed is not None and cf.fixed_speed < cf.MAX_SPEED:
                        cf.fixed_speed += 1
                else:
                    cf.start_tick_count = None
                    cf.reduce_speed = 1
                    cf.fixed_speed = None
            
            #############################
            # Chuyen lane 
            #############################
            elif cf.do_chuyen_lane > 0: # khong chuyen lane cho re
                cf.fixed_speed = cf.speed_chuyen_lane
                if cf.do_chuyen_lane == 1: # chuyen trai
                    cf.angle = cf.goc_chuyen_lane
                    cf.lane = 'trai'
                    if len(cf.turned) == 6: # chuyen o doan dau
                        cf.segment_chuyen_lane = 1
                    elif len(cf.turned) == 7:
                        cf.segment_chuyen_lane = 2
                    elif len(cf.turned) == 8:
                        cf.segment_chuyen_lane = 3
                    
                elif cf.do_chuyen_lane == 2: # chuyen phai
                    cf.FIXED_SPEED_STRAIGHT = cf.FIXED_SPEED_route_difficult
                    cf.angle = -cf.goc_chuyen_lane
                    cf.lane = 'phai'

                # if self.tick_chuyen_lane is None and abs(angle_from_center) < 16:
                if self.tick_chuyen_lane is None and self.do_not_chuyen_lane is False:
                        self.tick_chuyen_lane = 0
                if self.tick_chuyen_lane is not None:
                    self.tick_chuyen_lane += 1
                
                    if self.tick_chuyen_lane <= cf.tick_stop_giu_lane:
                        self.angle_giu_lane = -cf.angle*1.05

                    if self.tick_chuyen_lane > cf.tick_stop_chuyen_lane:
                        cf.do_chuyen_lane = -cf.do_chuyen_lane
                        cf.fixed_speed = None
                        self.tick_chuyen_lane = None

                        self.tick_giu_lane = 0
                print('\t\t ----> cf.do_chuyen_lane', cf.do_chuyen_lane, self.tick_chuyen_lane, cf.lane)

            #############################
            # Giu lane
            #############################
            elif self.tick_giu_lane is not None: # ket thuc chuyen lane phai o vi tri nhay cam!
                print('\t\t\t ~~~~~~ self.tick_giu_lane', self.tick_giu_lane, self.angle_giu_lane)
                cf.angle = self.angle_giu_lane

                self.tick_giu_lane += 1
                if self.tick_giu_lane > cf.tick_stop_giu_lane:
                    self.tick_giu_lane = None

                    if cf.do_chuyen_lane == -1:
                        # tang toc de vuot
                        cf.FIXED_SPEED_STRAIGHT = cf.SPEED_PASS_OBJ

            else:
                # if cf.current_route == 'thang':
                if cf.tick_to_pass_turn is not None:
                    print('\t\t ~~~~~~ cf.tick_to_pass_turn', cf.tick_to_pass_turn)

                    if cf.current_route == 'thang':
                        # cho nay khong the re phai or re trai, neu phat hien goc > 17, ep goc ve 0
                        if abs(angle_from_center) > 16:
                            angle_from_center = 0
                    elif cf.current_route == 'trai':
                        # khong the be phai
                        if angle_from_center < -16:
                            angle_from_center = 0

                    cf.tick_to_pass_turn += 1
                    if cf.tick_to_pass_turn > cf.max_tick_to_pass_turn:
                        cf.tick_to_pass_turn = None
                        # bat lidar
                        if cf.current_route == 'trai':
                            cf.run_lidar = True
                            

                # if cf.current_route is not None:
                if len(cf.turned) > 4:
                    if cf.angle < -25: # turn right
                        cf.angle -= 15
                    elif cf.angle > 25:
                        cf.angle += 15

                # if cf.current_route == 'trai' and len(cf.turned) == 6 and cf.run_lidar is False:
                #     # do lidar scan to detect moving object
                #     cf.tick_to_pass_turn = 0

                if len(cf.turned) == 0: # Doan dau khong bao gio duoc re trai
                    if angle_from_center > 10:
                        angle_from_center = 0

                # O doan cuoi, het tick giu lane, da chuyen sang lane phai thanh cong ma goc con < -16 (van co re phai) thi be ve 0, chi di thang thoi
                if len(cf.turned) == 8 and self.tick_giu_lane is None and cf.do_chuyen_lane == -2 and angle_from_center < -30:
                    angle_from_center = angle_from_center+10

                cf.angle = angle_from_center

            
            if cf.do_chuyen_lane == -1: # da chuyen sang lane trai (cf.lane = 'trai')
                if cf.segment_chuyen_lane == 1:
                    if len(cf.turned) == 7:
                        if cf.pass_object is True:
                            cf.do_chuyen_lane = 2 # chuyen phai
                        else: # chua vuot duoc vat
                            self.wait_pass_2_turn_to_chuyen_lane = True
                    elif len(cf.turned) == 8: # chuyen ve di thoi
                        if cf.pass_object is True:
                            cf.do_chuyen_lane = 2 # chuyen phai
                            self.wait_pass_2_turn_to_chuyen_lane = False # reset
                elif cf.segment_chuyen_lane == 2:
                    if len(cf.turned) == 8 and cf.pass_object is True: # qua doan va vuot vat
                        cf.do_chuyen_lane = 2 # chuyen phai
                elif cf.segment_chuyen_lane == 3:
                    if cf.pass_object is True:
                        cf.do_chuyen_lane = 2 # chuyen phai


    def whereAmI(self):
        if cf.SAN == 'phai':
            self.whereAmI__phai()
        elif cf.SAN == 'trai':
            self.whereAmI__trai()

        if len(cf.turned) == 8:
            if self.start_tick_detect_stop is not None:
                self.start_tick_detect_stop += 1
            else:
                self.start_tick_detect_stop = 0
            
            if (cf.SAN == 'phai' and cf.current_route == 'phai') or (cf.SAN == 'trai' and cf.current_route == 'trai'):
                if self.start_tick_detect_stop >= cf.end_tick_detect_stop:
                    # turn on sign detector to detect stop sign
                    cf.do_detect_stop_sign = True
                    cf.count_sign_step = 0
                    cf.reduce_speed = 1
                    # turn on get_depth_image to calculate distance to stop sign
                    cf.turn_on_depth = True
            elif (cf.SAN == 'trai' and cf.current_route == 'thang') or (cf.SAN == 'phai' and cf.current_route == 'thang'):
                # turn on sign detector to detect stop sign
                cf.do_detect_stop_sign = True
                cf.count_sign_step = 0
                cf.reduce_speed = 1
                # turn on get_depth_image to calculate distance to stop sign
                cf.turn_on_depth = True

        print('[whereAmI]   >> current_route', cf.current_route, ' | run_lidar', cf.run_lidar, ' || do_chuyen_lane', cf.do_chuyen_lane, ' | segment_chuyen_lane', cf.segment_chuyen_lane, ' | cf.pass_object', cf.pass_object, ' | len turned', len(cf.turned), '|| signSignal', cf.signSignal, ' | cf.do_detect_sign', cf.do_detect_sign, ' || cf.do_detect_stop_sign', cf.do_detect_stop_sign, '| stopSignal', cf.stopSignal, '| sign_bbox', cf.sign_bbox, ' | lane', cf.lane, ' | imu_angle', cf.imu_angle, ' | turned', cf.turned, ' | speed', cf.speed, ' | angle', cf.angle, ' | fps_count', cf.fps_count)



    def auto_control(self, angle_from_center):
        # while cf.running: # uncomment if run control in seperate thread
        # if cf.do_stop and not cf.pause:
        #     self.pause()
        if cf.stop_detected and not cf.pause:
            if cf.begin_stop_time is not None and (time.time()-cf.begin_stop_time) > cf.sec_to_stop:
                # print("PAUSING ................................")
                self.pause()
            else:
                cf.fixed_speed = 10
                self.speedControl(cf.angle, cf.fixed_speed)
                # print('SPEED DOWN NOW=',self.speed)
            
        elif cf.ready and not cf.pause: # uncomment if call auto_control inside getCenter
            self.whereAmI()

            if cf.SAN == 'phai':
                self.auto_control__phai(angle_from_center)
            elif cf.SAN == 'trai':
                self.auto_control__trai(angle_from_center)

            self.speedControl(cf.angle, cf.fixed_speed)
            self.set_steer(cf.angle)



class TF_TrtNet(object):
    def __init__(self, output_saved_model_dir, input_tensor_name, output_tensor_name):
        output_saved_model_dir = sys.path[1]+output_saved_model_dir
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)

        # First load the SavedModel into the session    
        tf.saved_model.loader.load(
            self.sess, [tf.saved_model.tag_constants.SERVING],
            output_saved_model_dir)

        self.output_tensor = self.sess.graph.get_tensor_by_name(output_tensor_name)
        self.input_tensor = self.sess.graph.get_tensor_by_name(input_tensor_name)


    def tftrt_predict(self, x):
        output = self.sess.run([self.output_tensor], feed_dict={self.input_tensor: x})
        return output


class SignDetector(object):
    kernel = np.ones((2,2),np.uint8)

    def __init__(self):
        # super(SignDetector, self).__init__()
        self.template = cv2.imread(sys.path[1]+'/template/stop__0.jpg', 1)
        self.template = cv2.resize(self.template, (40,40))
        
        self.load_model()
        return 

    def convertBack(self, x, y, w, h):
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def load_model(self):
        # global cf.metaMain, cf.netMain, cf.altNames
        configPath = sys.path[1]+"/models/yolo/yolov3-tiny.cfg"
        weightPath = sys.path[1]+"/models/yolo/yolov3-tiny.weights"
        metaPath = sys.path[1]+"/models/yolo/sign.data"
        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" +
                            os.path.abspath(configPath)+"`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                            os.path.abspath(weightPath)+"`")
        if not os.path.exists(metaPath):
            raise ValueError("Invalid data file path `" +
                            os.path.abspath(metaPath)+"`")
        if cf.netMain is None:
            cf.netMain = darknet.load_net_custom(configPath.encode(
                "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if cf.metaMain is None:
            cf.metaMain = darknet.load_meta(metaPath.encode("ascii"))
        if cf.altNames is None:
            try:
                with open(metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                    re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                cf.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass
        # Create an image we reuse for each detect
        darknet_image = darknet.make_image(darknet.network_width(cf.netMain),darknet.network_height(cf.netMain),3)
        self.darknet_image = darknet_image
        return darknet_image

    def sign_detect(self, image):
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ## 416 416
        frame_resized = cv2.resize(frame_rgb,
                                (darknet.network_width(cf.netMain),
                                    darknet.network_height(cf.netMain)),
                                interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())
        ## name tag, confidence, box
        # cv2.imwrite(sys.path[1]+'/output/frame_resized.jpg', frame_resized)
        detections = darknet.detect_image(cf.netMain, cf.metaMain, self.darknet_image, thresh=0.8)
        boxes = []
        print('** detections', detections)
        if len(detections) > 0:
            for detection in detections:
                cls_id = detection[0].decode()
                score = detection[1]
                x, y, w, h = detection[2][0],\
                    detection[2][1],\
                    detection[2][2],\
                    detection[2][3]
                xmin, ymin, xmax, ymax = self.convertBack(
                    float(x), float(y), float(w), float(h))
                xmin = xmin*160/darknet.network_width(cf.netMain)
                xmax = xmax*160/darknet.network_width(cf.netMain)
                ymin = ymin*120/darknet.network_height(cf.netMain)
                ymax = ymax*120/darknet.network_height(cf.netMain)
                boxes.append((int(xmin),int(ymin),int(xmax),int(ymax)))
            return cls_id, score, boxes

        return None, None, boxes


    def signRecognize(self, sign_to_detect='normal'):    
        if (cf.current_route is None and cf.do_detect_sign is True) or (cf.current_route is not None and cf.do_detect_stop_sign is True):
            cf.count_sign_step += 1

            if cf.count_sign_step % cf.sign_detect_step == 0:
                img_detect_sign = cf.img_rgb_raw[cf.detect_sign_region['top']:cf.detect_sign_region['bottom'], cf.detect_sign_region['left']:cf.detect_sign_region['right']]

                # cv2.imwrite(sys.path[1]+'/output/detect_sign_area.png', img_detect_sign)
                sign_class_id, score, boxes = self.sign_detect(img_detect_sign)
                # boxes = self.signDetector(img_detect_sign)

                if sign_class_id is None:
                    return None
                if sign_class_id == 'phai' and cf.SAN == 'trai':
                    sign_class_id = "trai"
                elif sign_class_id == 'trai' and cf.SAN == 'phai':
                    sign_class_id = "phai"

                for (x1,y1,x2,y2) in boxes:
                    cf.sign_bbox = (x1,y1,x2,y2)
                    cf.sign.append(sign_class_id)

                    # print('sign_class_id', sign_class_id)
                    sign_region = img_detect_sign[y1:y2, x1:x2]
                    cv2.imwrite(sys.path[1]+"/output/sign_region__{}__{}.jpg".format(cf.count_sign_step, time.time()), sign_region)

                    # if self.acceptSign(sign_class_id):
                    # if y2-y1>30 and x2-x1>30:
                    if (sign_to_detect == 'normal') or (y2-y1>30 and x2-x1>30):
                        if sign_to_detect == 'normal' and sign_class_id != 'dung': # or cf.do_detect_sign = True
                            cf.do_detect_sign = False # stop detecting sign
                            # Sang doan 2, bat dau bam lane
                            cf.lane = 'phai'

                            # After detecting sign successfully, set reduce_speed = 0 to go at normal speed
                            cf.reduce_speed = 0
                            cf.fixed_speed = None

                            # cf.current_route = cf.sign_mapping[sign_class_id]
                            cf.current_route = sign_class_id

                            if cf.SAN == 'phai' and cf.current_route == 'thang': # san phai route kho la route thang
                                cf.FIXED_SPEED_STRAIGHT = cf.FIXED_SPEED_route_difficult # route thang is difficult, set speed lower 
                            elif cf.SAN == 'trai' and cf.current_route == 'trai': # san trai route kho la route trai
                                cf.FIXED_SPEED_STRAIGHT = cf.FIXED_SPEED_route_difficult # route trai is difficult, set speed lower 

                            if cf.current_route == 'thang':
                                cf.tick_to_pass_turn = 0
                        elif sign_class_id == 'dung':
                            cf.do_detect_stop_sign = False

                        return "{}_certain".format(sign_class_id)

                    if sign_to_detect == 'normal' or sign_class_id == 'dung':
                        return sign_class_id

        return None
                    


    def acceptSign(self, value):
        if len(cf.sign) >= 2 and value in cf.sign:
            # ar = np.array(cf.sign + [value])
            # t = ar[ar==value]
            # if len(t) < 2:
            #     return False
            for v in cf.sign:
                if v != value:
                    cf.sign = []
                    return False
            cf.sign = []
            return True
        else:
            return None

class CenterDetector(object):
    def __init__(self):
        return

    def findCenter_drawmore(self):
        # img = cv2.resize(cf.img_rgb_raw, (cf.WIDTH, cf.HEIGHT))
        (h, w) = cf.img_rgb_raw.shape[:2]
        img = cf.img_rgb_raw[cf.crop_top_detect_center:, :]

        # draw where to go, straight or turn
        # on the second half, always draw this to make sure the model does not confuse 2 routes
        img_more = np.zeros(shape=[40, 320, 3], dtype=np.uint8)
        if cf.signSignal is not None:
            if cf.signSignal == 'thang_certain':
                cv2.circle(img_more, org_thang, 20, red, -1)
            if cf.signSignal == 'phai_certain':
                cv2.circle(img_more, org_phai, 20, red, -1)
            if cf.signSignal == 'trai_certain':
                cv2.circle(img_more, org_trai, 20, red, -1)
        # if cf.current_route is not None:
        #     if cf.current_route == 'thang':
        #         cv2.circle(img_more, org_thang, 20, red, -1)
        #     if cf.current_route == 'phai':
        #         cv2.circle(img_more, org_phai, 20, red, -1)
        #     if cf.current_route == 'trai':
        #         cv2.circle(img_more, org_trai, 20, red, -1)

            # draw guide to bam lane (chay nua sau)
            # lane phai
            if cf.lane == 'phai':
                cv2.rectangle(img_more, pts_lanephai[0], pts_lanephai[1], blue, -1)
            # lane trai
            elif cf.lane == 'trai':
                cv2.rectangle(img_more, pts_lanetrai[0], pts_lanetrai[1], blue, -1)

        img = cv2.vconcat([img, img_more])

        output = self.lane_net.tftrt_predict(np.array([img])/255.0)
        center = int(output[0]*img.shape[1])
        angle = self.calAngle(center, w)
        return center, angle


    def findCenter(self):
        # img = cv2.resize(cf.img_rgb_raw, (cf.WIDTH, cf.HEIGHT))
        (h, w) = cf.img_rgb_raw.shape[:2]
        img = cf.img_rgb_raw[cf.crop_top_detect_center:, :]

        # draw where to go, straight or turn
        # on the second half, always draw this to make sure the model does not confuse 2 routes
        if cf.signSignal is not None:
            if cf.signSignal == 'thang_certain':
                cv2.circle(img, org_thang, 20, red, -1)
            if cf.signSignal == 'phai_certain':
                cv2.circle(img, org_phai, 20, red, -1)
            if cf.signSignal == 'trai_certain':
                cv2.circle(img, org_trai, 20, red, -1)
        # if cf.current_route is not None:
        #     if cf.current_route == 'thang':
        #         cv2.circle(img_more, org_thang, 20, red, -1)
        #     if cf.current_route == 'phai':
        #         cv2.circle(img_more, org_phai, 20, red, -1)
        #     if cf.current_route == 'trai':
        #         cv2.circle(img_more, org_trai, 20, red, -1)

            # draw guide to bam lane (chay nua sau)
            # lane phai
            if cf.lane == 'phai':
                cv2.rectangle(img_more, pts_lanephai[0], pts_lanephai[1], blue, -1)
            # lane trai
            elif cf.lane == 'trai':
                cv2.rectangle(img_more, pts_lanetrai[0], pts_lanetrai[1], blue, -1)

        output = self.lane_net.tftrt_predict(np.array([img])/255.0)
        center = int(output[0]*img.shape[1])
        angle = self.calAngle(center, w)
        return center, angle


    def calAngle(self, center, width):
        temp = math.atan(float(abs(float(center)-float(width/2))/float(cf.line_from_bottom)))
        angle = math.degrees(float(temp))
        if center > width/2:
            return -angle
        else:
            return angle

    def calCenterFromAngle(self, angle, width):
        angle_r = math.radians(float(angle))
        center = math.tanh(angle_r)*cf.line_from_bottom + width/2
        return int(center)


class ImageProcessing(AutoControl):
    frame_num = 0

    def __init__(self):
        super(ImageProcessing, self).__init__()
        
        self.signDetector_obj = SignDetector()
        self.centerDetector_obj = CenterDetector() 


    def getCenter(self):
        self.centerDetector_obj.lane_net = TF_TrtNet(cf.lane_model_path, input_tensor_name='input_1:0', output_tensor_name='center/Sigmoid:0')

        start = time.time()
        # fps_count_prev = 0
        while cf.running: # seperate thread
            if cf.got_rgb_image:
                center, angle = self.centerDetector_obj.findCenter_drawmore()

                angle = self.PID(angle, 0.95, 0.0001, 0.02)
                # ###print('[getCenter] predict angle : '+str(angle)+' | after PID : '+str(cf.angle))

                self.auto_control(angle)

                if cf.fps_count > 29 and not cf.ready:
                    cf.ready = True
                    self.print_lcd('Ready!!')
                else:
                    if not cf.ready:
                        print('[getCenter] fps', cf.fps_count)

                seconds = time.time() - start
                # fps_count_prev = cf.fps_count
                # fps_count_now = self.frame_num / seconds
                # cf.fps_count = 0.9*fps_count_prev + 0.1*fps_count_now
                cf.fps_count = self.frame_num / seconds
                self.frame_num += 1

    # def getCenter(self):
    #     self.centerDetector_obj.lane_net = TF_TrtNet(cf.lane_model_path, input_tensor_name='input_1:0', output_tensor_name='center/Sigmoid:0')

    #     fps_count_prev = 0
    #     while cf.running: # seperate thread
    #         if cf.got_rgb_image:
    #             start = time.time()
    #             center, angle = self.centerDetector_obj.findCenter_drawmore()

    #             angle = self.PID(angle, 0.95, 0.0001, 0.02)
    #             # ###print('[getCenter] predict angle : '+str(angle)+' | after PID : '+str(cf.angle))

    #             self.auto_control(angle)

    #             if cf.fps_count > 29 and not cf.ready:
    #                 cf.ready = True
    #                 self.print_lcd('Ready!!')
    #             else:
    #                 if not cf.ready:
    #                     print('[getCenter] fps', cf.fps_count)

    #             seconds = time.time() - start
    #             fps_count_now = 1 / seconds
    #             if fps_count_prev==0:
    #                 cf.fps_count=fps_count_now
    #             else:
    #                 cf.fps_count = 0.9*fps_count_prev + 0.1*fps_count_now
    #             fps_count_prev = cf.fps_count

                

    def getSign(self):
        # self.signDetector_obj.sign_net = TF_TrtNet(cf.sign_model_path, input_tensor_name='input_1:0', output_tensor_name='sign/Softmax:0')
        # darknet_obj = SignDetector()

        while cf.running: # seperate thread
            if cf.got_rgb_image and not cf.pause:
                # cf.do_detect_sign = True
                if cf.do_detect_sign is True:
                    # if len(cf.turned) == 4 and cf.signSignal is None or 'certain' not in cf.signSignal:
                    cf.signSignal = self.signDetector_obj.signRecognize()
                    print('[getSign] cf.signSignal', cf.signSignal)
                    time.sleep(0.005)
                elif cf.do_detect_stop_sign is True:
                    # if len(cf.turned) == 8:
                    cf.stopSignal = self.signDetector_obj.signRecognize(sign_to_detect='stop')
                    # time.sleep(0.01)
                else:
                    time.sleep(0.1)


class Camera(ImageProcessing):
    def __init__(self):
        super(Camera, self).__init__()

        openni2.initialize(sys.path[1]+'/src/modules')
        ###print(sys.path[1]+'/src/modules')
        self.dev = openni2.Device.open_any()

        # rgb stream
        self.rgb_stream = self.dev.create_color_stream()
        self.rgb_stream.set_video_mode(c_api.OniVideoMode(
            pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=640, resolutionY=480, fps=30))
        self.rgb_stream.start()  # start stream

        # depth stream
        self.depth_stream = self.dev.create_depth_stream()
        self.depth_stream.set_video_mode(c_api.OniVideoMode(
            pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX=320, resolutionY=240, fps=30))
        self.depth_stream.start()  # start stream

    def get_rgb(self):
        self.print_lcd("get_rgb called")
        while cf.running:
            ### print('[get_rgb]')
            bgr = np.fromstring(self.rgb_stream.read_frame().get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = rgb[:, ::-1, :]  # flip
            cf.img_rgb_raw = cv2.resize(rgb, (cf.WIDTH, cf.HEIGHT))
            if cf.got_rgb_image is False:
                cf.got_rgb_image = True
                
            # time.sleep(0.001)

        ###print("get_rbg stoped")
    def average_distance(self, arr):
        arr = arr.flatten()
        mean_arr = np.mean(arr)
        std_arr = np.std(arr)
        sum = 0
        number_value = 0
        for value in arr:
            if abs(value-mean_arr) <= std_arr:
                number_value = number_value + 1
                sum = sum + value
            # else:
            #     print(value)
        if number_value == 0:
            return 1
        return sum//number_value

    def get_depth(self):
        while cf.running:
            # print('call get_depth')
            if not cf.ready or len(cf.turned) < 7:
                time.sleep(1.5)
            elif cf.turn_on_depth and not cf.pause:
                print('\n[get_depth]', cf.stopSignal, cf.sign_bbox)
                frame = self.depth_stream.read_frame()
                frame_data = frame.get_buffer_as_uint16()
                img_depth = np.frombuffer(frame_data, dtype=np.uint16)
                img_depth.shape = (cf.HEIGHT, cf.WIDTH)
                img_depth = cv2.flip(img_depth, 1)

                # if stop sign detected
                if cf.stopSignal == 'dung_certain' and cf.sign_bbox is not None:
                    x1,y1,x2,y2 = cf.sign_bbox

                    img_detect_sign = cf.img_rgb_raw[cf.detect_sign_region['top']:cf.detect_sign_region['bottom'], cf.detect_sign_region['left']:cf.detect_sign_region['right']]
                    sign_region = img_detect_sign[y1:y2, x1:x2]
                    print('sign_region.shape', sign_region.shape)
                    # if sign_region.shape[0] > 0 and sign_region.shape[1] > 0:
                    #     cv2.imshow('img_detect_sign', img_detect_sign)
                    #     cv2.imshow('sign_region', sign_region)

                    x1 = x1+12+cf.detect_sign_region['left']
                    x2 = x2+20+cf.detect_sign_region['left']
                    y1 = y1+cf.detect_sign_region['top']
                    y2 = y2+10+cf.detect_sign_region['top']

                    sign_depth = img_depth[y1:y2,x1:x2]
                    # cv2.imshow('img_depth', img_depth)
                    # cv2.imshow('sign_depth', sign_depth)
                    sd = sign_depth.ravel()
                    x = sd[(cf.low_thresh < sd) & (sd < cf.up_thresh)]
                    
                    delta = 3
                    (y2,x2) = sign_depth.shape[:2]
                    x1 = 0
                    y1 = 0
                    center_x = (x1+x2)//2
                    center_y = (y1+y2)//2
                    crop_depth_sign = sign_depth[center_y-delta:center_y+delta, center_x-delta:center_x+delta]

                    print("HERE")
                    print("Crop depth sign", crop_depth_sign)
                    print("--------------")
                    distance = round(np.mean(crop_depth_sign),2)
                    avg_mean = self.average_distance(crop_depth_sign)
                    
                    print("distance", distance)
                    print("mean", avg_mean)
                    
                    if avg_mean <= cf.MAX_DEPTH_DISTANCE:
                        print('SPEED DOWN 10 now')
                        cf.begin_stop_time = time.time()
                        cf.stop_detected = True
                        cf.turn_on_depth = False
                        cf.do_detect_sign = False

                if cf.got_depth_image is False:
                    cf.got_depth_image = True
            # else:
            #     time.sleep(0.12)
            # cv2.waitKey(1)


def sum_arr(Arr, start, end):
    sum_ = 0
    for i in range(start,end+1):
        sum_ = sum_ + Arr[i]
    return sum_

class Lidar(object):

    lidar_data = None
    straight_steer = [356, 5]
    right_steer = [315, 355]
    behind_steer = [180, 270]

    def __init__(self):
        return

    def object_scan(self):
        distance_arr = list(self.lidar_data.ranges)

        if cf.run_lidar and len(distance_arr) > 359:
            # print('\t\t ***** len(distance_arr)', len(distance_arr))
            obj = {}
            count = 0
            i = 0
            thesh_accept = 0.3
            while i < 359:
                while distance_arr[i] > 20:
                    i = i + 1
                start = i
                end = i 
                sum_distance = distance_arr[i]
                while i < 359:
                    if abs(distance_arr[i]-distance_arr[i+1]) < thesh_accept:
                        sum_distance = sum_distance + distance_arr[i+1]
                        i += 1
                    else:
                        if i < 358:
                            if abs(distance_arr[i]-distance_arr[i+2]) < thesh_accept:
                                distance_arr[i+1] = distance_arr[i+2]
                                sum_distance = sum_distance + 2* distance_arr[i+2]
                                i += 2
                            else:
                                break
                        else:
                            break
                if i < 358:
                    distance = sum_distance/(i-start+1)
                    obj[str(count)] = [start, i, distance]
                elif i == 358:
                    if abs(distance_arr[358]-distance_arr[0]) < thesh_accept:
                        end = obj['0'][1]
                        distance_arr[359] = distance_arr[0]
                        sum_distance = sum_distance + distance_arr[0]
                        distance = (sum_distance + sum_arr(distance_arr, start, end))/(361-start+end)
                        obj['0'] = [start, end, distance]
                elif i == 359:
                    if abs(distance_arr[359]-distance_arr[0]) < thesh_accept:
                        end = obj['0'][1]
                        distance = (sum_distance + sum_arr(distance_arr, 0, end))/(361-start+end)
                        obj['0'] = [start, end, distance]
                    elif abs(distance_arr[359]-distance_arr[1]) < thesh_accept:
                        end = obj['0'][1]
                        distance_arr[0] = distance_arr[1]
                        distance = (sum_distance + sum_arr(distance_arr, 0, end))/(361-start+end)
                        obj['0'] = [start, end, distance]
                i += 2
                count += 1
            ###print('[Lidar object_scan] findout OBJ', self.findoutObj(obj, 3.5))
            # found_obj,details = self.findoutObj(obj, 3.5)
            self.process(obj)
        
    def is_object(self, obj, steer):
        # print('obj', obj)
        object_length = obj[1]-obj[0]
        if object_length > steer:
            return True
        else:
            if object_length < 0:
                object_length = 361 - obj[0] + obj[1]
                if object_length > steer:
                    return True
                else:
                    return False
            else:
                return False

    def process(self, obj):
        found_obj = 0
        for key, value in obj.items():
            # print('value', value)
            if self.is_object(value, cf.max_angle_to_detect_lidar) and value[2] < cf.distance_to_detect_lidar:
                # print('\t\t **** found obj in front', value)
                if value[0] < self.straight_steer[1]:
                    found_obj = 1
                    break
                if value[0] < self.straight_steer[0]:
                    if value[1] > self.straight_steer[0] or value[1] < value[0]:
                        found_obj = 1
                        break
                    elif value[1] > self.right_steer[0]:
                        found_obj = 2
                        break
                else:
                    found_obj = 1
                    break
                if value[2] > cf.distance_behind_lidar and value[0] > self.behind_steer[0] and value[1] < self.behind_steer[1] and cf.do_chuyen_lane != -2: # chua chuyen lai ve lane phai
                    found_obj = -1
                    # print('\t\t **** found obj behind', value)
                    break
        if found_obj > 0 and cf.do_chuyen_lane == 0: # phat hien vat o phia truoc va chua chuyen lane trai
            cf.do_chuyen_lane = 1 # chuyen trai
        elif found_obj == -1 and cf.do_chuyen_lane == -1: # phat hien vat o phia sau va dang o lane trai
            cf.pass_object = True
        

class Subscribe(Control, Lidar):
    def __init__(self):
        super(Subscribe, self).__init__()

        self.tick_to_finish__route_thang = time.time()
        self.gyro_z = 0
        self.angle = 0
        self.wz = 0.0
        self.last_gyro_z = 0.0

        return

    def on_get_imu(self, data):
        delta_t = time.time() - cf.t_imu
        cf.t_imu = time.time()

        cf.gyro_z = data.angular_velocity.z
        cf.wz += delta_t*(cf.gyro_z+cf.last_gyro_z)/2.0
        cf.last_gyro_z = cf.gyro_z

        imu_angle = cf.wz*180.0/np.pi
        if cf.first_imu_angle is None:
            cf.first_imu_angle = imu_angle
        cf.imu_angle = - (imu_angle - cf.first_imu_angle)

        time.sleep(0.01)

    def on_get_lidar(self, data):
        self.lidar_data = data
        self.object_scan()
        time.sleep(0.02)

    def on_get_sensor2(self, res):
        if res.data is True and cf.do_detect_barrier is True: # free in front (barrier open)
            self.run_car_by_signal()
            cf.do_detect_barrier = False
            cf.sub_sensor2.unregister()
            time.sleep(0.02)
            cf.sub_sensor2 = None
        if res.data is False and not cf.pause: # something in front closely
            self.pause()
            time.sleep(0.1)
    def on_get_btn_1(self, res):
        '''
        If button 1 is clicked, set mode to start
        '''
        if res.data is True: # click
            cf.do_chuyen_lane = 2
        time.sleep(0.5)
    def on_get_btn_2(self, res):
        '''
        If button 2 is clicked, pause!
        '''
        if res.data is True: # click
            self.pause()
        time.sleep(0.1)
    def on_get_btn_3(self, res):
        '''
        If button 3 is clicked, quit!
        '''
        if res.data is True: # click
            self.quit()
        time.sleep(0.5)

def listenner():
    ros_sub = Subscribe()
    # If button 1 is clicked, set running = True
    if cf.sub_btn1 is not None:
        cf.sub_btn1.unregister()
    if cf.sub_btn2 is not None:
        cf.sub_btn2.unregister()
    if cf.sub_btn3 is not None:
        cf.sub_btn3.unregister()
    # if cf.sub_sensor2 is not None:
    #     cf.sub_sensor2.unregister()

    cf.sub_btn1 = rospy.Subscriber(
        '/bt1_status', Bool, ros_sub.on_get_btn_1, queue_size=1)
    cf.sub_btn2 = rospy.Subscriber(
        '/bt2_status', Bool, ros_sub.on_get_btn_2, queue_size=1)
    cf.sub_btn3 = rospy.Subscriber(
        '/bt3_status', Bool, ros_sub.on_get_btn_3, queue_size=1)
    cf.sub_sensor2 = rospy.Subscriber(
        '/ss2_status', Bool, ros_sub.on_get_sensor2, queue_size=1)
    cf.sub_getIMUAngle = rospy.Subscriber(
        '/imu', Imu, ros_sub.on_get_imu, queue_size=1)
    cf.sub_lidar = rospy.Subscriber(
        '/scan', LaserScan, ros_sub.on_get_lidar, queue_size=1)
    rospy.spin()



class App(Camera, ImageProcessing, HandControl, AutoControl):

    def __init__(self):
        super(App, self).__init__()
        return
    
    def warmup(self):
        ''' Display guidance on LCD '''
        self.clear_lcd()
        listenner()
        # self.print_lcd('1 santrai | 2 sanphai | 3 start')
        # # click button 1: chon santrai
        # cf.sub_btn1 = rospy.Subscriber(
        #     '/bt1_status', Bool, self.on_choose_santrai, queue_size=1)
        # # click button 2: chon sanphai
        # cf.sub_btn2 = rospy.Subscriber(
        #     '/bt2_status', Bool, self.on_choose_sanphai, queue_size=1)
        # # click button 3: start
        # cf.sub_btn2 = rospy.Subscriber(
        #     '/bt2_status', Bool, self.listen_to_start, queue_size=1)
    

    # warmup listening functions
    def on_choose_santrai(self, res):
        '''
        If button 1 is clicked, chon san trai
        '''
        if res.data is True: # click
            cf.SAN = 'trai'
            cf.sign_mapping = {0:'thang', 1:'trai', 2:'dung'}
            # Model
            cf.sign_model_path = "/models/signs/sign_santrai_tf_trt_FP16"
            cf.lane_model_path = "/models/lane_draw/draw_more__320x175__135+40__3_santrai_trt_FP16"
            self.print_lcd('Da chon SAN TRAI')
    def on_choose_sanphai(self, res):
        '''
        If button 1 is clicked, chon san trai
        '''
        if res.data is True: # click
            cf.SAN = 'phai'
            cf.sign_mapping = {0:'thang', 1:'phai', 2:'dung'}
            # Model
            cf.sign_model_path = "/models/signs/sign_sanphai_tf_trt_FP16"
            cf.lane_model_path = "/models/lane_draw/draw_more__320x175__135+40__3_sanphai_trt_FP16"
            self.print_lcd('Da chon SAN PHAI')
    def listen_to_start(self, res):
        if res.data is True:
            self.print_lcd('Pressed start. Opening thread...')
            self.run()



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

        get_sign_thread = threading.Thread(
            name="get_sign_thread", target=self.getSign)
        get_sign_thread.start()

        # using tensorrt cannot run this in seperate thread
        get_center_thread = threading.Thread(
            name="get_center_thread", target=self.getCenter) # Drive control in this function
        get_center_thread.start()

        get_depth_thread = threading.Thread(
            name="get_depth_thread", target=self.get_depth)
        get_depth_thread.start()

        control_thread = threading.Thread(
            name="control", target=self.hand_control)
        control_thread.start()

        listenner()

        get_rgb_thread.join()
        # get_sign_thread.join()
        get_center_thread.join()
        get_depth_thread.join()
        # auto_control_thread.join()
        control_thread.join()


if __name__ == "__main__":
    app = App()
    app.warmup()
