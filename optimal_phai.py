#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants

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

cf.sign_mapping = {0:'thang', 1:'phai', 2:'trai', 3:'dung'}

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
cf.detect_sign_region = {'top':0, 'bottom':240, 'left':320, 'right':640}
cf.fps_count = 0

# cf.SAN = 'phai'
# route = {
#     'thang': ['left', 'left', 'left', 'right', 'right', 'right', 'right', 'left'],
#     'phai': ['left', 'left', 'left', 'right', 'right', 'left', 'left', 'left']
# }
cf.turned = [] # what did I turn
cf.current_route = None
cf.start_tick_count = 0 # If this is set to 0, it will be in running straight mode (with fixed angle=0 and speed=MAX_SPEED) and ignore all other calculation. If set to None, it uses predictCenter to calculate angle and set speed.

# get and set path
rospack = rospkg.RosPack()
path = rospack.get_path('mtapos')
sys.path.insert(1, path)

# init node
rospy.init_node('optimal_phai', anonymous=True, disable_signals=True)


# set img / video variables
cf.HEIGHT = 240  # 480
cf.WIDTH = 320  # 640
# cf.HEIGHT_D = 240 # height of depth image
# cf.WIDTH_D = 320 # width of depth image
cf.img_rgb_raw = np.zeros((640, 480, 3), np.uint8)
cf.img_depth = np.zeros((cf.WIDTH, cf.HEIGHT), np.uint8)
cf.depth_processed = np.zeros((cf.WIDTH, cf.HEIGHT), np.uint8)
cf.rgb_viz = np.zeros((cf.WIDTH, cf.HEIGHT), np.uint8)

# Model
cf.sign_model_path = "/models/signs/sign_sanphai_tf_trt_FP16"
cf.lane_model_path = "/models/lane_draw/draw_more__320x175__135+40__dat_3_santrai_trt_FP16"

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
cf.FIXED_SPEED_NORMAL = 19
cf.FIXED_SPEED_STRAIGHT = cf.FIXED_SPEED_NORMAL
cf.FIXED_SPEED_route_difficult = 15
cf.SPEED_PASS_OBJ = 18
cf.FIXED_SPEED_TURN = 15
cf.end_tick_from_start = 47 #20(for 17,14) # 22(for 14,12)
cf.MAX_SPEED = 23
cf.MIN_SPEED = 14
cf.reduce_speed = 0 # 0: normal | 1: reduce speed | -1: speed up
cf.speed_reduced = 14
cf.imu_early = 15
cf.speed_increasement = 1
cf.fixed_speed = 13 # init speed to increase by time
cf.speed_chuyen_lane = 14 # 13

# detect chuyen lane
cf.lane = None
# Tin hieu chuyen lane:
#    0: chua chuyen, ko chuyen 
#    1: nhan tin hieu chuyen tu PHAI -> TRAI
#    2: nhan tin hieu chuyen tu TRAI -> PHAI
#   -1: da chuyen tu PHAI -> TRAI, ko bao h chuyen P->T nua
#   -2: da chuyen tu TRAI -> PHAI, ko bao h chuyen T->P nua
cf.do_chuyen_lane = 0
cf.tick_stop_chuyen_lane = 16
cf.goc_chuyen_lane = 35
cf.run_lidar = False
cf.distance_to_detect_lidar = 3.5 # 3.5
cf.distance_behind_lidar = 1.2
cf.max_angle_to_detect_lidar = 2.3
cf.tick_stop_giu_lane = 6
cf.segment_chuyen_lane = 0
cf.pass_object = False

# variables for sign detection
cf.sign = []
cf.sign_detect_step = 2
cf.do_detect_sign = False
cf.signSignal = None
cf.count_sign_step = 0
cf.tick_to_pass_turn = None
cf.max_tick_to_pass_turn = 40

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
    ###print("Reset MPU")
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
                ###print('\t set_speed', speed)
                cf.speed_pub.publish(speed)

    def set_steer(self, steer):
        if steer == 0 or (cf.running and not cf.pause and cf.ready):
            ###print('\t set_angle', steer)
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
        cf.turned = []
        reset_imu()
        cf.first_imu_angle = None
        cf.start_tick_count = 0
        cf.FIXED_SPEED_STRAIGHT = cf.FIXED_SPEED_NORMAL
        cf.do_chuyen_lane = 0
        cf.segment_chuyen_lane = 0
        cf.pass_object = False
        cf.reduce_speed = 0
        cf.lane = None
        cf.run_lidar = False
        self.run_car_by_signal()


    def run_car_by_signal(self):
        '''
        Called when a signal is sent to start run the car.
        But car is run only when everything is loaded.
        cf.ready makes sure of that
        '''
        if cf.pause is True:
            cf.pause = False
            ###print('Send signal to run', cf.ready)
            self.print_lcd('Send signal to run')
            while cf.ready is False:
                ###print('Received signal to run but not ready! Wait a moment...')
                self.print_lcd("Wait a moment...")
                if cf.ready is True:
                    break
            # if cf.ready is True:
            #     self.run_car_after_check()

    # def run_car_after_check(self):
    #     self.print_lcd('Ready! Run now!!')
    #     cf.speed = cf.FIXED_SPEED_STRAIGHT
    #     self.set_speed(cf.speed)

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
            ###print('Unsubscribe button1')
        if cf.sub_btn2 is not None:
            cf.sub_btn2.unregister()
            ###print('Unsubscribe button2')
        if cf.sub_btn3 is not None:
            cf.sub_btn3.unregister()
            ###print('Unsubscribe button3')
        if cf.sub_sensor2 is not None:
            cf.sub_sensor2.unregister()
            ###print('Unsubscribe sensor2')

        if cf.sub_lidar is not None:
            cf.sub_lidar.unregister()
            ###print('Unsubscribe lidar')
        if cf.sub_getIMUAngle is not None:
            cf.sub_getIMUAngle.unregister()
            ###print('Unsubscribe IMU')

        cv2.destroyAllWindows()
        ###print('Close cv2 windows')
        # self.print_lcd('Quit!')
        # time.sleep(0.1)
        self.clear_lcd()
        time.sleep(1)
        ###print('Quit')
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
                if cf.pause is True:
                    self.reset_all()
            if key.char == 'q':  # quit
                self.pause()
            if key.char == 'z':  # chuyen lane
                if cf.lane == 'trai':
                    cf.do_chuyen_lane = 2 # chuyen tu trai sang phai
                elif cf.lane == 'phai':
                    cf.do_chuyen_lane = 1 # chuyen tu phai sang trai

        except AttributeError:
            # ###print('special key {0} pressed'.format(key))
            ''' Control speed '''
            if key == keyboard.Key.up:
                ###print('cf.running, cf.pause', cf.running, cf.pause)
                if cf.running and not cf.pause:
                    cf.FIXED_SPEED_STRAIGHT += cf.speed_increasement
                    cf.FIXED_SPEED_STRAIGHT = min(cf.FIXED_SPEED_STRAIGHT, cf.MAX_SPEED)
            if key == keyboard.Key.down:
                if cf.running and not cf.pause:
                    cf.FIXED_SPEED_STRAIGHT -= cf.speed_increasement
                    cf.FIXED_SPEED_STRAIGHT = max(cf.FIXED_SPEED_STRAIGHT, cf.MIN_SPEED)

            if key == keyboard.Key.esc:
                self.quit()

class AutoControl(Control):
    arr_speed = np.zeros(5)
    timespeed = time.time()
    tick_to_finish__route_thang = None
    fixed_speed = cf.fixed_speed # init speed to increase by time
    tick_chuyen_lane = None
    tick_giu_lane = None
    angle_giu_lane = None
    last_angles_chuyen_lane = []
    tick_start_save_angle = cf.tick_stop_chuyen_lane - cf.tick_stop_giu_lane
    do_not_chuyen_lane = False
    wait_pass_2_turn_to_chuyen_lane = False
    
    def __init__(self):
        super(AutoControl, self).__init__()
        
        return

    def speedControl(self, angle, fixed_speed=None):
        tempangle = abs(angle)
        if fixed_speed is not None:
            speed = fixed_speed
            ###print('\t\t fixed_speed', fixed_speed)
        elif cf.reduce_speed == 1:
            speed = cf.speed_reduced
            ###print('\t\t Reduce speed !!!!!!!!!!!!!')
        elif cf.reduce_speed == -1:
            speed = cf.MAX_SPEED
            ###print('\t\t Speed up !!!!!!!!!!!!!')
        else: # cf.reduce_speed == 0
            if abs(angle) > 18:
                speed = cf.FIXED_SPEED_TURN
            else:
                speed = cf.FIXED_SPEED_STRAIGHT

        cf.speed = speed
        self.set_speed(cf.speed)
        # time.sleep(0.02)

    def whereAmI(self):
        if cf.imu_angle > 90-cf.imu_early-20 and len(cf.turned) == 0: # turn first left
            cf.turned.append('left')
            cf.reduce_speed = 0 # go at normal speed
        if 180+cf.imu_early > cf.imu_angle > 180-cf.imu_early and len(cf.turned) == 1: # turn second left
            cf.turned.append('left')
        if 270+cf.imu_early > cf.imu_angle > 270-cf.imu_early and len(cf.turned) == 2: # turn third left
            cf.turned.append('left')
            # cf.reduce_speed = 1 # need to reduce speed to turn the next right and detect sign
        if 180+cf.imu_early > cf.imu_angle > 180-cf.imu_early and len(cf.turned) == 3: # turn right at finish of segment 1
            cf.turned.append('right')
            cf.do_detect_sign = True # now need to detect sign
            cf.count_sign_step = 0
            self.fixed_speed = 10 # go really slow to detect sign !!!

        if cf.current_route is not None and cf.start_tick_count is None: # detect sign successfully!
            self.fixed_speed = None


        ''' Get to second segment '''
        # route phai
        if cf.current_route == 'phai':
            # After detecting sign and turn successfully, set reduce_speed = 0 to go at normal speed (set in signRecognize)
            
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

            if len(cf.turned) == 8: # after some tick_to_finish__route_thang we would need to slow down to detect stop sign
                self.tick_to_finish__route_thang += 1
                if self.tick_to_finish__route_thang > 50:
                    cf.reduce_speed = 1

        # route thang
        if cf.current_route == 'thang':
            # After detecting sign successfully, set reduce_speed = 0 to go at normal speed AND set cf.run_lidar = True to detect moving object (set in signRecognize)

            if 90+cf.imu_early > cf.imu_angle > 90-cf.imu_early and len(cf.turned) == 4: # turn right
                cf.turned.append('right')

            if 0+cf.imu_early > cf.imu_angle > 0-cf.imu_early and len(cf.turned) == 5: # turn right
                cf.turned.append('right')
            if -90+cf.imu_early > cf.imu_angle > -90-cf.imu_early and len(cf.turned) == 6: # turn right
                cf.turned.append('right')
            if 0+cf.imu_early > cf.imu_angle > 0-cf.imu_early and len(cf.turned) == 7: # turn left
                cf.turned.append('left')

        if len(cf.turned) == 8:
            # turn on sign detector
            cf.do_detect_sign = True 
            cf.count_sign_step = 0

        print('   >> ', cf.current_route, cf.lane, cf.imu_angle, cf.turned, cf.speed, cf.angle, cf.fps_count)


    def auto_control(self, angle_from_center):
        # while cf.running: # uncomment if run control in seperate thread
        if cf.ready and not cf.pause: # uncomment if call auto_control inside getCenter

            self.whereAmI()

            # just go straight first few ticks
            if cf.start_tick_count is not None:
                if cf.start_tick_count < cf.end_tick_from_start:
                    if cf.start_tick_count == 0:
                        self.fixed_speed = cf.fixed_speed
                    cf.start_tick_count += 1
                    cf.angle = 0
                    # cf.speed = cf.MAX_SPEED
                    # cf.reduce_speed = -1
                    if cf.start_tick_count % 5 == 1 and self.fixed_speed < cf.MAX_SPEED:
                        self.fixed_speed += 1
                    ###print('cf.start_tick_count', cf.start_tick_count, '~~~~ cf.end_tick_from_start', cf.end_tick_from_start)
                else:
                    cf.start_tick_count = None
                    cf.reduce_speed = 1
                    self.fixed_speed = None
            
            elif cf.do_chuyen_lane > 0: # khong chuyen lane cho re
                self.fixed_speed = cf.speed_chuyen_lane
                if cf.do_chuyen_lane == 1: # chuyen trai
                    # cf.angle = angle_from_center + cf.goc_chuyen_lane
                    cf.angle = cf.goc_chuyen_lane
                    cf.lane = 'trai'
                    if len(cf.turned) == 4: # chuyen o doan dau
                        cf.segment_chuyen_lane = 1
                    elif len(cf.turned) == 5:
                        cf.segment_chuyen_lane = 2
                    elif len(cf.turned) == 6:
                        cf.segment_chuyen_lane = 3
                    
                elif cf.do_chuyen_lane == 2: # chuyen phai
                    cf.FIXED_SPEED_STRAIGHT = cf.FIXED_SPEED_route_difficult
                    # cf.angle = angle_from_center - cf.goc_chuyen_lane
                    cf.angle = -cf.goc_chuyen_lane
                    cf.lane = 'phai'

                # if self.tick_chuyen_lane is None and abs(angle_from_center) < 16:
                if self.tick_chuyen_lane is None and self.do_not_chuyen_lane is False:
                        self.tick_chuyen_lane = 0
                if self.tick_chuyen_lane is not None:
                    self.tick_chuyen_lane += 1
                
                    # if self.tick_chuyen_lane >= self.tick_start_save_angle:
                    if self.tick_chuyen_lane <= cf.tick_stop_giu_lane:
                        self.last_angles_chuyen_lane.append(-cf.angle)
                        # print('self.last_angles_chuyen_lane', cf.angle, self.last_angles_chuyen_lane)

                        self.angle_giu_lane = -cf.angle*1.2

                    if self.tick_chuyen_lane > cf.tick_stop_chuyen_lane:
                        cf.do_chuyen_lane = -cf.do_chuyen_lane
                        self.fixed_speed = None
                        self.tick_chuyen_lane = None

                        self.tick_giu_lane = 0
                print('\t\t ----> cf.do_chuyen_lane', cf.do_chuyen_lane, self.tick_chuyen_lane, cf.lane)

            # elif len(cf.turned) > 6 and cf.do_chuyen_lane == 2 and self.tick_giu_lane is not None: # ket thuc chuyen lane phai o vi tri nhay cam!
            elif self.tick_giu_lane is not None: # ket thuc chuyen lane phai o vi tri nhay cam!
                print('\t\t\t ~~~~~~ self.tick_giu_lane', self.tick_giu_lane, self.last_angles_chuyen_lane)
                cf.angle = self.angle_giu_lane
                # cf.angle = self.last_angles_chuyen_lane[self.tick_giu_lane]

                self.tick_giu_lane += 1
                if self.tick_giu_lane > cf.tick_stop_giu_lane:
                    self.tick_giu_lane = None
                    self.last_angles_chuyen_lane = []

                    if cf.do_chuyen_lane == -1:
                        # tang toc de vuot
                        cf.FIXED_SPEED_STRAIGHT = cf.SPEED_PASS_OBJ

            else:
                if cf.current_route == 'thang':
                    # if len(cf.turned) == 3: # turn third left
                    #     # cho nay khong the re trai, neu phat hien goc > 16, ep goc ve -angle
                    #     if angle_from_center > 16:
                    #         angle_from_center = -angle_from_center
                    #         ###print('ep goc!!', angle_from_center)

                    # elif len(cf.turned) == 7: # last turn left of route thang
                    #     # cho nay khong the re phai, neu phat hien goc < -30, ep goc ve -angle
                    #     if angle_from_center < -36:
                    #         angle_from_center = -angle_from_center
                    #         print('ep goc khuc cuoi!!', angle_from_center)

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

                
                cf.angle = angle_from_center
            
            
            if cf.do_chuyen_lane == -1: # da chuyen sang lane trai (cf.lane = 'trai')
                if cf.segment_chuyen_lane == 1:
                    if len(cf.turned) == 5 and self.wait_pass_2_turn_to_chuyen_lane is False:
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


            self.speedControl(cf.angle, self.fixed_speed)
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
        
        return

    def signDetector(self, img, color="blue"):
        img_height, img_width = img.shape[:2]
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        
        # blue
        # lower = np.array([90,80,50]) 
        # upper = np.array([120,255,255]) # 100, 97, 45
        lower = np.array([0, 200, 0]) 
        upper = np.array([179,255,255]) # 179,255,255

        # red
        if color == 'red':
            lower = np.array([0, 100, 0]) 
            upper = np.array([18,255,255]) # 100, 97, 45
        
        mask = cv2.inRange(hsv, lower, upper)

        dilation = cv2.dilate(mask, self.kernel,iterations = 1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, self.kernel)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, self.kernel)

        edge = cv2.Canny(closing, 175, 175)
        contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # contour = contours[0]
        bnd_ar = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if 200 > w >= 40 and 200 > h >= 40 and abs(w-h) < 10:   
                bnd_ar.append((x, y, x+w, y+h))         
                # return x, y, x+w, y+h
        
        return bnd_ar
        

    def signRecognize(self, sign_to_detect='normal'):
        if cf.do_detect_sign:
            cf.count_sign_step += 1

            if cf.count_sign_step % cf.sign_detect_step == 0:
                img_detect_sign = cf.img_rgb_raw[cf.detect_sign_region['top']:cf.detect_sign_region['bottom'], cf.detect_sign_region['left']:cf.detect_sign_region['right']]
                
                cf.signSignal = None
                if sign_to_detect == 'normal':
                    boxes = self.signDetector(img_detect_sign)
                else: # detect stop sign
                    boxes = self.signDetector(img_detect_sign, 'red')

                if len(boxes) == 0:
                    return None

                for (x1,y1,x2,y2) in boxes:
                    sign_region = img_detect_sign[y1:y2, x1:x2]
                    img = cv2.resize(sign_region, (100,100))

                    output = self.sign_net.tftrt_predict(np.array([img])/255.0)[0][0]

                    sign_class_id = np.argmax(output)
                    prob = output[sign_class_id]

                    ###print(sign_class_id, prob, output)
                    if prob > 0.75:
                        cf.sign.append(sign_class_id)

                        if self.acceptSign(sign_class_id):
                            cf.do_detect_sign = False # stop detecting sign

                            if sign_to_detect == 'normal':
                                # Sang doan 2, bat dau bam lane
                                cf.lane = 'phai'

                                # After detecting sign successfully, set reduce_speed = 0 to go at normal speed
                                cf.reduce_speed = 0

                                cf.current_route = cf.sign_mapping[sign_class_id]
                                if cf.current_route == 'thang':
                                    cf.FIXED_SPEED_STRAIGHT = cf.FIXED_SPEED_route_difficult # route thang is difficult, set speed lower 
                                    cf.tick_to_pass_turn = 0

                            if sign_class_id == 0: # thang
                                return "thang_certain"
                            elif sign_class_id == 1: # phai
                                return "phai_certain"
                            elif sign_class_id == 2: # trai
                                return "trai_certain"
                            elif sign_class_id == 3: # stop
                                return "dung_certain"

                        return cf.sign_mapping[sign_class_id]

        return None

    def acceptSign(self, value):
        if len(cf.sign) >= 2:
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
        img = cv2.resize(cf.img_rgb_raw, (cf.WIDTH, cf.HEIGHT))
        (h, w) = img.shape[:2]
        img = img[cf.crop_top_detect_center:, :]

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
        img = cv2.resize(cf.img_rgb_raw, (cf.WIDTH, cf.HEIGHT))
        (h, w) = img.shape[:2]
        img = img[cf.crop_top_detect_center:, :]

        # draw where to go, straight or turn
        # on the second half, always draw this to make sure the model does not confuse 2 routes
        if cf.signSignal is not None:
            if cf.signSignal == 'thang_certain':
                cv2.circle(img, org_thang, 20, red, -1)
            if cf.signSignal == 'phai_certain':
                cv2.circle(img, org_phai, 20, red, -1)
            if cf.signSignal == 'trai_certain':
                cv2.circle(img, org_trai, 20, red, -1)

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

                print('[getCenter] fps', cf.fps_count)
                if cf.fps_count > 25:
                    cf.ready = True
                    self.print_lcd('Ready!!')

                seconds = time.time() - start
                # fps_count_prev = cf.fps_count
                # fps_count_now = self.frame_num / seconds
                # cf.fps_count = 0.9*fps_count_prev + 0.1*fps_count_now
                cf.fps_count = self.frame_num / seconds
                self.frame_num += 1


    def getSign(self):
        self.signDetector_obj.sign_net = TF_TrtNet(cf.sign_model_path, input_tensor_name='input_1:0', output_tensor_name='sign/Softmax:0')

        while cf.running: # seperate thread
            if cf.got_rgb_image:
                if cf.do_detect_sign:
                    if cf.signSignal is None or 'certain' not in cf.signSignal:
                        cf.signSignal = self.signDetector_obj.signRecognize()

                        print('[getSign] cf.signSignal', cf.signSignal)
                        time.sleep(0.02)
                    else:
                        cf.stopSignal = self.signDetector_obj.signRecognize('stop')
                        time.sleep(0.02)
                else:
                    time.sleep(0.3)


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
            bgr = np.fromstring(self.rgb_stream.read_frame().get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            cf.img_rgb_raw = rgb[:, ::-1, :]  # flip
            if cf.got_rgb_image is False:
                cf.got_rgb_image = True
                
            # print('[get_rgb]')
            # time.sleep(0.001)

        ###print("get_rbg stoped")

    def get_depth(self):
        while cf.running:
            frame = self.depth_stream.read_frame()
            frame_data = frame.get_buffer_as_uint16()
            img_depth = np.frombuffer(frame_data, dtype=np.uint16)
            img_depth.shape = (cf.HEIGHT, cf.WIDTH)
            cf.img_depth = img_depth
            if cf.got_depth_image is False:
                cf.got_depth_image = True
        ###print("get_depth stopped")


def sumArr(Arr, start, end):
    sum_ = 0
    for i in range(start,end+1):
        sum_ = sum_ + Arr[i]
    return sum_

class Lidar(object):

    lidar_data = None
    straightSteer = [356, 5]
    rightSteer = [315, 355]
    behindSteer = [180, 270]

    def __init__(self):
        return

    def objectScan(self):
        distanceArr = list(self.lidar_data.ranges)

        if cf.run_lidar and len(distanceArr) > 359:
            # print('\t\t ***** len(distanceArr)', len(distanceArr))
            obj = {}
            count = 0
            i = 0
            theshAccept = 0.3
            while i < 359:
                while distanceArr[i] > 20:
                    i = i + 1
                start = i
                end = i 
                tong = distanceArr[i]
                while i < 359:
                    if abs(distanceArr[i]-distanceArr[i+1]) < theshAccept:
                        tong = tong + distanceArr[i+1]
                        i += 1
                    else:
                        if i < 358:
                            if abs(distanceArr[i]-distanceArr[i+2]) < theshAccept:
                                distanceArr[i+1] = distanceArr[i+2]
                                tong = tong + 2* distanceArr[i+2]
                                i += 2
                            else:
                                break
                        else:
                            break
                if i < 358:
                    distance = tong/(i-start+1)
                    obj[str(count)] = [start, i, distance]
                elif i == 358:
                    if abs(distanceArr[358]-distanceArr[0]) < theshAccept:
                        end = obj['0'][1]
                        distanceArr[359] = distanceArr[0]
                        tong = tong + distanceArr[0]
                        distance = (tong + sumArr(distanceArr, start, end))/(361-start+end)
                        obj['0'] = [start, end, distance]
                elif i == 359:
                    if abs(distanceArr[359]-distanceArr[0]) < theshAccept:
                        end = obj['0'][1]
                        distance = (tong + sumArr(distanceArr, 0, end))/(361-start+end)
                        obj['0'] = [start, end, distance]
                    elif abs(distanceArr[359]-distanceArr[1]) < theshAccept:
                        end = obj['0'][1]
                        distanceArr[0] = distanceArr[1]
                        distance = (tong + sumArr(distanceArr, 0, end))/(361-start+end)
                        obj['0'] = [start, end, distance]
                i += 2
                count += 1
            ###print('[Lidar objectScan] findout OBJ', self.findoutObj(obj, 3.5))
            # foundObj,details = self.findoutObj(obj, 3.5)
            self.process(obj)
        
    def checkObject(self, obj, steer):
        # print('obj', obj)
        objectLength = obj[1]-obj[0]
        if objectLength > steer:
            return True
        else:
            if objectLength < 0:
                objectLength = 361 - obj[0] + obj[1]
                if objectLength > steer:
                    return True
                else:
                    return False
            else:
                return False

    def process(self, obj):
        foundObj = 0
        for key, value in obj.items():
            # print('value', value)
            if self.checkObject(value, cf.max_angle_to_detect_lidar) and value[2] < cf.distance_to_detect_lidar:
                # print('\t\t **** found obj in front', value)
                if value[0] < self.straightSteer[1]:
                    foundObj = 1
                    break
                if value[0] < self.straightSteer[0]:
                    if value[1] > self.straightSteer[0] or value[1] < value[0]:
                        foundObj = 1
                        break
                    elif value[1] > self.rightSteer[0]:
                        foundObj = 2
                        break
                else:
                    foundObj = 1
                    break
                if value[2] > cf.distance_behind_lidar and value[0] > self.behindSteer[0] and value[1] < self.behindSteer[1] and cf.do_chuyen_lane != -2: # chua chuyen lai ve lane phai
                    foundObj = -1
                    # print('\t\t **** found obj behind', value)
                    break
        if foundObj > 0 and cf.do_chuyen_lane == 0: # phat hien vat o phia truoc va chua chuyen lane trai
            cf.do_chuyen_lane = 1 # chuyen trai
        elif foundObj == -1 and cf.do_chuyen_lane == -1: # phat hien vat o phia sau va dang o lane trai
            # cf.do_chuyen_lane = 2 # chuyen phai
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

    def on_get_lidar(self, data):
        self.lidar_data = data
        self.objectScan()
        time.sleep(0.05)

    def on_get_sensor2(self, res):
        if res.data is True and cf.do_detect_barrier is True: # free in front (barrier open)
            self.run_car_by_signal()
            cf.do_detect_barrier = False
            cf.sub_sensor2.unregister()
            time.sleep(0.02)
            cf.sub_sensor2 = None
        if res.data is False and not cf.pause: # something in front closely
            self.pause()
        # time.sleep(0.2)
    def on_get_btn_1(self, res):
        '''
        If button 1 is clicked, set mode to start
        '''
        if res.data is True: # click
            # self.print_lcd('Running!')
            # ###print('Running!')
            # cf.running = True
            # if cf.sub_sensor2 is None:
            #     self.run_car_by_signal()
            # else:
            #     cf.do_detect_barrier = True
            cf.do_chuyen_lane = 2
        time.sleep(0.2)
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
        time.sleep(0.2)

def listenner():
    ros_sub = Subscribe()
    # If button 1 is clicked, set running = True
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

        # get_depth_thread = threading.Thread(
        #     name="get_depth_thread", target=self.get_depth)
        # get_depth_thread.start()

        control_thread = threading.Thread(
            name="control", target=self.hand_control)
        control_thread.start()

        listenner()

        get_rgb_thread.join()
        # get_sign_thread.join()
        get_center_thread.join()
        # get_depth_thread.join()
        # auto_control_thread.join()
        control_thread.join()


if __name__ == "__main__":
    app = App()
    app.run()
