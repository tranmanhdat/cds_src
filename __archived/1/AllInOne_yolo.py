#!/usr/bin/env python3

import tensorflow as tf
from keras.models import load_model
from tensorflow import keras
import keras.backend as K
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
from yolo import YOLO
from PIL import Image


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
cf.lane_model_path = "/models/lane/dithang_320x140.h5"
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
cf.FIXED_SPEED = 15
cf.FIXED_SPEED_TURN = 14
cf.end_tick_from_start = 5
 #20(for 17,14) # 22(for 14,12)
cf.MAX_SPEED = 20
cf.MIN_SPEED = 10
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
            

            # if speed != cf.speed:
            if (not cf.pause and cf.ready) or speed == 0:
                print('\t set_speed', speed)
                cf.speed_pub.publish(speed)
                # rospy.loginfo('Published')
            # cf.rate.sleep()

    def set_steer(self, steer):
        if steer == 0 or (cf.running and not cf.pause and cf.ready):
            cf.change_steer = True
            print('\t set_angle', steer)
            cf.steer_pub.publish(steer)

    def set_lcd(self, text):
        cf.lcd_pub.publish(text)

    def clear_lcd(self):
        # clear lcd
        for row in range(4):
            self.set_lcd("0:{}:{}".format(row, ' '*20))

    def print_lcd(self, text):
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

    def run_car_by_signal(self):
        '''
        Called when a signal is sent to start run the car.
        But car is run only when everything is loaded.
        cf.ready makes sure of that
        '''
        if cf.pause is True:
            cf.pause = False
            print('Send signal to run', cf.ready)
            self.print_lcd('Send signal to run')
            while cf.ready is False:
                print('Received signal to run but not ready! Wait a moment...')
                self.print_lcd("Wait a moment...")
                if cf.ready is True:
                    break
            if cf.ready is True:
                self.run_car_after_check()

    def run_car_after_check(self):
        print('Ready! Run now!!')
        self.print_lcd('Ready! Run now!!')
        cf.speed = cf.FIXED_SPEED
        self.set_speed(cf.speed)

    def pause(self):
        print('Pause!')
        # time_to_sleep = cf.speed*0.5/15
        # # before pause set speed to its negative value to go backwards
        # cf.speed = -cf.speed
        # self.set_speed(cf.speed)
        cf.pause = True
        # time.sleep(time_to_sleep)
        # and = 0 to stop
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
            print('Unsubscribe button1')
        if cf.sub_btn2 is not None:
            cf.sub_btn2.unregister()
            print('Unsubscribe button2')
        if cf.sub_btn3 is not None:
            cf.sub_btn3.unregister()
            print('Unsubscribe button3')
        if cf.sub_sensor2 is not None:
            cf.sub_sensor2.unregister()
            print('Unsubscribe sensor2')

        if cf.sub_lidar is not None:
            cf.sub_lidar.unregister()
            print('Unsubscribe lidar')
        if cf.sub_getIMUAngle is not None:
            cf.sub_getIMUAngle.unregister()
            print('Unsubscribe IMU')

        cv2.destroyAllWindows()
        print('Close cv2 windows')
        # self.print_lcd('Quit!')
        # time.sleep(0.1)
        self.clear_lcd()
        time.sleep(1)
        print('QUit')
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
                print('Start!')
                # set_lcd('Start!')
                if cf.pause is True:
                    cf.ready = True
                    self.run_car_by_signal()
            if key.char == 'q':  # quit
                self.pause()
            if key.char == 'z':  # toggle saving images
                cf.save_image = not cf.save_image
        except AttributeError:
            # print('special key {0} pressed'.format(key))
            ''' Control speed '''
            if key == keyboard.Key.up:
                print('cf.running, cf.pause', cf.running, cf.pause)
                if cf.running and not cf.pause:
                    cf.FIXED_SPEED += cf.speed_increasement
                    cf.FIXED_SPEED = min(cf.FIXED_SPEED, cf.MAX_SPEED)
                    # self.set_speed(cf.speed)
            if key == keyboard.Key.down:
                if cf.running and not cf.pause:
                    cf.FIXED_SPEED -= cf.speed_increasement
                    cf.FIXED_SPEED = max(cf.FIXED_SPEED, cf.MIN_SPEED)
                    # self.set_speed(cf.speed)

            ''' Control steer '''
            # if key == keyboard.Key.right:
            #     if cf.running and not cf.pause:
            #         cf.angle -= cf.angle_increasement
            #         cf.angle = max(cf.angle, cf.MIN_ANGLE)
            #         self.set_steer(cf.angle)
            # if key == keyboard.Key.left:
            #     if cf.running and not cf.pause:
            #         cf.angle += cf.angle_increasement
            #         cf.angle = min(cf.angle, cf.MAX_ANGLE)
            #         self.set_steer(cf.angle)

            if key == keyboard.Key.esc:
                self.quit()


class AutoControl(Control):
    arr_speed = np.zeros(5)
    timespeed = time.time()

    def __init__(self):
        super(AutoControl, self).__init__()
        
        self.turnControl = TurnControl(0,0)
        cf.specialCorner = 1
        return

    def speedControl(self, speed, angle):
        tempangle = abs(angle)
        # if tempangle < 5:
        #     if speed < cf.MAX_SPEED-3:
        #         speed = speed+1
        #     elif speed < cf.MAX_SPEED-1:
        #         speed = speed+0.3
        # elif tempangle < 15:
        #     if speed < cf.MAX_SPEED-3:
        #         speed = speed + 1
        #     elif speed < cf.MAX_SPEED-1:
        #         speed = speed+0.3
        # elif tempangle > 30:
        #     if speed > cf.MIN_SPEED+4:
        #         speed = speed-2
        #     elif speed > cf.MIN_SPEED+2:
        #         speed = speed-0.5
        # elif tempangle > 20:
        #     if speed > cf.MIN_SPEED+4:
        #         speed = speed-1
        #     elif speed > cf.MIN_SPEED+2:
        #         speed = speed-0.25
        if abs(angle) > 16 or cf.turnSignal:
            speed = cf.FIXED_SPEED_TURN
        else:
            speed = cf.FIXED_SPEED
        
        # if speed != cf.speed:
        #     cf.speed = speed
        #     self.set_speed(cf.speed)
        # else:
        #     time.sleep(0.1)
        cf.speed = speed
        self.set_speed(cf.speed)
        # time.sleep(0.02)

        # return float(speed)

    def whereAmI(self):
        if cf.signSignal is not None: # sign detected! change the current route!
            cf.current_route = cf.signSignal
            if cf.signSignal == 'thang':
                if cf.start_tick_count is None:
                    cf.start_tick_count = 0

        if cf.imu_angle > 45-cf.imu_early and len(cf.turned) == 0: # turn first left
            cf.turned.append('left')
        if cf.imu_angle > 90-cf.imu_early and len(cf.turned) == 1: # turn second left
            cf.turned.append('left')
        if cf.imu_angle > 135-cf.imu_early and len(cf.turned) == 2: # turn third left
            cf.turned.append('left')
            cf.reduce_speed = True
        if cf.imu_angle > 90-cf.imu_early and len(cf.turned) == 3: # turn right at finish of segment 1
            cf.turned.append('right')
            cf.reduce_speed = False

        ''' Get to second segment '''
        if cf.current_route == 'phai':
            # skip some tick to just go straight without calculating the center
            if cf.imu_angle > 45-cf.imu_early and len(cf.turned) == 4: # turn right
                cf.turned.append('right')

            if cf.imu_angle > 90-cf.imu_early and len(cf.turned) == 5: # turn left
                cf.turned.append('left')
                cf.current_route = 'phai'
            if cf.imu_angle > 135-cf.imu_early and len(cf.turned) == 6: # turn left
                cf.turned.append('left')
            if cf.imu_angle > 180-cf.imu_early and len(cf.turned) == 7: # turn left
                cf.turned.append('left')

            if len(cf.turned) == 7: # we pass all the route. Now go straight to finish line
                go_straight = True

        if cf.current_route == 'thang':
            if cf.imu_angle > 45-cf.imu_early and len(cf.turned) == 4: # turn right
                cf.turned.append('right')

            if cf.imu_angle > 0-cf.imu_early and len(cf.turned) == 5: # turn right
                cf.turned.append('left')



    def auto_control(self):
        # while cf.running: # uncomment if run control in seperate thread
        # if True: # uncomment if call auto_control inside ImageProcessing
        if cf.ready and not cf.pause: # uncomment if call auto_control inside ImageProcessing

            self.whereAmI()

            if cf.start_tick_count is not None and cf.start_tick_count < cf.end_tick_from_start:
                cf.start_tick_count += 1
                cf.angle = 0
                cf.speed = cf.FIXED_SPEED
                print('cf.start_tick_count', cf.start_tick_count, '~~~~ cf.end_tick_from_start', cf.end_tick_from_start)
            else: 
                cf.start_tick_count = None

            if cf.signSignal == 'thang':
                cf.angle = 0

            # if cf.reduce_speed is True:
            #     speed = cf.speed_reduced
            # else:
            #     speed = self.speedControl(cf.speed, cf.angle)
            self.speedControl(cf.speed, cf.angle)
            self.set_steer(cf.angle)
            # time.sleep(0.01)
            # self.angleControl()

    def angleControl(self):
        # Only publish when everything is ready
        if not cf.pause and cf.ready:
            # Set speed and angle
            self.set_steer(cf.angle)
            # cf.speed = speed
            # self.set_speed(cf.speed)
            # print(cf.speed, cf.angle)

class TurnControl(object):
    def __init__(self, curspeed, maxspeed):
        self.maxTimeTurn = 60
        self.timeTurn = 0
        self.speedmax = maxspeed
        self.speedmin = 32
        self.speedDelta = 2
        self.k = 1.0
        self.currentspeed = curspeed - self.speedDelta*self.timeTurn
        self.leftDelta = 1.2
        self.leftAngle = -10
        self.rightDelta = 1.2
        self.rightAngle = 10

    def update(self):
        self.timeTurn = self.timeTurn + 1
        temp = self.currentspeed - self.speedDelta*self.timeTurn
        if temp >= self.speedmin:
            self.currentspeed = temp
        if self.timeTurn == self.maxTimeTurn//2:
            self.k = -self.k*0.6
        self.leftAngle = self.leftAngle + self.leftDelta*self.k
        self.rightAngle = self.rightAngle - self.rightDelta*self.k
        print("turn control k " + str(self.k))


class TurnDetector(object):
    crop_top = 180
    crop_bottom = 320
    kernel = np.ones((7,7),np.uint8)

    def __init__(self):
        return

    def region_of_interest(self, img, vertices):

        mask = np.zeros_like(img)   
        
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def detectTurn(self):  
        ### Params for region of interest
        bot_left = [0, 480]
        bot_right = [640, 480]
        apex_right = [640, 170]
        apex_left = [0, 170]
        v = [np.array([bot_left, bot_right, apex_right, apex_left], dtype=np.int32)]

        cropped_raw_image = self.region_of_interest(cf.img_rgb_raw, v)
        # cropped_raw_image = cf.img_rgb_raw[self.crop_top:self.crop_bottom, :]
        
        ### Run canny edge dection and mask region of interest
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hsv = cv2.cvtColor(cropped_raw_image, cv2.COLOR_BGR2HSV) 
        lower_white = np.array([0,0,255], dtype=np.uint8)
        upper_white = np.array([179,255,255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_white, upper_white) 
        dilation = cv2.dilate(mask, self.kernel, iterations=1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, self.kernel)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, self.kernel)

        blur = cv2.GaussianBlur(closing, (9,9), 0)
        edge = cv2.Canny(blur, 150,255)

        cropped_image = self.region_of_interest(edge, v)
        # cropped_image = edge[self.crop_top:self.crop_bottom, :]
        
        # blank_image = np.zeros(cropped_raw_image.shape)

        # turnSignal = False

        lines = cv2.HoughLines(cropped_image, rho=0.2, theta=np.pi/80, threshold=70)
        if lines is not None:
            # print('lines', len(lines))
            for line in lines:
                for rho,theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))

                    cv2.line(cropped_raw_image, (x1,y1), (x2,y2), cf.listColor[0], 2)
                    # cv2.line(blank_image, (x1,y1), (x2,y2), cf.listColor[0], 2)

                    if abs(y1-y2) < 40:
                        # turnSignal = True
                        # break
                        return True
        
        # cv2.imshow('hsv', hsv)
        # cv2.imshow('closing', closing)
        # cv2.imshow('cropped_image', cropped_image)
        # cv2.imshow('cropped_raw_image', cropped_raw_image)
        # cv2.imshow('blank_image', blank_image)

        return False




class SignDetector(object):
    def __init__(self):

        FLAGS = {
            "model_path": sys.path[1] + cf.sign_weight_h5_path,
            "anchors_path": sys.path[1] + cf.sign_anchor_path,
            "classes_path": sys.path[1] + cf.sign_class_name_path,
            "score" : 0.9,
            "iou" : 0.45,
            "model_image_size" : (416, 416),
            "gpu_num" : 1,
        }
        self.yolo = YOLO(FLAGS)

        return

    def signDetector(self, image):

        im_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        out_boxes, out_scores, out_classes = self.yolo.detect_image(im_pil)

        has_sign = False
        box = None

        if len(out_boxes) > 0:
            rs = True

            # print('Found', len(out_boxes), 'boxes', out_boxes, out_scores, out_classes))

            c = np.argmax(out_scores)
            class_id = out_classes[c]
            predicted_class = self.yolo.class_names[class_id]
            top, left, bottom, right = out_boxes[c]
            score = out_scores[c]
            left = int(left)
            right = int(right)
            top = int(top)
            bottom = int(bottom)
            w = right - left
            h = bottom - top
            if w >= 40 and h >= 40:
                label = 'Sign {} - {} ({},{})'.format(predicted_class, score, w, h)
                box = [left+cf.detect_sign_region['left'], top, w, h, predicted_class, score, label]
                cv2.rectangle(image, (left,top), (right,bottom), blue)

                return predicted_class, box
        
        return None, None


    def signRecognize(self):
        cf.k += 1
        # if cf.k >= 0 and cf.k % cf.sign_detect_step == 0:
        if True:
            # print('signRecognize', cf.k, cf.sign_detect_step, cf.signTrack)
            if cf.signTrack != -1 and abs(cf.signTrack-cf.k) >= 10:
                cf.sign = []
                cf.signTrack = -1
                print("clear")
                # cf.maxspeed = maxspeed

            img_detect_sign = cf.img_rgb_raw[cf.detect_sign_region['top']:cf.detect_sign_region['bottom'], cf.detect_sign_region['left']:cf.detect_sign_region['right']]
            
            cf.signSignal = None
            result, box = self.signDetector(img_detect_sign)
            if result != None:
                # cf.sign.append(result)
                # # cf.speed = 30
                # # cf.maxspeed = 30  # khi phat hien bien bao thi giam toc do
                # cf.signTrack = cf.k
                # print('Sign', cf.sign)
                # if self.acceptSign(result):
                if result:
                    if result == 'thang':
                        # print("THANG")
                        return "thang_certain", box
                    elif result == 'phai':
                        # print("PHAI")
                        return "phai_certain", box
                return result, box
        return None, None

    def acceptSign(self, value):
        if len(cf.sign) >= 2:
            for v in cf.sign:
                if v != value:
                    cf.sign = []
                    return False
            cf.sign = []
            cf.k = -100  # bo qua 100 frame tiep theo
            return True
        else:
            cf.k = cf.k - cf.sign_detect_step + 1  # xem xet 2 frame lien tiep
            return None


class TFModels():
    def __init__(self, model_path):
        super(TFModels, self).__init__()

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


class ImageProcessing(SignDetector, TurnDetector, AutoControl):

    def __init__(self):
        super(ImageProcessing, self).__init__()

        # output folder
        self.oDir = sys.path[1]+'/output/'+time.strftime("%Y-%m-%d_%H-%M-%S")
        self.oDirs = {
            'rgb': self.oDir+'/rgb',
            'depth': self.oDir+'/depth',
            'crop_cut': self.oDir+'/crop_cut'
        }
        for key in self.oDirs:
            odir = self.oDirs[key]
            if not os.path.isdir(odir):
                os.makedirs(odir)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.vid_out_rgb = cv2.VideoWriter(
            self.oDir+'/output_rgb.avi', fourcc, 20.0, (640,480))  # out video
        self.vid_out_rgb_viz = cv2.VideoWriter(
            self.oDir+'/output_rgb_viz.avi', fourcc, 20.0, (640,480))  # out processed video
        
        self.lane_tfmodel = TFModels(cf.lane_model_path)
        # self.lane_tfmodel_2 = TFModels(cf.lane_model_path)


    def getCenter(self):
        while cf.running: # seperate thread
        # if cf.running: # inside get_rgb
            # if cf.got_rgb_image and not cf.pause:
            if cf.got_rgb_image:
                cf.img_rgb_resized = cv2.resize(cf.img_rgb_raw, (cf.WIDTH, cf.HEIGHT))
                (h, w) = cf.img_rgb_resized.shape[:2]
                cf.img_rgb_resized = cf.img_rgb_resized[cf.crop_top_detect_center:, :]
                angle, cf.center = self.lane_tfmodel.predictCenter(cf.img_rgb_resized)
                # if abs(angle) > 30:
                # cf.angle = self.PID(angle,0.95,0.0001,0.01)
                #0.95,0.001,0.01
                # 1 0.0001 0.01 
                # else:
                # cf.angle = angle
                cf.angle = self.PID(angle,0.95,0.0001,0.02)
                print('[getCenter] predict angle : '+str(angle)+' | after PID : '+str(cf.angle))

                # cf.angle = 0

                # cf.signSignal, cf.sign_bbbox = self.signRecognize()
                # cf.ready = True
                # self.auto_control()
                if cf.ready:
                    self.auto_control()
                    # self.angleControl()
                self.visualize()

    def getSign(self):
        while cf.running: # seperate thread
            if cf.got_rgb_image:
                # cf.signSignal, cf.sign_bbbox = None, None
                cf.signSignal, cf.sign_bbbox = self.signRecognize()
                print('[getSign] cf.signSignal', cf.signSignal)
                cf.ready = True

    def getTurn(self):
        while cf.running: # seperate thread
            if cf.got_rgb_image:
                cf.turnSignal = self.detectTurn()
                print('[getTurn] cf.turnSignal', cf.turnSignal)
                time.sleep(0.1)

    def putTextSignal(self):
        if cf.signSignal is not None:
            color = None
            size = 5
            if cf.signSignal == 'thang': #red
                color = cf.listColor[0]
            elif cf.signSignal == 'phai': #green
                color = cf.listColor[1]
            elif cf.signSignal == "thang_certain": #blue
                color = cf.listColor[0]
                size = 10
            elif cf.signSignal == "phai_certain": #yellow
                color = cf.listColor[1]
                size = 10
            # if color is not None:
            #     cv2.circle(cf.rgb_viz, (10, 20), size, color, -1)

    def draw(self):
        (h, w) = cf.rgb_viz.shape[:2]

        # # line standard
        # pt1_start = (w//2, h//3)
        # pt1_end = (w//2, h)
        # cv2.line(cf.rgb_viz, pt1_start, pt1_end, red, 2)

        # # line predict
        # pt2_start = (cf.center*2, h//3)
        # pt2_end = pt1_end
        # cv2.line(cf.rgb_viz, pt2_start, pt2_end, green, 2)

        # draw speed and angle
        cv2.putText(cf.rgb_viz, 'IMU angle: '+str(cf.imu_angle)+' - '+str(cf.first_imu_angle), (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)
        cv2.putText(cf.rgb_viz, 'Speed: '+str(cf.speed), (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)
        cv2.putText(cf.rgb_viz, 'Angle: '+str(cf.angle), (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)
        if cf.signSignal is not None:
            cv2.putText(cf.rgb_viz, 'Sign signal: '+cf.signSignal, (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)
        cv2.putText(cf.rgb_viz, 'Turned: '+','.join(cf.turned), (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)

        # sign singal
        self.putTextSignal()

        # sign area
        cv2.rectangle(cf.rgb_viz, (cf.detect_sign_region['left'], cf.detect_sign_region['top']), (cf.detect_sign_region['right'], cf.detect_sign_region['bottom']), yellow, 2)
        
        # sign detect
        if cf.sign_bbbox is not None:
            x, y, w, h, predicted_class, score, label= cf.sign_bbbox

            x_plus_w = x+w
            y_plus_h = y+h

            cv2.rectangle(cf.rgb_viz, (x, y), (x_plus_w, y_plus_h), blue, 2)

            cv2.putText(cf.rgb_viz, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue, 2)

            cv2.putText(cf.rgb_viz, label, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)


    def visualize(self):
        log_angles = []
        # while cf.running: # seperate thread
        if cf.running: # inside getCenter (run in same thread)
            if cf.got_rgb_image:
                cf.rgb_viz = cf.img_rgb_raw.copy()
                self.draw()

                # cv2.imshow('cf.rgb_viz', cf.img_rgb_raw)
                cv2.imshow('cf.rgb_viz', cf.rgb_viz)
                # cv2.imshow('cf.depth_processed', cf.depth_processed)

                ## visualize center
                img_rgb_resized_viz = cf.img_rgb_resized.copy()
                (h, w) = img_rgb_resized_viz.shape[:2]
                # line standard
                pt1_start = (w//2, 160-cf.line_from_bottom)
                pt1_end = (w//2, 160)
                cv2.line(img_rgb_resized_viz, pt1_start, pt1_end, red, 2)

                # line predict
                pt2_start = (cf.center, 160-cf.line_from_bottom)
                pt2_end = pt1_end
                cv2.line(img_rgb_resized_viz, pt2_start, pt2_end, green, 2)

                cv2.circle(img_rgb_resized_viz, (cf.center, 160-cf.line_from_bottom), 5, green, -1)
                cv2.putText(img_rgb_resized_viz, 'predict: '+str(cf.predict), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)
                cv2.putText(img_rgb_resized_viz, 'center: '+str(cf.center)+' - '+str(cf.first_imu_angle), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)

                cv2.imshow('img_rgb_resized_viz', img_rgb_resized_viz)
                cv2.imwrite(self.oDirs['depth']+'/{}__{}_{}_{}.jpg'.format(self.frame_num, cf.speed, round(cf.angle,1), round(cf.imu_angle, 1)), img_rgb_resized_viz)
                cv2.imwrite(self.oDirs['crop_cut']+'/{}__{}_{}.jpg'.format(self.frame_num, cf.speed, round(cf.angle,1)), cf.img_rgb_resized)


                if cf.save_image:
                    if self.frame_num % 1 == 0 or cf.change_steer:
                        # print('save ', self.frame_num)
                        # cv2.imwrite(self.oDirs['rgb']+'/{}__{}_{}_{}.jpg'.format(self.frame_num, cf.speed, cf.angle, round(cf.imu_angle,2)), cf.img_rgb_raw)
                        cv2.imwrite(self.oDirs['rgb']+'/{}__{}_{}_{}.jpg'.format(self.frame_num, cf.speed, cf.angle, round(cf.imu_angle, 2)), cf.rgb_viz)
                        if cf.change_steer is True:
                            cf.change_steer = False
                    self.frame_num += 1

                if cf.is_record:
                    # self.vid_out_rgb.write(cf.img_rgb_raw)
                    self.vid_out_rgb_viz.write(cf.rgb_viz)
                if cf.save_log_angles:
                    log_angles.append(str(cf.angle))

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    cf.pause = True
        
        if cf.save_log_angles:
            with open(self.oDirs['rgb']+'/angles.txt', 'w') as outfile:
                outfile.write(' '.join(log_angles))


# class Camera(AutoControl):
class Camera(ImageProcessing):
    frame_num = 0

    def __init__(self):
        super(Camera, self).__init__()

        openni2.initialize(sys.path[1]+'/src/modules')
        print(sys.path[1]+'/src/modules')
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
        print("get_rgb called")
        self.print_lcd("get_rgb called")
        while cf.running:
            # if not cf.pause:
            if True:
                # start = time.time()
                # print('Get_rbg')
                bgr = np.fromstring(self.rgb_stream.read_frame(
                ).get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                cf.img_rgb_raw = rgb[:, ::-1, :]  # flip
                if cf.got_rgb_image is False:
                    cf.got_rgb_image = True
                # end = time.time()
                # seconds = end-start
                # cf.fps = self.frame_num / seconds
                # self.getCenter()

        print("get_rbg stoped")

    def get_depth(self):
        print("get_depth called")
        while cf.running:
            if True:
                frame = self.depth_stream.read_frame()
                frame_data = frame.get_buffer_as_uint16()
                img_depth = np.frombuffer(frame_data, dtype=np.uint16)
                img_depth.shape = (cf.HEIGHT, cf.WIDTH)
                cf.img_depth = img_depth
                if cf.got_depth_image is False:
                    cf.got_depth_image = True
        print("get_depth stopped")


class Subscribe(Control):
    def __init__(self):
        super(Subscribe, self).__init__()
        return

    def on_get_imu(self, data):
        delta_t = time.time() - self.t
        self.t = time.time()

        self.gyro_z = data.angular_velocity.z
        self.wz += delta_t*(self.gyro_z+self.last_gyro_z)/2.0
        cf.imu_angle = self.wz*180.0/np.pi
        self.last_gyro_z =  self.gyro_z

    def reset(self, data):
        print("Reset MPU")
        self.gyro_z = 0
        cf.imu_angle = 0
        self.wz = 0.0
        self.last_gyro_z = 0.0

        self.gyro_y = 0
        self.angle_y = 0
        self.wy = 0.0
        self.last_gyro_y = 0.0
 
    def on_get_imu_angle(self, res):
        data = res.data
        if cf.first_imu_angle is None:
            cf.first_imu_angle = data
        
        cf.imu_angle = - (data - cf.first_imu_angle)
        # print('imu_angle', cf.imu_angle)
        time.sleep(0.01)

    def on_get_lidar(self, data):
        self.lidar_data = data
        time.sleep(0.2)

    def on_get_sensor2(self, res):
        if res.data is True and cf.do_detect_barrier is True: # free in front (barrier open)
            self.run_car_by_signal()
            cf.do_detect_barrier = False
            cf.sub_sensor2.unregister()
            time.sleep(0.02)
            cf.sub_sensor2 = None
        if res.data is False: # something in front closely
            self.pause()
        # time.sleep(0.2)
    def on_get_btn_1(self, res):
        '''
        If button 2 is clicked, set mode to start
        '''
        if res.data is True: # click
            self.print_lcd('Running!')
            print('Running!')
            cf.running = True
            if cf.sub_sensor2 is None:
                self.run_car_by_signal()
            else:
                cf.do_detect_barrier = True
            cf.start_tick_count = 0
            # self.run_car_by_signal()
            # cf.do_detect_barrier = False
            # cf.sub_btn1.unregister()
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
    h = rospy.Subscriber(
        '/bt1_status', Bool, ros_sub.on_get_btn_1, queue_size=1)
    cf.sub_btn2 = rospy.Subscriber(
        '/bt2_status', Bool, ros_sub.on_get_btn_2, queue_size=1)
    cf.sub_btn3 = rospy.Subscriber(
        '/bt3_status', Bool, ros_sub.on_get_btn_3, queue_size=1)
    cf.sub_sensor2 = rospy.Subscriber(
        '/ss2_status', Bool, ros_sub.on_get_sensor2, queue_size=1)
    # cf.sub_getIMUAngle = rospy.Subscriber(
    #     '/imu_angle', Float32, ros_sub.on_get_imu_angle, queue_size=1)
    cf.sub_getIMUAngle = rospy.Subscriber(
        '/imu', Float32, ros_sub.on_get_imu, queue_size=1)
    cf.sub_lidar = rospy.Subscriber(
        '/scan', LaserScan, ros_sub.on_get_lidar, queue_size=1)
    # while cf.running:
    #     if not cf.pause:
    #         # cf.sub_lidar = rospy.Subscriber('/scan', LaserScan, ros_sub.on_get_lidar, queue_size=1)
    #         cf.sub_getIMUAngle = rospy.Subscriber(
    #             '/imu_angle', Float32, ros_sub.on_get_imu_angle, queue_size=1)
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


        # process_rgb_thread = threading.Thread(
        #     name="process_rgb_thread", target=self.process_rgb_image)
        # process_rgb_thread.start()

        # get_sign_thread = threading.Thread(
        #     name="get_sign_thread", target=self.getSign)
        # get_sign_thread.start()

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

        control_thread = threading.Thread(
            name="control", target=self.hand_control)
        control_thread.start()
        # self.hand_control()

        # listen_thread = threading.Thread(name="listen_thread", target=listenner)
        # listen_thread.start() # save data thread
        listenner()

        get_rgb_thread.join()
        # get_sign_thread.join()
        get_turn_thread.join()
        get_center_thread.join()
        # get_depth_thread.join()
        # show_thread.join()
        # auto_control_thread.join()
        control_thread.join()


if __name__ == "__main__":
    app = App()
    app.run()
