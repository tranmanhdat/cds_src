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
import glob

cf.sign_mapping = {0:'thang', 1:'phai', 2:'trai', 3:'dung'}
# cf.start_half_2 = False

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
# cf.sign_weight_h5_path = "/models/yolo/yolov3-tiny-sign__phai_thang.h5"
# cf.sign_anchor_path = "/models/yolo/yolov3-tiny-sign__phai_thang-anchors.txt"
# cf.sign_class_name_path = "/models/yolo/sign__phai_thang.names"
cf.sign_model_path = "/models/signs/sign_2__ok.h5"

# cf.lane_model_path = "/models/lane/model_full+den_2000_320x140_100.h5"
# cf.lane_model_path = "/models/lane/dem_full_320x140.h5"
# cf.lane_model_path = "/models/lane/dem_full+cuasang_320x140.h5"
# cf.lane_model_path = "/models/lane/dithang_320x140.h5"
# cf.lane_model_path = "/models/lane/dem_320x140.h5"
cf.lane_model_path = "/models/lane_draw/draw_320x200__160+40.h5"
# cf.lane_model_path = "/models/lane_draw/draw_320x180__140+40__old_model.h5"

cf.crop_top_detect_center = 80
org_thang = (160, 20)
org_phai = (300, 20)
org_trai = (20, 20)
pts_lanephai = [(220, 0), (260,40)]
pts_lanetrai = [(60, 0), (100,40)]


cf.line_from_bottom = 90
cf.predict = 0
cf.reduce_speed = False
cf.speed_reduced = 10
cf.imu_early = 10

# set control variables
cf.running = True # Must set to True. Scripts executed only when cf.running == True
cf.do_detect_barrier = False # = True when button 1 is clicked. = False when sensor 2 detects barrier open. Used everytime bring the car to the start position to redetect barrier open and restart everything
cf.pause = True # set speed = 0 and stop publishing stuff
cf.ready = True # make sure everything is loaded and image is successfully retrieved before car runs or speed/angle is published
cf.change_steer = False
cf.got_rgb_image = False
cf.got_depth_image = False

# Speed and angle
cf.center = 0 # center to calculate angle
cf.init_speed = 10  # init speed
cf.angle = 0
cf.speed = 0
cf.FIXED_SPEED = 13
cf.FIXED_SPEED_TURN = 13
cf.end_tick_from_start = 5
cf.tick_start_detect_sign = 30
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
cf.sign = []
cf.signTrack = -1
cf.sign_detect_step = 3
cf.specialCorner = 1
cf.do_detect_sign = True
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
                    self.run_car_by_signal()
            if key.char == 'r':  # reset
                if cf.running is False:
                    cf.running = True
                print('Start!')
                if cf.pause is True:
                    self.run_car_by_signal()
                    cf.signSignal = None
                    cf.sign = []
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

            if cf.start_tick_count is not None:
                if cf.start_tick_count < cf.end_tick_from_start:
                    cf.start_tick_count += 1
                    cf.angle = 0
                    cf.speed = cf.FIXED_SPEED
                    print('cf.start_tick_count', cf.start_tick_count, '~~~~ cf.end_tick_from_start', cf.end_tick_from_start)
                elif cf.start_tick_count == cf.tick_start_detect_sign:
                    cf.do_detect_sign = True
                    cf.start_tick_count = None

            self.speedControl(cf.speed, cf.angle)
            self.set_steer(cf.angle)

            self.whereAmI()

            # if cf.reduce_speed is True:
            #     speed = cf.speed_reduced
            # else:
            #     speed = self.speedControl(cf.speed, cf.angle)
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




class TrtFaceNet(object):
    """TrtFaceNet

    # Arguments
        engine: path to the TensorRT engine file
    """
    def __init__(self, engine, input_size, output_size):
        self.engine = trt.legacy.utils.load_engine(G_LOGGER, engine)
        self.runtime = trt.legacy.infer.create_infer_runtime(G_LOGGER)
        self.context = self.engine.create_execution_context()

        sample_input = np.empty(input_size, dtype=np.float32)
        self.output = np.empty(output_size, dtype=np.float32)

        self.d_input = cuda.mem_alloc(1 * sample_input.size * sample_input.dtype.itemsize)
        self.d_output = cuda.mem_alloc(1 * self.output.size * self.output.dtype.itemsize)

        assert(self.engine.get_nb_bindings() == 2)

    def predict(self, img):
        img = img.astype(np.float32)

        bindings = [int(self.d_input), int(self.d_output)]
        stream = cuda.Stream()

        # transfer input data to device
        cuda.memcpy_htod_async(self.d_input, img, stream)
        # execute model
        self.context.enqueue(1, bindings, stream.handle, None)
        # transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, stream)
        # syncronize threads
        stream.synchronize()

        return self.output

    def __del__(self):
        self.context.destroy()
        self.engine.destroy()
        self.runtime.destroy()



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

        # cropped_raw_image = cf.img_rgb_resized
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

        # cropped_image = edge
        cropped_image = self.region_of_interest(edge, v)

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

                    # cv2.line(cropped_raw_image, (x1,y1), (x2,y2), cf.listColor[0], 2)

                    if abs(y1-y2) < 40:
                        # turnSignal = True
                        # break
                        return True

        return False



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


    def preprocess(self, image):
        image = cv2.resize(image, (25, 25))
        # image = imutils.resize(image, 25)
        blur = cv2.GaussianBlur(image, (3, 3), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
        image = gray
        image = image.reshape((25, 25, 1))
        return image



class SignDetector(object):
    kernel = np.ones((2,2),np.uint8)
    k = 0

    def __init__(self):
        self.sign_tf = TFModels(cf.sign_model_path)
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
        self.k += 1

        run = False
        if cf.do_detect_sign and sign_to_detect == 'normal' and self.k % cf.sign_detect_step == 0:
            run = True
        elif cf.signSignal is not None and sign_to_detect == 'stop' and self.k % cf.sign_detect_step == 0:
            run = True

        if run:
            img_detect_sign = cf.img_rgb_raw[cf.detect_sign_region['top']:cf.detect_sign_region['bottom'], cf.detect_sign_region['left']:cf.detect_sign_region['right']]
            
            cf.signSignal = None
            if sign_to_detect == 'normal':
                boxes = self.signDetector(img_detect_sign)
            else: # detect stop sign
                boxes = self.signDetector(img_detect_sign, 'red')

            if len(boxes) == 0:
                return None, None

            for (x1,y1,x2,y2) in boxes:
                sign_region = img_detect_sign[y1:y2, x1:x2]
                # sign_region = cv2.cvtColor(sign_region, cv2.COLOR_BGR2GRAY)

                img = cv2.resize(sign_region, (100,100))

                # predict = self.sign_tf.predict(np.array([img])/255.0)[0]
                with self.sign_tf.session.as_default():
                    with self.sign_tf.session.graph.as_default():
                        predict = self.sign_tf.model.predict(np.array([img])/255.0)[0]

                sign_class_id = np.argmax(predict)
                prob = predict[sign_class_id]
                # print(predict, sign_class_id, prob)
                if prob > 0.97:
                    # print('\t', sign_class_id, prob)
                    # return sign_class_id, sign_bnd, prob
                    cf.sign.append(sign_class_id)
                    box = (x1+cf.detect_sign_region['left'], y1+cf.detect_sign_region['top'], x2+cf.detect_sign_region['left'], y2+cf.detect_sign_region['top'], sign_class_id, prob)
                    
                    if self.acceptSign(sign_class_id):

                        cf.do_detect_sign = False # stop detecting sign
                        # self.k = -200 # stop detecting sign for 200 frames (since no need to run this until stop sign)
                        # self.k = 0
                        # time.sleep(10) # or simply sleep until need to detect stop sign

                        # if cf.start_half_2 is False:
                        #     cf.start_half_2 = True
                        if sign_class_id == 0: # thang
                            return "thang_certain", box
                        elif sign_class_id == 1: # phai
                            return "phai_certain", box
                        elif sign_class_id == 2: # trai
                            return "trai_certain", box
                        elif sign_class_id == 3: # stop
                            return "dung_certain", box

                    return cf.sign_mapping[sign_class_id], box
        return None, None

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
    THRESHOLD_DAYTIME_NORMAL = "daytime-normal"
    THRESHOLD_DAYTIME_SHADOW = "daytime-shadow"
    THRESHOLD_DAYTIME_BRIGHT = "daytime-bright"
    THRESHOLD_DAYTIME_FILTER = "daytime-filter-pavement"
    THRESHOLD_WINDOWING = "windowing"

    def __init__(self):
        objpoints, imgpoints, chessboards = self.camera_chessboards()
        test_img = sys.path[1]+'/camera_cal/calibration1.jpg'

        img = cv2.imread(test_img)
        self.ret, self.mtx, self.dist, self.dst = self.camera_calibrate(objpoints, imgpoints, img)

        self.lane_net = TrtFaceNet(cf.lane_model_path, input_size=(180,320,3), output_size=(1,1))


    def camera_chessboards(self, glob_regex='/camera_cal/calibration*.jpg'):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.
        chessboards = [] # array of chessboard images
        
        # Make a list of calibration images
        images = glob.glob(sys.path[1]+glob_regex)

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (9,6), corners, ret)
                chessboards.append(img)
            
        return objpoints, imgpoints, chessboards


    def camera_calibrate(self, objpoints, imgpoints, img):
        # Test undistortion on an image
        img_size = img.shape[0:2]

        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        
        return ret, mtx, dist, dst


    def undistort_image(self, img, mtx, dist):
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        return dst


    def calc_warp_points(self, img_height,img_width,x_center_adj=0):
        
        # calculator the vertices of the region of interest
        imshape = (img_height, img_width)
        xcenter = imshape[1]/2+x_center_adj
        xcenter = imshape[1]/2 - 20
        xfd = 160
        yf = 300 # 350
        xoffset = 100 # 100

        src = np.float32(
            [(xoffset,imshape[0]),
            (xcenter-xfd, yf), 
            (xcenter+xfd,yf), 
            (imshape[1]-xoffset,imshape[0])])
        
        # calculator the destination points of the warp
        dst = np.float32(
            [(xoffset,imshape[1]),
            (xoffset,0),
            (imshape[0]-xoffset, 0),
            (imshape[0]-xoffset,imshape[1])])
        
        # print('src', src)
        # print('dst', dst)
        
        return src, dst
        
    def perspective_transforms(self, src, dst):
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        
        return M, Minv

    def perspective_warp(self, img, M):
        #img_size = (img.shape[1], img.shape[0])
        img_size = (img.shape[0], img.shape[1])
        
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        
        return warped

    def perspective_unwarp(self, img, Minv):
        #img_size = (img.shape[1], img.shape[0])
        img_size = (img.shape[0], img.shape[1])
        
        unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
        
        return unwarped
        

    def _warp_undistorted_threshold(self, undistorted_image, threshold):
        binary_image = combined_threshold(undistorted_image, threshold=threshold)
        warped = perspective_warp(binary_image, self.__M)
        return (binary_image, warped, threshold)

    def combined_threshold(self, img, kernel=3, grad_thresh=(30,100), mag_thresh=(70,100), dir_thresh=(0.8, 0.9),
                        s_thresh=(100,255), r_thresh=(150,255), u_thresh=(140,180),
                        #threshold="daytime-normal")
                        threshold="daytime-filter-pavement"):

        def binary_thresh(channel, thresh = (200, 255), on = 1):
            binary = np.zeros_like(channel)
            binary[(channel > thresh[0]) & (channel <= thresh[1])] = on

            return binary
        
        # check up the default red_min threshold to cut out noise and detect white lines
        if threshold in ["daytime-bright","daytime-filter-pavement"]:
            r_thresh=(210,255)
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
        
        # calculate the scobel x gradient binary
        abs_sobelx = np.absolute(sobelx)
        scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        gradx = binary_thresh(scaled_sobelx, grad_thresh)
        
        # calculate the scobel y gradient binary
        abs_sobely = np.absolute(sobely)
        scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
        grady = binary_thresh(scaled_sobely, grad_thresh)
        
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        mag_binary = binary_thresh(gradmag, mag_thresh)
        
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        dir_binary = binary_thresh(absgraddir, dir_thresh)
        
        # HLS colour
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        H = hls[:,:,0]
        L = hls[:,:,1]
        S = hls[:,:,2]
        sbinary = binary_thresh(S, s_thresh)
        
        # RGB colour
        R = img[:,:,2]
        G = img[:,:,1]
        B = img[:,:,0]
        rbinary = binary_thresh(R, r_thresh)
        
        # YUV colour
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        Y = yuv[:,:,0]
        U = yuv[:,:,1]
        V = yuv[:,:,2]
        ubinary = binary_thresh(U, u_thresh)
        
        combined = np.zeros_like(dir_binary)
        
        if threshold == "daytime-normal": # default
            combined[(gradx == 1)  | (sbinary == 1) | (rbinary == 1) ] = 1
        elif threshold == "daytime-shadow":
            combined[((gradx == 1) & (grady == 1)) | (rbinary == 1)] = 1
        elif threshold == "daytime-bright":
            combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (sbinary ==1 ) | (rbinary == 1)] = 1
        elif threshold == "daytime-filter-pavement":
            road_binary=binary_thresh(binary_filter_road_pavement(img))
            combined[(((gradx == 1)  | (sbinary == 1)) &(road_binary==1)) | (rbinary == 1 ) ] = 1
        else:
            combined[((gradx == 1) | (rbinary == 1)) & ( (sbinary ==1) | (ubinary ==1)| (rbinary == 1 ))] = 1
            
        return combined


    def region_of_interest(self, img, vertices):
        mask = np.zeros_like(img)
        match_mask_color = 255
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def findCenterSimple(self):
        height, width = cf.img_rgb_raw.shape[:2] # (480, 640)

        bot_left = [0, 480]
        bot_right = [640, 480]
        apex_right = [420, 160] # [width/4, height/3]
        apex_left = [220, 160]
        v = [np.array([bot_left, bot_right, apex_right, apex_left], dtype=np.int32)]
        region = self.region_of_interest(cf.img_rgb_raw, v)
        
        undistorted_image = self.undistort_image(cf.img_rgb_raw, self.mtx, self.dist)
        binary_image = self.combined_threshold(undistorted_image, threshold=self.THRESHOLD_DAYTIME_BRIGHT)

        warp_src, warp_dst = self.calc_warp_points(height, width)
        M, Minv = self.perspective_transforms(warp_src, warp_dst)
        warped_ = self.perspective_warp(binary_image, M)
        
        # pts = np.array([(220,160), (420, 160), (640, 480), (0, 480)], dtype = "float32")
        # warped = four_point_transform(binary_image, pts)

        # cannyed = cv2.Canny(warped_, 10, 255)
        line_from_bottom = 250
        y = warped_.shape[0]-line_from_bottom
        center = None
        for x in range(warped_.shape[1]//2, warped_.shape[1]):
            if warped_[y][x] == 1:
                center = (x-130,y)
                break
        if center is None: # cant detect right lane, can't calculate center 
            return 0,0
            
        cv2.circle(warped_, center, 23, 255, -1)
        cv2.line(warped_, (0,y), (warped_.shape[1],y), 2, 3)
        
        # unwarp_ = transform_back(warped_)
        unwarp_ = self.perspective_unwarp(warped_, Minv)

        # cv2.imshow('undistorted_image', undistorted_image)
        cv2.imshow('binary_image', binary_image)
        # cv2.imshow('warped', warped)
        cv2.imshow('warped_', warped_)
        cv2.imshow('unwarp_', unwarp_)
        cv2.imshow('region', region)

        angle = self.calAngle(center[0], warped_.shape[1])
        cf.img_rgb_resized = cv2.resize(cf.img_rgb_raw, (cf.WIDTH, cf.HEIGHT))
        center_origin = self.calCenterFromAngle(angle, cf.img_rgb_resized.shape[1])

        return center_origin, angle


    def findCenterAdvance(self):
        cf.img_rgb_resized = cv2.resize(cf.img_rgb_raw, (cf.WIDTH, cf.HEIGHT))
        (h, w) = cf.img_rgb_resized.shape[:2]
        cf.img_rgb_resized = cf.img_rgb_resized[cf.crop_top_detect_center:, :]

        # angle, cf.center = self.lane_tf.predictCenter(cf.img_rgb_resized)

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

            # draw guide to bam lane
            # chay nua sau (cf.start_half_2)
            # lane phai
            cv2.rectangle(img_more, pts_lanephai[0], pts_lanephai[1], blue, -1)
            # lane trai
            # cv2.rectangle(img_more, pts_lanetrai[0], pts_lanetrai[1], blue, -1)

        img = cv2.vconcat([cf.img_rgb_resized, img_more])
        # print(img.shape)
        cv2.imshow('img', img)

        # with self.lane_tf.session.as_default():
        #     with self.lane_tf.session.graph.as_default():
        #         predict = self.lane_tf.model.predict(np.array([img])/255.0)[0]
        predict = self.lane_net.predict(img)
        cf.predict = predict[0]
        center = int(predict[0]*cf.img_rgb_resized.shape[1])
        angle = self.calAngle(center, w)
        return center


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
        
        self.lane_tf = TFModels(cf.lane_model_path)
        self.SignDetector = SignDetector()
        self.TurnDetector = TurnDetector()
        self.CenterDetector = CenterDetector() 


    def getCenter(self):
        while cf.running: # seperate thread
        # if cf.running: # inside get_rgb
            # if cf.got_rgb_image and not cf.pause:
            if cf.got_rgb_image:
                cf.center, angle = self.CenterDetector.findCenterSimple()

                # cf.angle = self.PID(angle, 0.95, 0.0001, 0.02)
                cf.angle = angle
                print('[getCenter] predict angle : '+str(angle)+' | after PID : '+str(cf.angle))

                # cf.turnSignal = self.detectTurn()

                if cf.ready:
                    self.auto_control()
                    # self.angleControl()
                self.visualize()


    def getSign(self):
        while cf.running: # seperate thread
            if cf.got_rgb_image:
                # cf.signSignal, cf.sign_bbbox = None, None
                if cf.do_detect_sign:
                    if cf.signSignal is None or 'certain' not in cf.signSignal:
                        cf.signSignal, cf.sign_bbbox = self.SignDetector.signRecognize()
                        print('[getSign] cf.signSignal', cf.signSignal)
                        time.sleep(0.1)
                    else:
                        cf.sign_bbbox = None
                        time.sleep(15)
                elif cf.signSignal: # do_detect_sign = False and cf.signSignal <=> need to detect stop sign
                    cf.sign_bbbox = None
                    self.k = 0
                    cf.stopSignal, cf.sign_bbbox = self.SignDetector.signRecognize('stop')

    def getTurn(self):
        while cf.running: # seperate thread
            if cf.got_rgb_image:
                cf.turnSignal = self.TurnDetector.detectTurn()
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
        cv2.putText(cf.rgb_viz, 'Turned: '+','.join(cf.turned), (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)

        # draw fps
        cv2.putText(cf.rgb_viz, 'FPS: '+str(cf.fps_count), (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, blue, 2)

        # sign singal
        self.putTextSignal()

        # sign area
        cv2.rectangle(cf.rgb_viz, (cf.detect_sign_region['left'], cf.detect_sign_region['top']), (cf.detect_sign_region['right'], cf.detect_sign_region['bottom']), yellow, 2)
        
        # sign detect
        if cf.sign_bbbox is not None:
            x1, y1, x2, y2, predicted_class, score = cf.sign_bbbox
            label = cf.signSignal+' - '+str(score)

            cv2.rectangle(cf.rgb_viz, (x1, y1), (x2, y2), blue, 2)
            cv2.putText(cf.rgb_viz, label, (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue, 2)
        if cf.signSignal is not None:
            cv2.putText(cf.rgb_viz, cf.signSignal, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)


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

cf.fps_count = 0

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
        start = time.time()
        # fps_count_prev = 0
        while cf.running:
            # if not cf.pause:
            if True:
                # print('Get_rbg')
                bgr = np.fromstring(self.rgb_stream.read_frame(
                ).get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                cf.img_rgb_raw = rgb[:, ::-1, :]  # flip
                if cf.got_rgb_image is False:
                    cf.got_rgb_image = True
                
                # print('[get_rgb]')

                seconds = time.time() - start
                # fps_count_prev = cf.fps_count
                # fps_count_now = self.frame_num / seconds
                # cf.fps_count = 0.9*fps_count_prev + 0.1*fps_count_now
                cf.fps_count = self.frame_num / seconds
                self.frame_num += 1
                # time.sleep(0.01)

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

        self.tick = time.time()
        self.gyro_z = 0
        self.angle = 0
        self.wz = 0.0
        self.last_gyro_z = 0.0

        return

    def on_get_imu(self, data):
        delta_t = time.time() - self.tick
        self.tick = time.time()

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
    # h = rospy.Subscriber(
    #     '/bt1_status', Bool, ros_sub.on_get_btn_1, queue_size=1)
    # cf.sub_btn2 = rospy.Subscriber(
    #     '/bt2_status', Bool, ros_sub.on_get_btn_2, queue_size=1)
    # cf.sub_btn3 = rospy.Subscriber(
    #     '/bt3_status', Bool, ros_sub.on_get_btn_3, queue_size=1)
    cf.sub_sensor2 = rospy.Subscriber(
        '/ss2_status', Bool, ros_sub.on_get_sensor2, queue_size=1)
    # cf.sub_getIMUAngle = rospy.Subscriber(
    #     '/imu_angle', Float32, ros_sub.on_get_imu_angle, queue_size=1)
    # cf.sub_getIMUAngle = rospy.Subscriber(
    #     '/imu', Imu, ros_sub.on_get_imu, queue_size=1)
    # cf.sub_lidar = rospy.Subscriber(
    #     '/scan', LaserScan, ros_sub.on_get_lidar, queue_size=1)
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

        control_thread = threading.Thread(
            name="control", target=self.hand_control)
        control_thread.start()
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
        control_thread.join()



if __name__ == "__main__":
    app = App()
    app.run()
