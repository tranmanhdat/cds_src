#!/usr/bin/env python3

import tensorflow as tf
from keras.models import load_model
from tensorflow import keras
import math
import sys
import time
import rospy
from std_msgs.msg import String, Float32, Bool
from sensor_msgs.msg import LaserScan
from primesense import openni2
from primesense import _openni2 as c_api
import numpy as np
import rospkg
import cv2
import config as cf
from pynput import keyboard
import os
import threading



red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
yellow = (52, 235, 232)
listColor = [red, green, blue, yellow]
detect_sign_region = {'top':10, 'bottom':240, 'left':320, 'right':640}



# get and set path
rospack = rospkg.RosPack()
path = rospack.get_path('mtapos')

# init node
rospy.init_node('AllInOne', anonymous=True, disable_signals=True)


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
cf.MAX_SPEED = 15
cf.MIN_SPEED = 12
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
# cf.img_rgb_resized = np.zeros((cf.WIDTH, cf.HEIGHT, 3), np.uint8)
cf.img_depth = np.zeros((cf.WIDTH, cf.HEIGHT), np.uint8)
cf.depth_processed = np.zeros((cf.WIDTH, cf.HEIGHT), np.uint8)
cf.rgb_viz = np.zeros((cf.WIDTH, cf.HEIGHT), np.uint8)
# cf.img_detect_sign = np.zeros((detect_sign_region['right']-detect_sign_region['left'], detect_sign_region['bottom']-detect_sign_region['top']), np.uint8)
# cf.rgb_viz = np.zeros((detect_sign_region['right']-detect_sign_region['left'], detect_sign_region['bottom']-detect_sign_region['top']), np.uint8)

# variables for sign detection
cf.signMode = 0
cf.signstep = 0
cf.signCount = 0
cf.k = 1
cf.sign = []
cf.signTrack = -1
cf.sign_detect_step = 3
cf.specialCorner = 1
cf.signal = None
cf.sign_bbbox = None

# Model
cf.sign_weight_path = "/models/sign.weights"
cf.sign_config_path = "/models/yolov3-tiny-obj.cfg"
cf.lane_model_path = "/models/model.h5"
cf.sign_model_path = "/models/signgray.h5"
sys.path.insert(1, path)

cf.line = 70

# subscribe stuff
cf.sub_btn1 = None
cf.sub_btn2 = None
cf.sub_btn3 = None
cf.sub_sensor2 = None
cf.sub_lidar = None
cf.imu_angle = 0  # imu_angle subscribe from topic
cf.first_imu_angle = None



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
                # print('set_speed', speed)
                speed_pub = rospy.Publisher('/set_speed', Float32, queue_size=1)
                speed_pub.publish(speed)

    def set_steer(self, steer):
        if steer == 0:
            # print('set_angle', steer)
            steer_pub = rospy.Publisher('/set_angle', Float32, queue_size=1)
            steer_pub.publish(steer)
        elif cf.running and not cf.pause and cf.ready:
            cf.change_steer = True
            print('set_angle', steer)
            steer_pub = rospy.Publisher('/set_angle', Float32, queue_size=1)
            steer_pub.publish(steer)

    def set_lcd(self, text):
        lcd_pub = rospy.Publisher('/lcd_print', String, queue_size=1)
        lcd_pub.publish(text)

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

    def PID(self, error, p= 0.9, i =0, d = 0.005):
        self.error_arr[1:] = self.error_arr[0:-1]
        self.error_arr[0] = error
        P = error*p
        delta_t = time.time() - self.t
        self.t = time.time()
        D = (error-self.error_arr[1])/delta_t*d
        I = np.sum(self.error_arr)*delta_t*i
        angle = P + I + D
        if abs(angle)>60:
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
        cf.speed = cf.init_speed
        self.set_speed(cf.speed)

    def pause(self):
        if cf.pause is False:
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

            self.print_lcd('Pause!')

            if cf.sub_lidar is not None:
                cf.sub_lidar.unregister()
                print('Unsubscribe lidar')
    
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

        cv2.destroyAllWindows()
        print('Close cv2 windows')
        # self.print_lcd('Quit!')
        # time.sleep(0.1)
        self.clear_lcd()
        time.sleep(1)
        os._exit(0)

class HandControl(Control):
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
            if key.char == 'a':  # Set running True (to start sensor and camera and all that stuff)
                if cf.running is False:
                    cf.running = True
                print('Start!')
                self.print_lcd('Start!')
            if key.char == 's':  # Set pause false (to start running)
                if cf.running is False:
                    cf.running = True
                if cf.pause is True:
                    self.run_car_by_signal()
            if key.char == 'q':  # quit
                self.pause()
            if key.char == 'z':  # toggle saving images
                cf.save_image = not cf.save_image
        except AttributeError:
            # print('special key {0} pressed'.format(key))
            ''' Control speed '''
            if key == keyboard.Key.up:
                # print('cf.running, cf.pause', cf.running, cf.pause)
                if cf.running and not cf.pause:
                    cf.speed += cf.speed_increasement
                    cf.speed = min(cf.speed, cf.MAX_SPEED)
                    self.set_speed(cf.speed)
            if key == keyboard.Key.down:
                if cf.running and not cf.pause:
                    cf.speed -= cf.speed_increasement
                    cf.speed = max(cf.speed, cf.MIN_SPEED)
                    self.set_speed(cf.speed)

            ''' Control steer '''
            if key == keyboard.Key.right:
                if cf.running and not cf.pause:
                    cf.angle -= cf.angle_increasement
                    cf.angle = max(cf.angle, cf.MIN_ANGLE)
                    self.set_steer(cf.angle)
            if key == keyboard.Key.left:
                if cf.running and not cf.pause:
                    cf.angle += cf.angle_increasement
                    cf.angle = min(cf.angle, cf.MAX_ANGLE)
                    self.set_steer(cf.angle)

            if key == keyboard.Key.esc:
                self.quit()

                return False


class AutoControl(Control):
    arr_speed = np.zeros(5)
    timespeed = time.time()

    def __init__(self):
        super(AutoControl, self).__init__()
        return

    def speedControl(self, speed, angle):
        tempangle = abs(angle)
        if tempangle < 5:
            if speed < cf.MAX_SPEED-3:
                speed = speed+1
            elif speed < cf.MAX_SPEED-1:
                speed = speed+0.3
        elif tempangle < 15:
            if speed < cf.MAX_SPEED-3:
                speed = speed + 1
            elif speed < cf.MAX_SPEED-1:
                speed = speed+0.3
        elif tempangle > 30:
            if speed > cf.MIN_SPEED+4:
                speed = speed-2
            elif speed > cf.MIN_SPEED+2:
                speed = speed-0.5
        elif tempangle > 20:
            if speed > cf.MIN_SPEED+4:
                speed = speed-1
            elif speed > cf.MIN_SPEED+2:
                speed = speed-0.25
        if speed > cf.MAX_SPEED:
            speed = cf.MAX_SPEED
        elif speed < cf.MIN_SPEED:
            speed = cf.MIN_SPEED
        return float(speed)

    def auto_control(self):
        # while cf.running: # uncomment if run control in seperate thread
        if True: # uncomment if call auto_control inside ImageProcessing
            if cf.signMode == 1:
                if cf.signstep == 0:
                    print("START TURN LEFT")
                    turnControl = TurnControl(cf.speed, cf.MAX_SPEED)
                    cf.signCount = cf.signCount + 1
                    if cf.signCount == cf.specialCorner:
                        turnControl.maxTimeTurn = turnControl.maxTimeTurn + 10
                        turnControl.k = turnControl.k - 0.25
                cf.angle = turnControl.leftAngle
                speed = turnControl.currentspeed
                cf.signstep = cf.signstep + 1
                turnControl.update()
                if cf.signstep > turnControl.maxTimeTurn:
                    print('-------DONE--------')
                    cf.signMode = 0
                    cf.signstep = 0
                    cf.MAX_SPEED = turnControl.speedmax
            elif cf.signMode == 2:
                if cf.signstep == 0:
                    print("START TURN RIGHT")
                    turnControl = TurnControl(cf.speed, cf.MAX_SPEED)
                    cf.signCount = cf.signCount + 1
                    if cf.signCount == cf.specialCorner:
                        turnControl.maxTimeTurn = turnControl.maxTimeTurn + 10
                        turnControl.k = turnControl.k - 0.25
                cf.angle = turnControl.rightAngle
                speed = turnControl.currentspeed
                cf.signstep = cf.signstep + 1
                turnControl.update()
                if cf.signstep > turnControl.maxTimeTurn:
                    print('-------DONE--------')
                    cf.signMode = 0
                    cf.signstep = 0
                    cf.MAX_SPEED = turnControl.speedmax
            else:
                speed = self.speedControl(cf.speed, cf.angle)


            # Only publish when everything is ready
            if not cf.pause and cf.ready:
                # Set speed and angle
                cf.speed = speed
                self.set_speed(cf.speed)
                self.set_steer(cf.angle)
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


class SignDetector(object):
    def __init__(self):
        self.signDetectorNet = None
        return

    def getOutputsNames(self):
        layersNames = self.signDetectorNet.getLayerNames()
        return [layersNames[i[0] - 1] for i in self.signDetectorNet.getUnconnectedOutLayers()]

    def loadNetSign(self, weightPath, configPath):
        configPath = sys.path[1] + configPath
        weightPath = sys.path[1] + weightPath
        # print(configPath)
        net = cv2.dnn.readNet(weightPath, configPath)
        return net

    def signDetector(self, image):
        blob = cv2.dnn.blobFromImage(
            image, 1.0/255.0, (416, 416), [0, 0, 0], True, crop=False)
        Width = image.shape[1]
        Height = image.shape[0]
        cv2.imshow('area to detect sign', image)
        self.signDetectorNet.setInput(blob)
        outs = self.signDetectorNet.forward(self.getOutputsNames())
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.45
        nms_threshold = 0.4
        for out in outs:
            for detection in out:
                scores = detection[5:]  # classes scores starts from index 5
                # print('scores', scores)
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.03:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, conf_threshold, nms_threshold)
        rs = False
        box = None
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = int(box[0])
            y = int(box[1])
            w = int(box[2])
            h = int(box[3])
            box = (x, y, w, h, confidences[i])
            rs = True
            break
        return rs, box


class TFModels(SignDetector):
    def __init__(self):
        super(TFModels, self).__init__()

        self.config = tf.ConfigProto(
            device_count={'GPU': 1},
            intra_op_parallelism_threads=1,
            allow_soft_placement=True
        )
        self.config.gpu_options.allow_growth = True
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.6
        self.session = tf.Session(config=self.config)
        keras.backend.set_session(self.session)

        self.lane_model = load_model(sys.path[1]+cf.lane_model_path)
        self.sign_model = load_model(sys.path[1]+cf.sign_model_path)

        self.signDetectorNet = self.loadNetSign(weightPath=cf.sign_weight_path, configPath=cf.sign_config_path)

    def caculateAngle(self, center, width):
        temp = math.atan(float(abs(float(center)-float(width/2))/float(cf.line)))
        angle = math.degrees(float(temp))
        if center > width/2:
            return -angle
        else:
            return angle

    def predictCenter(self, image):
        img = cv2.resize(image, (320, 160))
        with self.session.as_default():
            with self.session.graph.as_default():
                predict = self.lane_model.predict(np.array([img])/255.0)[0]
        center = int(predict[0]*image.shape[1])
        angle = self.caculateAngle(center, image.shape[1])
        return angle, center

    def preprocess(self, image):
        image = cv2.resize(image, (25, 25))
        # image = imutils.resize(image, 25)
        blur = cv2.GaussianBlur(image, (3, 3), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
        image = gray
        image = image.reshape((25, 25, 1))
        return image

    def detectSign(self, image):
        temp = image.copy()
        rs, box = self.signDetector(image)
        if rs:
            image = temp[box[1]:(box[1]+box[3]), box[0]:(box[2]+box[0])]
            (a, b) = image.shape[:2]
            if abs(a-b) <= 3 and a >= 13 and b >= 13 and a <= 28 and b <= 28:
                image = cv2.resize(image, (25, 25))
                predictResult = None
                with self.session.as_default():
                    with self.session.graph.as_default():
                        predictResult = self.sign_model.predict(
                            np.array([self.preprocess(image)])/255.0)
                # for index, t in enumerate(temp):
                    # if t <= 0.1:
                    #     print("LEFT")
                    #     return 0, box
                    # elif t >= 0.9:
                    #     print("RIGHT")
                    #     return 1, box
                if predictResult is not None:
                    if predictResult[0][0] >= 0.7:
                        print("LEFT")
                        return 0, box
                    elif predictResult[0][1] >= 0.7:
                        print("RIGHT")
                        return 1, box
        return None, None


class Camera(TFModels, AutoControl):
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
                # print('Get_rbg')
                bgr = np.fromstring(self.rgb_stream.read_frame(
                ).get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                cf.img_rgb_raw = rgb[:, ::-1, :]  # flip
                if cf.got_rgb_image is False:
                    cf.got_rgb_image = True
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

class ImageProcessing(TFModels, AutoControl):

    def __init__(self):
        super(ImageProcessing, self).__init__()

        # output folder
        self.oDir = sys.path[1]+'/output/'+time.strftime("%Y-%m-%d_%H-%M-%S")
        self.oDirs = {
            'rgb': self.oDir+'/rgb',
            'depth': self.oDir+'/depth'
        }
        for key in self.oDirs:
            odir = self.oDirs[key]
            if not os.path.isdir(odir):
                os.makedirs(odir)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.vid_out_rgb = cv2.VideoWriter(
            self.oDir+'/output_rgb.avi', fourcc, 30.0, (640,480))  # out video
        self.vid_out_rgb_viz = cv2.VideoWriter(
            self.oDir+'/output_rgb_viz.avi', fourcc, 30.0, (640,480))  # out processed video
    
    def getCenter(self):
        while cf.running:
            # print('cf.got_rgb_image', cf.got_rgb_image)
            if cf.got_rgb_image:
                img_rgb_resized = cv2.resize(cf.img_rgb_raw, (cf.WIDTH, cf.HEIGHT))
                (height, _) = img_rgb_resized.shape[:2]
                angle, cf.center = self.predictCenter(img_rgb_resized[height//3:, :])
                # print('\tpredict angle : '+str(angle))
                cf.angle = angle
                # cf.angle = PID(angle)
                # print('after PID : '+str(cf.angle))

                cf.ready = True
                self.auto_control()
                # if cf.ready:
                #     self.auto_control()

    def getSign(self):
        while cf.running:
            # print('cf.got_rgb_image', cf.got_rgb_image)
            if cf.got_rgb_image:
                # cf.signal, cf.sign_bbbox = None, None
                cf.signal, cf.sign_bbbox = self.signRecognize()
                cf.ready = True

    def signRecognize(self):
        cf.k += 1
        if cf.k >= 0 and cf.k % cf.sign_detect_step == 0:
            # print('signRecognize', cf.k, cf.sign_detect_step, cf.signTrack)
            if cf.signTrack != -1 and abs(cf.signTrack-cf.k) >= 10:
                cf.sign = []
                cf.signTrack = -1
                print("clear")
                # cf.maxspeed = maxspeed

            # (h, w) = cf.img_rgb_raw.shape[:2]
            # img_detect_sign = cf.img_rgb_raw[h//10:h//2, w//2:]
            # img_detect_sign = cf.img_rgb_raw[10:240, 320:640]
            img_detect_sign = cf.img_rgb_raw[detect_sign_region['top']:detect_sign_region['bottom'], detect_sign_region['left']:detect_sign_region['right']]
            result, box = self.detectSign(img_detect_sign)
            if result != None:
                cf.sign.append(result)
                # cf.speed = 30
                # cf.maxspeed = 30  # khi phat hien bien bao thi giam toc do
                cf.signTrack = cf.k
                print('Sign', cf.sign)
                if self.acceptSign(result):
                    if result == 0:
                        print("OK LEFT HERE")
                        cf.signMode = 1
                        return "left_certain", box
                    elif result == 1:
                        print("OK RIGHT HERE")
                        cf.signMode = 2
                        return "right_certain", box
                return result, box
        return None, None

    def acceptSign(value):
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



    def drawSign(self):
        # cf.sign_bbbox = (x, y, w, h, confidences[i])
        if cf.sign_bbbox is None:
            return
        
        x = int(cf.sign_bbbox[0])
        y = int(cf.sign_bbbox[1])
        w = int(cf.sign_bbbox[2])
        h = int(cf.sign_bbbox[3])
        confidence = cf.sign_bbbox[4]

        x_plus_w = x+w
        y_plus_h = y+h
        label = str(confidence)

        color = (0, 0, 255)

        cv2.rectangle(cf.rgb_viz, (x, y), (x_plus_w, y_plus_h), color, 2)

        cv2.putText(cf.rgb_viz, label, (x-10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def putTextSignal(self):
        if cf.signal is not None:
            index = 0
            if cf.signal == 0: #red
                index = 0
            elif cf.signal == 1: #green
                index = 1
            elif cf.signal == "left_certain": #blue
                index =2
            elif cf.signal == "right_certain": #yellow
                index = 3
            color = listColor[index]
            cv2.circle(cf.rgb_viz, (10*index, 20), 10, color, -1)

    def drawCenterAndArea(self):
        cf.rgb_viz = cf.img_rgb_raw.copy()
        (h, w) = cf.rgb_viz.shape[:2]

        # line standard
        pt1_start = (w//2, h//3)
        pt1_end = (w//2, h)
        cv2.line(cf.rgb_viz, pt1_start, pt1_end, red, 2)

        # line predict
        pt2_start = (cf.center*2, h//3)
        pt2_end = pt1_end
        cv2.line(cf.rgb_viz, pt2_start, pt2_end, green, 2)

        # sign singal
        self.putTextSignal()

        # sign area
        cv2.rectangle(cf.rgb_viz, (detect_sign_region['left'], detect_sign_region['top']), (detect_sign_region['right'], detect_sign_region['bottom']), yellow, 2)
        
        # sign detect
        if cf.sign_bbbox is not None:
            (x, y, w1, h1, confidence) = cf.sign_bbbox
            cv2.rectangle(cf.rgb_viz, (x+w//2, y+h//10), (x+w//2+w1, y+h//10+h1), green, 2)


    def visualize(self):
        log_angles = []
        while cf.running:
            if cf.got_rgb_image:
                self.drawCenterAndArea()
                self.drawSign()

                # cv2.imshow('cf.rgb_viz', cf.rgb_viz)
                # cv2.imshow('cf.rgb_viz', cf.rgb_viz)
                # cv2.imshow('cf.depth_processed', cf.depth_processed)

                if cf.save_image:
                    if self.frame_num % 1 == 0 or cf.change_steer:
                        # print('save ', self.frame_num)
                        cv2.imwrite(self.oDirs['rgb']+'/{}__{}_{}_{}.jpg'.format(self.frame_num, cf.speed, cf.angle, round(cf.imu_angle,2)), cf.img_rgb_raw)
                        if cf.change_steer is True:
                            cf.change_steer = False
                    self.frame_num += 1

                if cf.is_record:
                    self.vid_out_rgb.write(cf.img_rgb_raw)
                    self.vid_out_rgb_viz.write(cf.rgb_viz)
                if cf.save_log_angles:
                    log_angles.append(str(cf.angle))

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    cf.running = False
                if key == ord('q'):
                    cf.pause = True
        
        if cf.save_log_angles:
            with open(self.oDirs['rgb']+'/angles.txt', 'w') as outfile:
                outfile.write(' '.join(log_angles))


class Subscribe(Control):
    def __init__(self):
        super(Subscribe, self).__init__()
        return

    def on_get_imu_angle(self, res):
        data = res.data
        if cf.first_imu_angle is None:
            cf.first_imu_angle = data
        
        cf.imu_angle = - (data - cf.first_imu_angle)
        # print('imu_angle', cf.imu_angle)

    def on_get_lidar(self, data):
        self.data = data

    def on_get_sensor2(self, res):
        if res.data is True and cf.do_detect_barrier is True: # free in front (barrier open)
            self.run_car_by_signal()
            cf.do_detect_barrier = False
        if res.data is False: # something in front closely
            self.pause()
    def on_get_btn_1(self, res):
        '''
        If button 2 is clicked, set mode to start
        '''
        if res.data is True: # click
            self.print_lcd('Running!')
            print('Running!')
            cf.running = True
            cf.do_detect_barrier = True
    def on_get_btn_2(self, res):
        '''
        If button 2 is clicked, pause!
        '''
        if res.data is True: # click
            self.pause()
    def on_get_btn_3(self, res):
        '''
        If button 3 is clicked, quit!
        '''
        if res.data is True: # click
            self.quit()

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
    while cf.running:
        cf.sub_sensor2 = rospy.Subscriber(
            '/ss2_status', Bool, ros_sub.on_get_sensor2, queue_size=1)
    #     if not cf.pause:
    #         cf.sub_lidar = rospy.Subscriber('/scan', LaserScan, m_Lidar.on_receive, queue_size=1)
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

        
        get_rgb_thread = threading.Thread(
            name="get_rbg_thread", target=self.get_rgb)
        get_rgb_thread.start()


        # process_rgb_thread = threading.Thread(
        #     name="process_rgb_thread", target=self.process_rgb_image)
        # process_rgb_thread.start()

        # get_sign_thread = threading.Thread(
        #     name="get_sign_thread", target=self.getSign)
        # get_sign_thread.start()

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

        listenner()

        get_rgb_thread.join()
        # get_sign_thread.join()
        get_center_thread.join()
        # get_depth_thread.join()
        # show_thread.join()
        # auto_control_thread.join()
        # control_thread.join()


if __name__ == "__main__":
    app = App()
    app.run()
