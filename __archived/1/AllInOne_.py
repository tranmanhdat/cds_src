#!/usr/bin/env python3

import tensorflow as tf
from keras.models import load_model
from tensorflow import keras
import math
import sys
import time
import rospy
from std_msgs.msg import String, Float32
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



# get and set path
rospack = rospkg.RosPack()
path = rospack.get_path('mtapos')

# init node
rospy.init_node('AllInOne', anonymous=True, disable_signals=True)


# set control variables
cf.running = True
cf.pause = True
cf.ready = True
cf.change_steer = False

# Speed and angle
cf.center = 0 # center to calculate angle
cf.init_speed = 15  # init speed
cf.angle = 0
cf.speed = 0
cf.MAX_SPEED = 25
cf.MIN_SPEED = 0
# 10 * maxLeft angle(20 degree) = -200, mul 10 to smooth control
cf.MIN_ANGLE = -60
cf.MAX_ANGLE = 60  # 10 * maxRight angle
cf.angle_increasement = 11
cf.speed_increasement = 1.0

# data collection
cf.is_record = True
cf.save_image = False

# set img / video variables
cf.HEIGHT = 240  # 480
cf.WIDTH = 320  # 640
# cf.HEIGHT_D = 240 # height of depth image
# cf.WIDTH_D = 320 # width of depth image
cf.img_rgb = np.zeros((cf.WIDTH, cf.HEIGHT, 3), np.uint8)
cf.img_depth = np.zeros((cf.WIDTH, cf.HEIGHT), np.uint8)
cf.rgb_processed = np.zeros((cf.WIDTH, cf.HEIGHT), np.uint8)
cf.depth_processed = np.zeros((cf.WIDTH, cf.HEIGHT), np.uint8)

# variables for sign detection
cf.signMode = 0
cf.signstep = 0
cf.signCount = 0
cf.k = 1
cf.sign = []
cf.signTrack = -1
cf.sign_detect_step = 3
cf.specialCorner = 1

# Model
cf.sign_weight_path = "/models/sign.weights"
cf.sign_config_path = "/models/yolov3-tiny-obj.cfg"
cf.lane_model_path = "/models/model.h5"
cf.sign_model_path = "/models/signgray.h5"
sys.path.insert(1, path)

cf.line = 70


cf.sub_lidar = None
cf.imu_angle = 0  # imu_angle subscribe from topic
cf.first_imu_angle = None



class Subscribe(object):
    def on_get_imu_angle(self, res):
        data = res.data
        if cf.first_imu_angle is None:
            cf.first_imu_angle = data
        
        cf.imu_angle = - (data - cf.first_imu_angle)
        # print('imu_angle', cf.imu_angle)

    def on_get_lidar(self, data):
        self.data = data


def listenner():
    ros_sub = Subscribe()
    while cf.running:
        if not cf.pause:
            # cf.sub_lidar = rospy.Subscriber('/scan', LaserScan, m_Lidar.on_receive, queue_size=1)
            cf.sub_getIMUAngle = rospy.Subscriber(
                '/imu_angle', Float32, ros_sub.on_get_imu_angle, queue_size=1)
            rospy.spin()


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
            print('set_speed', speed)
            speed_pub = rospy.Publisher('/set_speed', Float32, queue_size=1)
            speed_pub.publish(speed)

    def set_steer(self, steer):
        if steer == 0:
            print('set_angle', steer)
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

    def print_speed_lcd(self):
        self.set_lcd("0:0:      ")
        time.sleep(0.1)
        self.set_lcd("0:0:Speed = "+str(cf.speed))
        time.sleep(0.1)


class HandControl(Publish):
    def __init__(self):
        super(HandControl, self).__init__()
        return

    def hand_control(self):
        self.t_start = time.time()
        with keyboard.Listener(
                on_press=self.on_key_press) as listener:
            listener.join()

    def start(self):
        cf.pause = False
        print('Run!')
        # set_lcd('Run!')
        cf.speed = cf.init_speed
        self.set_speed(cf.speed)

    def pause(self):
        time_to_sleep = cf.speed*0.5/15
        print('cf.pause = True!')
        # before pause set speed to its negative value
        cf.speed = -cf.speed
        self.set_speed(cf.speed)

        cf.pause = True

        time.sleep(time_to_sleep)
        cf.speed = 0
        self.set_speed(cf.speed)

        cf.steer = 0
        self.set_steer(cf.steer)

        if cf.sub_lidar is not None:
            cf.sub_lidar.unregister()
            print('Unsubscribe lidar')

    def on_key_press(self, key):
        try:
            if key.char == 's':  # start
                if cf.running is False:
                    cf.running = True
                print('Start!')
                # set_lcd('Start!')
                if cf.pause is True:
                    self.start()
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
                if cf.pause is False:
                    self.pause()
                cv2.destroyAllWindows()
                print('Close cv2 windows')
                os._exit(0)

                return False


class AutoControl(Publish):
    arr_speed = np.zeros(5)
    timespeed = time.time()

    def __init__(self):
        super(AutoControl, self).__init__()
        return

    def speedControl(self, speed, angle, maxspeed, p=0.7, i=0, d=0.05):
        self.arr_speed[1:] = self.arr_speed[0:-1]
        self.arr_speed[0] = speed
        tempangle = abs(angle)
        if tempangle < 2:
            if speed < maxspeed-20:
                speed = speed+20
            elif speed < maxspeed-10:
                speed = speed+10
        elif tempangle < 5:
            if speed < maxspeed-20:
                speed = speed + 15
            elif speed < maxspeed-10:
                speed = speed+7
        elif tempangle > 15:
            if speed > maxspeed-10:
                speed = speed-20
            elif speed > maxspeed-20:
                speed = speed-10
        elif tempangle > 10:
            if speed > maxspeed-10:
                speed = speed-20
            elif speed > maxspeed-20:
                speed = speed-10
        # P = speed*p
        # delta_t = time.time() - self.timespeed
        # self.timespeed = time.time()
        # D = (speed-self.arr_speed[1])/delta_t*d
        # I = np.sum(self.arr_speed)*delta_t*i
        # speed = P + I + D
        if speed > maxspeed:
            speed = maxspeed
        elif speed < 10:
            speed = 20
        return int(speed)

    def auto_control(self):
        while cf.running:
            if cf.signMode == 1:
                if cf.signstep == 0:
                    print("START TURN LEFT")
                    turnControl = TurnControl(cf.speed, cf.MAX_SPEED)
                    cf.signCount = cf.signCount + 1
                    if cf.signCount == cf.specialCorner:
                        turnControl.maxTimeTurn = turnControl.maxTimeTurn + 10
                        turnControl.k = turnControl.k - 0.25
                cf.angle = turnControl.leftAngle
                cf.speed = turnControl.currentspeed
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
                cf.speed = turnControl.currentspeed
                cf.signstep = cf.signstep + 1
                turnControl.update()
                if cf.signstep > turnControl.maxTimeTurn:
                    print('-------DONE--------')
                    cf.signMode = 0
                    cf.signstep = 0
                    cf.MAX_SPEED = turnControl.speedmax
            else:
                cf.speed = self.speedControl(cf.speed, cf.angle, cf.MAX_SPEED)
                # cv2.putText(cf.img_rgb,"after: "+str(cf.angle), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

            # Only publish when everything is ready
            if not cf.pause and cf.ready:
                # set speed and angle
                self.set_steer(cf.angle)
                # cf.speed = 21
                self.set_speed(cf.speed)


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
        self.leftAngle = 10
        self.rightDelta = 1.2
        self.rightAngle = -10

    def update(self):
        self.timeTurn = self.timeTurn + 1
        temp = self.currentspeed - self.speedDelta*self.timeTurn
        if temp >= self.speedmin:
            self.currentspeed = temp
        if self.timeTurn == self.maxTimeTurn//2:
            self.k = -self.k*0.6
        self.leftAngle = self.leftAngle - self.leftDelta*self.k
        self.rightAngle = self.rightAngle + self.rightDelta*self.k
        print("turn control k " + str(self.k))


class SignDetector(object):
    def __init__(self):
        self.signDetector_net = None
        return

    def draw_pred(self, img, confidence, x, y, x_plus_w, y_plus_h):
        x = int(x)
        y = int(y)
        x_plus_w = int(x_plus_w)
        y_plus_h = int(y_plus_h)
        label = str(confidence)

        color = (0, 0, 255)

        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

        cv2.putText(img, label, (x-10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def getOutputsNames(self):
        layersNames = self.signDetector_net.getLayerNames()
        return [layersNames[i[0] - 1] for i in self.signDetector_net.getUnconnectedOutLayers()]

    def loadNetSign(self, weightPath, configPath):
        configPath = sys.path[1] + configPath
        weightPath = sys.path[1] + weightPath
        print(configPath)
        net = cv2.dnn.readNet(weightPath, configPath)
        return net

    def signDetector(self, image):
        blob = cv2.dnn.blobFromImage(
            image, 1.0/255.0, (416, 416), [0, 0, 0], True, crop=False)
        Width = image.shape[1]
        Height = image.shape[0]
        self.signDetector_net.setInput(blob)
        outs = self.signDetector_net.forward(self.getOutputsNames())
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.45
        nms_threshold = 0.4
        for out in outs:
            for detection in out:
                scores = detection[5:]  # classes scores starts from index 5
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
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
            # self.draw_pred(image,confidences[i], round(x), round(y), round(x+w), round(y+h))
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

        self.signDetector_net = self.loadNetSign(weightPath=cf.sign_weight_path, configPath=cf.sign_config_path)

    def caculateAngle(self, center, width):
        temp = math.atan(float(abs(float(center)-float(width/2))/float(cf.line)))
        angle = math.degrees(float(temp))
        if center > width/2:
            return angle
        else:
            return -angle

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

    def detectSign(self, frame_read):
        (h, w) = frame_read.shape[:2]
        #  detect sign only right corner (w//2,h//10) (w,h//2)
        frame_read = frame_read[h//10:h//2, w//2:]
        temp = frame_read.copy()
        rs, box = self.signDetector(frame_read)
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


class ImageProcessing(TFModels, AutoControl):
    frame_num = 0

    def __init__(self):
        super(ImageProcessing, self).__init__()

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
        cf.vid_out_rgb = cv2.VideoWriter(
            self.oDir+'/output_rgb.avi', fourcc, 30.0, (cf.WIDTH, cf.HEIGHT))  # out video

    def process_rgb(self):
        print("Get_rbg started!")
        while cf.running:
            # if not cf.pause:
            if True:
                # print('Get_rbg')
                bgr = np.fromstring(self.rgb_stream.read_frame(
                ).get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (cf.WIDTH, cf.HEIGHT))
                cf.img_rgb = rgb[:, ::-1, :]  # code cua anh De, ko hieu

                self.process_rgb_image()
        print("Get_rbg stoped")

    def process_depth(self):
        print("Get depth started!")
        while cf.running:
            # if not cf.pause:
            if True:
                frame = self.depth_stream.read_frame()
                frame_data = frame.get_buffer_as_uint16()
                img_depth = np.frombuffer(frame_data, dtype=np.uint16)
                img_depth.shape = (cf.HEIGHT, cf.WIDTH)
                cf.img_depth = img_depth
        print("Get depth stopped")

    def process_rgb_image(self):
        (height, _) = cf.img_rgb.shape[:2]
        angle, cf.center = self.predictCenter(cf.img_rgb[height//3:, :])
        print('\tpredict angle : '+str(angle))
        cf.angle = angle
        # cf.angle = PID(angle)
        # print('after PID : '+str(cf.angle))

        # phat hien bien bao
        # signal, box = None, None
        self.signal, self.sign_bbbox = self.signRecognize()
        # self.auto_control()

        self.visualize()


    def signRecognize(self):
        print('signRecognize')
        cf.k += 1
        if cf.k >= 0 and cf.k % cf.sign_detect_step == 0:
            if cf.signTrack != -1 and abs(cf.signTrack-cf.k) >= 10:
                cf.sign = []
                cf.signTrack = -1
                print("clear")
                # cf.maxspeed = maxspeed
            result, box = self.detectSign(cf.img_rgb)
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


    def putTextSignal(self):
        if self.signal is None:
            return cf.img_rgb
        index = 0
        if self.signal == 0: #red
            index = 0
        elif self.signal == 1: #green
            index = 1
        elif self.signal == "left_certain":#blue
            index =2
        elif self.signal == "right_certain":#yellow
            index = 3
        color = listColor[index]
        cv2.circle(cf.img_rgb, (10*index,20),10,color,-1)

    def drawCenter(self):
        (h, w) = cf.img_rgb.shape[:2]
        # line standard
        pt1_start = (w//2, h//3)
        pt1_end = (w//2, h)
        cv2.line(cf.img_rgb, pt1_start, pt1_end, red, 2)
        # line predict
        pt2_start = (cf.center, h//3)
        pt2_end = pt1_end
        cv2.line(cf.img_rgb, pt2_start, pt2_end, green, 2)
        # sign singal
        self.putTextSignal()
        # sign area
        cv2.rectangle(cf.img_rgb, (w//2,h//10), (w,h//2), yellow, 2)
        # sign detect
        if self.sign_bbbox is not None:
            (x,y,w1,h1,confidence) = self.sign_bbbox
            cv2.rectangle(cf.img_rgb, (x+w//2,y+h//10), (x+w//2+w1,y+h//10+h1), green, 2)

    def visualize(self):
        while cf.running:
            if True:
                self.drawCenter()
                cv2.imshow('cf.img_rgb', cf.img_rgb)

                # cv2.imshow('cf.rgb_processed', cf.rgb_processed)
                # cv2.imshow('cf.depth_processed', cf.depth_processed)
                if cf.save_image:
                    if self.frame_num % 3 == 0 or cf.change_steer:
                        # print('save ', self.frame_num)
                        cv2.imwrite(self.oDirs['rgb']+'/{}__{}_{}_{}.jpg'.format(self.frame_num, cf.speed, cf.angle, round(cf.imu_angle,2)), cf.img_rgb)
                        if cf.change_steer is True:
                            cf.change_steer = False
                    self.frame_num += 1

                if cf.is_record:
                    cf.vid_out_rgb.write(cf.img_rgb)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    cf.running = False
                if key == ord('q'):
                    cf.pause = True


class App(ImageProcessing, HandControl, AutoControl):

    def __init__(self):
        super(App, self).__init__()
        return

    def run(self):
        # start thread
        auto_control_thread = threading.Thread(
            name="auto_control_thread", target=self.auto_control)
        auto_control_thread.start()

        
        process_rbg_thread = threading.Thread(
            name="get_rbg_thread", target=self.process_rgb)
        process_rbg_thread.start()

        # process_depth_thread = threading.Thread(
        #     name="process_depth_thread", target=self.process_depth)
        # process_depth_thread.start()

        show_thread = threading.Thread(name="show_thread", target=self.visualize)
        show_thread.start() # save data thread

        # control_thread = threading.Thread(
        #     name="control", target=self.hand_control)
        # control_thread.start()
        self.hand_control()

        # listenner()

        process_rbg_thread.join()
        # process_depth_thread.join()
        show_thread.join()
        auto_control_thread.join()
        # control_thread.join()
        # show.join()


if __name__ == "__main__":
    app = App()
    app.run()
