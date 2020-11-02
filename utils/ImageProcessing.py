from utils.SignDetector import SignDetector
from utils.TurnDetector import TurnDetector
from utils.Control import AutoControl
from utils.TFModels import TFModels
import sys
import os
import numpy as np
import cv2
import config as cf
import time

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
            if cf.got_rgb_image and not cf.pause:
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
            if cf.got_rgb_image and not cf.pause:
                cf.turnSignal = self.detectTurn()
                print('[getTurn] cf.turnSignal', cf.turnSignal)

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


