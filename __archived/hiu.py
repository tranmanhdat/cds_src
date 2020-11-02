#!/usr/bin/env python
import cv2
import numpy as np
import rospkg 
from primesense import openni2
from primesense import _openni2 as c_api
import config as cf
import time
import rospy

cf.running = True
cf.pause = False

rospy.init_node('hiu', anonymous=True, disable_signals=True)

rospack = rospkg.RosPack()
path = rospack.get_path('mtapos')
openni2.initialize(path+'/src/modules') #
dev = openni2.Device.open_any()
rgb_stream = dev.create_color_stream()
rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=640, resolutionY=480, fps=30))
rgb_stream.start()
depth_stream = dev.create_depth_stream()
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 320, resolutionY = 240, fps = 30))
depth_stream.start()

def get_rgb():
    print("Get_rbg started!")
    while cf.running:
        if not cf.pause:
            while True:
                bgr   = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(480,640,3)
                rgb   = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb,(480, 320))
                cf.img = rgb[:, ::-1, :]
                # cf.syn = True
                cv2.imshow('rgb', rgb)
                key = cv2.waitKey(1)
                if key == ord('q') or k == 27: # quit or esc
                    break
    print("Get_rbg stoped")

def get_depth():
    print("Get_depth started!")
    while cf.running:
        if not cf.pause:
            time.sleep(0.05)
            frame = depth_stream.read_frame()
            frame_data = frame.get_buffer_as_uint16()
            img = np.frombuffer(frame_data, dtype=np.uint16)
            img.shape = (240, 320)
            img = cv2.flip(img,1)
            #cv2.imshow("img",img)
            img2 = img.copy()
            img2 = (img2*1.0/2**8).astype(np.uint8)
            img2 = 255 - img2
            cv2.imshow("depth", img2)
            cf.depth = cv2.resize(img2, (480, int(img2.shape[0]*360/240)))
    print("Get_depth stoped")


get_rgb()