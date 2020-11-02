#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from primesense import openni2#, nite2
from primesense import _openni2 as c_api
import numpy as np
import rospkg 
import cv2
import config as cf

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
while True:
    bgr = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(480,640,3)
    rgb   = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb,(480, 320))
    rgb = cv2.flip(rgb,1)

    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()
    img = np.frombuffer(frame_data, dtype=np.uint16)
    img.shape = (240, 320)
    img = cv2.flip(img,1)
    img2 = img.copy()
    img2 = (img2*1.0/2**8).astype(np.uint8)
    img2 = 255 - img2
    cv2.imshow('depth',img2)
    cv2.imshow('rgb',rgb)
    key = cv2.waitKey(1)
    if key==ord('q'):
        break    