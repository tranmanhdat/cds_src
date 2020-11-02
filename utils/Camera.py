from utils.ImageProcessing import ImageProcessing
import sys
import numpy as np
import cv2
import config as cf
from primesense import openni2
from primesense import _openni2 as c_api

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
                # print('Get_rbg')
                bgr = np.fromstring(self.rgb_stream.read_frame(
                ).get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                cf.img_rgb_raw = rgb[:, ::-1, :]  # flip
                if cf.got_rgb_image is False:
                    cf.got_rgb_image = True
                
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


