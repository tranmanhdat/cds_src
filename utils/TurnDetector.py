import cv2
import config as cf
import numpy as np

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
        # bot_left = [0, 480]
        # bot_right = [640, 480]
        # apex_right = [640, 170]
        # apex_left = [0, 170]
        # v = [np.array([bot_left, bot_right, apex_right, apex_left], dtype=np.int32)]

        # cropped_raw_image = self.region_of_interest(image, v)
        cropped_raw_image = cf.img_rgb_raw[self.crop_top:self.crop_bottom, :]

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

        # cropped_image = self.region_of_interest(edge, v)
        cropped_image = edge[self.crop_top:self.crop_bottom, :]
        
        # blank_image = np.zeros(cropped_raw_image.shape)

        turnSignal = False

        lines = cv2.HoughLines(cropped_image, rho=0.2, theta=np.pi/80, threshold=70)
        if lines is not None:
            print('lines', len(lines))
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
                        turnSignal = True
                        break
        
        # cv2.imshow('hsv', hsv)
        # cv2.imshow('closing', closing)
        # cv2.imshow('cropped_image', cropped_image)
        cv2.imshow('cropped_raw_image', cropped_raw_image)
        # cv2.imshow('blank_image', blank_image)

        return turnSignal


