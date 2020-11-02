import sys
import config as cf
import numpy as np
from yolo import YOLO
from PIL import Image
import cv2

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

        # has_sign = False
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
                cv2.rectangle(image, (left,top), (right,bottom), cf.listColor[1])

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

