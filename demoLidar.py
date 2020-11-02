#!/usr/bin/env python3
# license removed for brevity
import rospy
import numpy as np
import time, math
from sensor_msgs.msg import LaserScan
import rospkg
import cv2
from statistics import median
rospy.init_node('Demo', anonymous=True, disable_signals=True)
# def draw_angled_rec(x0, y0, length, angle, img):
#     px = int(round(x0 + length * math.cos(angle * math.pi / 180.0)))
#     py = int(round(y0 + length * math.sin(angle * math.pi / 180.0)))
#     img = cv2.circle(img, (px,py), 3, (0,255,0), 3)
#     return img
## draw
    # img = cv2.imread('/home/ubuntu/catkin_ws/src/mtapos/src/white-box.png')
    # cv2.imshow('img',img)
    # print(img.shape)
    # for i in range(0,180):
    #     if distanceArr[i]<10:
    #         img = draw_angled_rec(240,240,distanceArr[i]*50,-i,img)
    # cv2.imshow('img',img)
    # cv2.waitKey(1)
def sumArr(Arr, start, end):
    sum = 0
    for i in range(start,end+1):
        sum = sum + Arr[i]
    return sum
def findoutObj(obj, distanceMax):
    """
    detect if have object in front or right

    Parameters:
    obj (dictionary) : contain [start, end, distance]

    distanceMax (int) : distance max can detect obj

    Returns:
    int : 0 if dont have object, 1 if have object in front, 2 if have object in right

    [start, end, distance] : object location,None if doesn't have object

    """
    straightSteer = [356,5]
    rightSteer = [315,355]
    behindSteer = [180,270]
    for key, value in obj.items():
        # print('value', value)
        if value[2] < distanceMax:
            if value[0] < straightSteer[1]:
                return 1,value
            if value[0]<straightSteer[0]:
                if value[1]>straightSteer[0] or value[1]<value[0]:
                    return 1,value
                elif value[1]>rightSteer[0]:
                    return 2,value
            else:
                return 1,value
            if value[2] > 1.0 and value[0]>behindSteer[0] and value[1]<behindSteer[1]:
                return 3, value
    return 0,None
                
def objectScan(data):
    distanceArr = list(data.ranges)
    # print(distanceArr)
    # print(len(distanceArr))
    obj = {}
    count = 0
    i=0
    theshAccept = 0.3

    while i < 359:
        while distanceArr[i]>20:
            i = i + 1
        start = i
        end = i 
        sum = distanceArr[i]
        while i < 359:
            if abs(distanceArr[i]-distanceArr[i+1]) < theshAccept:
                sum = sum + distanceArr[i+1]
                i = i + 1
            else :
                if i < 358:
                    if abs(distanceArr[i]-distanceArr[i+2]) < theshAccept:
                        distanceArr[i+1] = distanceArr[i+2]
                        sum = sum + 2* distanceArr[i+2]
                        i = i + 2
                    else:
                        break
                else:
                    break
        if i < 358:
            distance = sum/(i-start+1)
            obj[str(count)] = [start, i, distance]
        elif i==358:
            if  abs(distanceArr[358]-distanceArr[0]) < theshAccept:
                end = obj['0'][1]
                distanceArr[359] = distanceArr[0]
                sum = sum + distanceArr[0]
                distance = (sum+sumArr(distanceArr, start, end))/(361-start+end)
                obj['0'] = [start, end, distance]
        elif i==359:
            if  abs(distanceArr[359]-distanceArr[0]) < theshAccept:
                end = obj['0'][1]
                distance = (sum+sumArr(distanceArr, 0, end))/(361-start+end)
                obj['0'] = [start, end, distance]
            elif  abs(distanceArr[359]-distanceArr[1]) < theshAccept:
                end = obj['0'][1]
                distanceArr[0] = distanceArr[1]
                distance = (sum+sumArr(distanceArr, 0, end))/(361-start+end)
                obj['0'] = [start, end, distance]
        i = i +2
        count = count + 1
    # print(distanceArr)
    print(obj)
    print('===========================')
    print('findout OBJ', findoutObj(obj,3.5))
    print('--------------------------------------------------')
    time.sleep(1)
def listenner():
    data = rospy.Subscriber(
        '/scan', LaserScan, objectScan, queue_size=1)
    
    rospy.spin()
listenner()

