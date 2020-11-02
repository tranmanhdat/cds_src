#!/usr/bin/env python
# license removed for brevity
import rospy,cv2
from std_msgs.msg import String, Float32,Bool

def talker():
    rospy.init_node('talk', anonymous=True)#node name
    # pub = rospy.Publisher('chatter', String, queue_size=1)# topic name
    pub = rospy.Publisher('/set_angle', Float32, queue_size=1)
    pub2 = rospy.Publisher('/set_speed', Float32, queue_size=1)
    # pub3 = rospy.Publisher('/led_status',Bool,queue_size=1)
    # pub4 = rospy.Publisher('/lcd_print',String,queue_size=1)

    rate = rospy.Rate(10) # 10hz
    # stop = False
    while not rospy.is_shutdown():
        # hello_str = "hello world %s" % rospy.get_time()
        # rospy.loginfo(hello_str)#in ra man hinh

        pub.publish(10) # public cho thang sub nhan
        pub2.publish(0) # speed

        # if not stop:
        #     pub.publish(20) # public cho thang sub nhan
        #     pub2.publish(40) # speed
        # else:
        #     pub2.publish(0)
        #     pub.publish(0) 
        # key = cv2.waitKey(1)
        # if key == ord('q'):
        #     stop = True
        # else:
        #     stop = False
        rate.sleep()# dung du lau de thang kia nhan nhung van dam bao toc do thuc thi cua vong lap
        # continue

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
