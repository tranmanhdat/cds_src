#!/usr/bin/env python
# license removed for brevity
import rospy,cv2
from std_msgs.msg import String, Float32,Bool

def talker():
    # pub = rospy.Publisher('chatter', String, queue_size=1)# topic name
    pub = rospy.Publisher('/set_angle', Float32, queue_size=1)
    pub2 = rospy.Publisher('/set_speed', Float32, queue_size=1)
    # pub3 = rospy.Publisher('/led_status',Bool,queue_size=1)
    # pub4 = rospy.Publisher('/lcd_print',String,queue_size=1)
    rospy.init_node('talker', anonymous=True)#node name
    rate = rospy.Rate(10) # 10hz
    stop = False
    while not rospy.is_shutdown():
        # hello_str = "hello world %s" % rospy.get_time()
        # rospy.loginfo(hello_str)#in ra man hinh
        if not stop:
            pub2.publish(20)
            pub.publish(10) # public cho thang sub nhan
        else:
            pub2.publish(0)
            pub.publish(0) 
        key = cv2.waitKey(1)
        if key == ord('q'):
            stop = True
        else:
            stop = False
        rate.sleep()# dung du lau de thang kia nhan nhung van dam bao toc do thuc thi cua vong lap

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
