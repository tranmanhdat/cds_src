import config as cf
import time

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
            

            # if speed != cf.speed:
            if (not cf.pause and cf.ready) or speed == 0:
                print('\t set_speed', speed)
                cf.speed_pub.publish(speed)
                # rospy.loginfo('Published')
            # cf.rate.sleep()

    def set_steer(self, steer):
        if steer == 0 or (cf.running and not cf.pause and cf.ready):
            cf.change_steer = True
            print('\t set_angle', steer)
            cf.steer_pub.publish(steer)

    def set_lcd(self, text):
        cf.lcd_pub.publish(text)

    def clear_lcd(self):
        # clear lcd
        for row in range(4):
            self.set_lcd("0:{}:{}".format(row, ' '*20))

    def print_lcd(self, text):
        self.clear_lcd()
        time.sleep(0.1)
        self.set_lcd("0:0:"+text) #col:row:content
        time.sleep(0.1)


