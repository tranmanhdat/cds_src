import os
import cv2
import config as cf
from utils.Publish import Publish
import numpy as np
import time

class Control(Publish):
    error_arr = np.zeros(5)
    t = time.time()

    def __init__(self):
        super(Control, self).__init__()
        return

    def PID(self, error, p=0.43, i=0, d=0.02):
        self.error_arr[1:] = self.error_arr[0:-1]
        self.error_arr[0] = error
        P = error*p
        delta_t = time.time() - self.t
        self.t = time.time()
        D = (error-self.error_arr[1])/delta_t*d
        I = np.sum(self.error_arr)*delta_t*i
        angle = P + I + D
        if abs(angle) > 60:
            angle = np.sign(angle)*60
        return float(angle)

    def run_car_by_signal(self):
        '''
        Called when a signal is sent to start run the car.
        But car is run only when everything is loaded.
        cf.ready makes sure of that
        '''
        if cf.pause is True:
            cf.pause = False
            print('Send signal to run', cf.ready)
            self.print_lcd('Send signal to run')
            while cf.ready is False:
                print('Received signal to run but not ready! Wait a moment...')
                self.print_lcd("Wait a moment...")
                if cf.ready is True:
                    break
            if cf.ready is True:
                self.run_car_after_check()

    def run_car_after_check(self):
        print('Ready! Run now!!')
        self.print_lcd('Ready! Run now!!')
        cf.speed = cf.MAX_SPEED
        self.set_speed(cf.speed)

    def pause(self):
        if cf.pause is False:
            print('Pause!')
            time_to_sleep = cf.speed*0.5/15
            # before pause set speed to its negative value to go backwards
            cf.speed = -cf.speed
            self.set_speed(cf.speed)
            cf.pause = True
            time.sleep(time_to_sleep)
            # and = 0 to stop
            cf.speed = 0
            self.set_speed(cf.speed)
            # and reset angle = 0 for next time
            cf.steer = 0
            self.set_steer(cf.steer)

            self.print_lcd('Pause!')
    
    def quit(self):
        self.pause()
        
        # Unscribe all sensors
        if cf.sub_btn1 is not None:
            cf.sub_btn1.unregister()
            print('Unsubscribe button1')
        if cf.sub_btn2 is not None:
            cf.sub_btn2.unregister()
            print('Unsubscribe button2')
        if cf.sub_btn3 is not None:
            cf.sub_btn3.unregister()
            print('Unsubscribe button3')
        if cf.sub_sensor2 is not None:
            cf.sub_sensor2.unregister()
            print('Unsubscribe sensor2')

        if cf.sub_lidar is not None:
            cf.sub_lidar.unregister()
            print('Unsubscribe lidar')
        if cf.sub_getIMUAngle is not None:
            cf.sub_getIMUAngle.unregister()
            print('Unsubscribe IMU')

        cv2.destroyAllWindows()
        print('Close cv2 windows')
        # self.print_lcd('Quit!')
        # time.sleep(0.1)
        self.clear_lcd()
        time.sleep(1)
        print('QUit')
        os._exit(0)


class AutoControl(Control):
    arr_speed = np.zeros(5)
    timespeed = time.time()

    def __init__(self):
        super(AutoControl, self).__init__()
        
        self.turnControl = TurnControl(0,0)
        cf.specialCorner = 1
        return

    def speedControl(self, speed, angle):
        tempangle = abs(angle)
        # if tempangle < 5:
        #     if speed < cf.MAX_SPEED-3:
        #         speed = speed+1
        #     elif speed < cf.MAX_SPEED-1:
        #         speed = speed+0.3
        # elif tempangle < 15:
        #     if speed < cf.MAX_SPEED-3:
        #         speed = speed + 1
        #     elif speed < cf.MAX_SPEED-1:
        #         speed = speed+0.3
        # elif tempangle > 30:
        #     if speed > cf.MIN_SPEED+4:
        #         speed = speed-2
        #     elif speed > cf.MIN_SPEED+2:
        #         speed = speed-0.5
        # elif tempangle > 20:
        #     if speed > cf.MIN_SPEED+4:
        #         speed = speed-1
        #     elif speed > cf.MIN_SPEED+2:
        #         speed = speed-0.25
        if abs(angle) > 16 or cf.turnSignal:
            speed = cf.FIXED_SPEED_TURN
        else:
            speed = cf.MAX_SPEED
        
        # if speed != cf.speed:
        #     cf.speed = speed
        #     self.set_speed(cf.speed)
        # else:
        #     time.sleep(0.1)
        cf.speed = speed
        self.set_speed(cf.speed)
        # time.sleep(0.02)

        # return float(speed)

    def whereAmI(self):
        if cf.signSignal is not None: # sign detected! change the current route!
            cf.current_route = cf.signSignal
            if cf.signSignal == 'thang':
                if cf.start_tick_count is None:
                    cf.start_tick_count = 0

        if cf.imu_angle > 45-cf.imu_early and len(cf.turned) == 0: # turn first left
            cf.turned.append('left')
        if cf.imu_angle > 90-cf.imu_early and len(cf.turned) == 1: # turn second left
            cf.turned.append('left')
        if cf.imu_angle > 135-cf.imu_early and len(cf.turned) == 2: # turn third left
            cf.turned.append('left')
            cf.reduce_speed = True
        if cf.imu_angle > 90-cf.imu_early and len(cf.turned) == 3: # turn right at finish of segment 1
            cf.turned.append('right')
            cf.reduce_speed = False

        ''' Get to second segment '''
        if cf.current_route == 'phai':
            # skip some tick to just go straight without calculating the center
            if cf.imu_angle > 45-cf.imu_early and len(cf.turned) == 4: # turn right
                cf.turned.append('right')

            if cf.imu_angle > 90-cf.imu_early and len(cf.turned) == 5: # turn left
                cf.turned.append('left')
                cf.current_route = 'phai'
            if cf.imu_angle > 135-cf.imu_early and len(cf.turned) == 6: # turn left
                cf.turned.append('left')
            if cf.imu_angle > 180-cf.imu_early and len(cf.turned) == 7: # turn left
                cf.turned.append('left')

            if len(cf.turned) == 7: # we pass all the route. Now go straight to finish line
                go_straight = True

        if cf.current_route == 'thang':
            if cf.imu_angle > 45-cf.imu_early and len(cf.turned) == 4: # turn right
                cf.turned.append('right')

            if cf.imu_angle > 0-cf.imu_early and len(cf.turned) == 5: # turn right
                cf.turned.append('left')



    def auto_control(self):
        # while cf.running: # uncomment if run control in seperate thread
        # if True: # uncomment if call auto_control inside ImageProcessing
        if cf.ready and not cf.pause: # uncomment if call auto_control inside ImageProcessing

            self.whereAmI()

            if cf.start_tick_count is not None and cf.start_tick_count < cf.end_tick_from_start:
                cf.start_tick_count += 1
                cf.angle = 0
                cf.speed = cf.MAX_SPEED
                print('cf.start_tick_count', cf.start_tick_count, '~~~~ cf.end_tick_from_start', cf.end_tick_from_start)
            else: 
                cf.start_tick_count = None

            if cf.signSignal == 'thang':
                cf.angle = 0

            # if cf.reduce_speed is True:
            #     speed = cf.speed_reduced
            # else:
            #     speed = self.speedControl(cf.speed, cf.angle)
            self.speedControl(cf.speed, cf.angle)
            self.set_steer(cf.angle)
            # time.sleep(0.01)
            # self.angleControl()

    def angleControl(self):
        # Only publish when everything is ready
        if not cf.pause and cf.ready:
            # Set speed and angle
            self.set_steer(cf.angle)
            # cf.speed = speed
            # self.set_speed(cf.speed)
            # print(cf.speed, cf.angle)

class TurnControl(object):
    def __init__(self, curspeed, maxspeed):
        self.maxTimeTurn = 60
        self.timeTurn = 0
        self.speedmax = maxspeed
        self.speedmin = 32
        self.speedDelta = 2
        self.k = 1.0
        self.currentspeed = curspeed - self.speedDelta*self.timeTurn
        self.leftDelta = 1.2
        self.leftAngle = -10
        self.rightDelta = 1.2
        self.rightAngle = 10

    def update(self):
        self.timeTurn = self.timeTurn + 1
        temp = self.currentspeed - self.speedDelta*self.timeTurn
        if temp >= self.speedmin:
            self.currentspeed = temp
        if self.timeTurn == self.maxTimeTurn//2:
            self.k = -self.k*0.6
        self.leftAngle = self.leftAngle + self.leftDelta*self.k
        self.rightAngle = self.rightAngle - self.rightDelta*self.k
        print("turn control k " + str(self.k))


