from utils.Control import Control
import config as cf
import time


class Subscribe(Control):
    def __init__(self):
        super(Subscribe, self).__init__()
        return

    def on_get_imu_angle(self, res):
        data = res.data
        if cf.first_imu_angle is None:
            cf.first_imu_angle = data
        
        cf.imu_angle = - (data - cf.first_imu_angle)
        # print('imu_angle', cf.imu_angle)
        time.sleep(0.01)

    def on_get_lidar(self, data):
        self.lidar_data = data
        time.sleep(0.2)

    def on_get_sensor2(self, res):
        if res.data is True and cf.do_detect_barrier is True: # free in front (barrier open)
            self.run_car_by_signal()
            cf.do_detect_barrier = False
            cf.sub_sensor2.unregister()
            time.sleep(0.02)
            cf.sub_sensor2 = None
        if res.data is False: # something in front closely
            self.pause()
        # time.sleep(0.2)
    def on_get_btn_1(self, res):
        '''
        If button 2 is clicked, set mode to start
        '''
        if res.data is True: # click
            self.print_lcd('Running!')
            print('Running!')
            cf.running = True
            if cf.sub_sensor2 is None:
                self.run_car_by_signal()
            else:
                cf.do_detect_barrier = True
            cf.start_tick_count = 0
            # self.run_car_by_signal()
            # cf.do_detect_barrier = False
            # cf.sub_btn1.unregister()
        # time.sleep(0.2)
    def on_get_btn_2(self, res):
        '''
        If button 2 is clicked, pause!
        '''
        if res.data is True: # click
            self.pause()
        # time.sleep(0.1)
    def on_get_btn_3(self, res):
        '''
        If button 3 is clicked, quit!
        '''
        if res.data is True: # click
            self.quit()
        # time.sleep(0.1)

