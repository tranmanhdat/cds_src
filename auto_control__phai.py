import config as cf

class AutoControl(Control):
    arr_speed = np.zeros(5)
    timespeed = time.time()
    tick_to_finish__route_thang = None
    fixed_speed = cf.fixed_speed # init speed to increase by time
    tick_chuyen_lane = None
    tick_giu_lane = None
    angle_giu_lane = None
    last_angles_chuyen_lane = []
    tick_start_save_angle = cf.tick_stop_chuyen_lane - cf.tick_stop_giu_lane
    do_not_chuyen_lane = False
    wait_pass_2_turn_to_chuyen_lane = False
    
    def __init__(self):
        super(AutoControl, self).__init__()
        
        return

    def speedControl(self, angle, fixed_speed=None):
        tempangle = abs(angle)
        if fixed_speed is not None:
            speed = fixed_speed
            ###print('\t\t fixed_speed', fixed_speed)
        elif cf.reduce_speed == 1:
            speed = cf.speed_reduced
            ###print('\t\t Reduce speed !!!!!!!!!!!!!')
        elif cf.reduce_speed == -1:
            speed = cf.MAX_SPEED
            ###print('\t\t Speed up !!!!!!!!!!!!!')
        else: # cf.reduce_speed == 0
            if abs(angle) > 18:
                speed = cf.FIXED_SPEED_TURN
            else:
                speed = cf.FIXED_SPEED_STRAIGHT

        cf.speed = speed
        self.set_speed(cf.speed)
        # time.sleep(0.02)

    def whereAmI(self):
        if cf.imu_angle > 90-cf.imu_early-20 and len(cf.turned) == 0: # turn first left
            cf.turned.append('left')
            cf.reduce_speed = 0 # go at normal speed
        if 180+cf.imu_early > cf.imu_angle > 180-cf.imu_early and len(cf.turned) == 1: # turn second left
            cf.turned.append('left')
        if 270+cf.imu_early > cf.imu_angle > 270-cf.imu_early and len(cf.turned) == 2: # turn third left
            cf.turned.append('left')
            # cf.reduce_speed = 1 # need to reduce speed to turn the next right and detect sign
        if 180+cf.imu_early > cf.imu_angle > 180-cf.imu_early and len(cf.turned) == 3: # turn right at finish of segment 1
            cf.turned.append('right')
            cf.do_detect_sign = True # now need to detect sign
            cf.count_sign_step = 0
            self.fixed_speed = 10 # go really slow to detect sign !!!

        if cf.current_route is not None and cf.start_tick_count is None: # detect sign successfully!
            self.fixed_speed = None


        ''' Get to second segment '''
        # route phai
        if cf.current_route == 'phai':
            # After detecting sign and turn successfully, set reduce_speed = 0 to go at normal speed (set in signRecognize)
            
            if 90+cf.imu_early > cf.imu_angle > 90-cf.imu_early and len(cf.turned) == 4: # turn right
                cf.turned.append('right')

            if 180+cf.imu_early > cf.imu_angle > 180-cf.imu_early and len(cf.turned) == 5: # turn left
                cf.turned.append('left')
            if 270+cf.imu_early > cf.imu_angle > 270-cf.imu_early and len(cf.turned) == 6: # turn left
                cf.turned.append('left')
            if 360+cf.imu_early > cf.imu_angle > 360-cf.imu_early and len(cf.turned) == 7: # turn left
                cf.turned.append('left')
                # we pass all the route. Now speed up to finish line!
                cf.reduce_speed = -1
                self.tick_to_finish__route_thang = 0

            if len(cf.turned) == 8: # after some tick_to_finish__route_thang we would need to slow down to detect stop sign
                self.tick_to_finish__route_thang += 1
                if self.tick_to_finish__route_thang > 50:
                    cf.reduce_speed = 1

        # route thang
        if cf.current_route == 'thang':
            # After detecting sign successfully, set reduce_speed = 0 to go at normal speed AND set cf.run_lidar = True to detect moving object (set in signRecognize)

            if 90+cf.imu_early > cf.imu_angle > 90-cf.imu_early and len(cf.turned) == 4: # turn right
                cf.turned.append('right')

            if 0+cf.imu_early > cf.imu_angle > 0-cf.imu_early and len(cf.turned) == 5: # turn right
                cf.turned.append('right')
            if -90+cf.imu_early > cf.imu_angle > -90-cf.imu_early and len(cf.turned) == 6: # turn right
                cf.turned.append('right')
            if 0+cf.imu_early > cf.imu_angle > 0-cf.imu_early and len(cf.turned) == 7: # turn left
                cf.turned.append('left')

        if len(cf.turned) == 8:
            # turn on sign detector
            cf.do_detect_sign = True 
            cf.count_sign_step = 0

        print('   >> ', cf.current_route, cf.lane, cf.imu_angle, cf.turned, cf.speed, cf.angle, cf.fps_count)


    def auto_control(self, angle_from_center):
        # while cf.running: # uncomment if run control in seperate thread
        if cf.ready and not cf.pause: # uncomment if call auto_control inside getCenter

            self.whereAmI()

            # just go straight first few ticks
            if cf.start_tick_count is not None:
                if cf.start_tick_count < cf.end_tick_from_start:
                    if cf.start_tick_count == 0:
                        self.fixed_speed = cf.fixed_speed
                    cf.start_tick_count += 1
                    cf.angle = 0
                    # cf.speed = cf.MAX_SPEED
                    # cf.reduce_speed = -1
                    if cf.start_tick_count % 5 == 1 and self.fixed_speed < cf.MAX_SPEED:
                        self.fixed_speed += 1
                    ###print('cf.start_tick_count', cf.start_tick_count, '~~~~ cf.end_tick_from_start', cf.end_tick_from_start)
                else:
                    cf.start_tick_count = None
                    cf.reduce_speed = 1
                    self.fixed_speed = None
            
            elif cf.do_chuyen_lane > 0: # khong chuyen lane cho re
                self.fixed_speed = cf.speed_chuyen_lane
                if cf.do_chuyen_lane == 1: # chuyen trai
                    # cf.angle = angle_from_center + cf.goc_chuyen_lane
                    cf.angle = cf.goc_chuyen_lane
                    cf.lane = 'trai'
                    if len(cf.turned) == 4: # chuyen o doan dau
                        cf.segment_chuyen_lane = 1
                    elif len(cf.turned) == 5:
                        cf.segment_chuyen_lane = 2
                    elif len(cf.turned) == 6:
                        cf.segment_chuyen_lane = 3
                    
                elif cf.do_chuyen_lane == 2: # chuyen phai
                    cf.FIXED_SPEED_STRAIGHT = cf.FIXED_SPEED_route_difficult
                    # cf.angle = angle_from_center - cf.goc_chuyen_lane
                    cf.angle = -cf.goc_chuyen_lane
                    cf.lane = 'phai'

                # if self.tick_chuyen_lane is None and abs(angle_from_center) < 16:
                if self.tick_chuyen_lane is None and self.do_not_chuyen_lane is False:
                        self.tick_chuyen_lane = 0
                if self.tick_chuyen_lane is not None:
                    self.tick_chuyen_lane += 1
                
                    # if self.tick_chuyen_lane >= self.tick_start_save_angle:
                    if self.tick_chuyen_lane <= cf.tick_stop_giu_lane:
                        self.last_angles_chuyen_lane.append(-cf.angle)
                        # print('self.last_angles_chuyen_lane', cf.angle, self.last_angles_chuyen_lane)

                        self.angle_giu_lane = -cf.angle*1.2

                    if self.tick_chuyen_lane > cf.tick_stop_chuyen_lane:
                        cf.do_chuyen_lane = -cf.do_chuyen_lane
                        self.fixed_speed = None
                        self.tick_chuyen_lane = None

                        self.tick_giu_lane = 0
                print('\t\t ----> cf.do_chuyen_lane', cf.do_chuyen_lane, self.tick_chuyen_lane, cf.lane)

            # elif len(cf.turned) > 6 and cf.do_chuyen_lane == 2 and self.tick_giu_lane is not None: # ket thuc chuyen lane phai o vi tri nhay cam!
            elif self.tick_giu_lane is not None: # ket thuc chuyen lane phai o vi tri nhay cam!
                print('\t\t\t ~~~~~~ self.tick_giu_lane', self.tick_giu_lane, self.last_angles_chuyen_lane)
                cf.angle = self.angle_giu_lane
                # cf.angle = self.last_angles_chuyen_lane[self.tick_giu_lane]

                self.tick_giu_lane += 1
                if self.tick_giu_lane > cf.tick_stop_giu_lane:
                    self.tick_giu_lane = None
                    self.last_angles_chuyen_lane = []

                    if cf.do_chuyen_lane == -1:
                        # tang toc de vuot
                        cf.FIXED_SPEED_STRAIGHT = cf.SPEED_PASS_OBJ

            else:
                if cf.current_route == 'thang':
                    # if len(cf.turned) == 3: # turn third left
                    #     # cho nay khong the re trai, neu phat hien goc > 16, ep goc ve -angle
                    #     if angle_from_center > 16:
                    #         angle_from_center = -angle_from_center
                    #         ###print('ep goc!!', angle_from_center)

                    # elif len(cf.turned) == 7: # last turn left of route thang
                    #     # cho nay khong the re phai, neu phat hien goc < -30, ep goc ve -angle
                    #     if angle_from_center < -36:
                    #         angle_from_center = -angle_from_center
                    #         print('ep goc khuc cuoi!!', angle_from_center)

                    if cf.tick_to_pass_turn is not None:
                        print('\t\t ~~~~~~ cf.tick_to_pass_turn', cf.tick_to_pass_turn)
                        # cho nay khong the re phai or re trai, neu phat hien goc > 17, ep goc ve 0
                        if abs(angle_from_center) > 16:
                            angle_from_center = 0
                        
                        cf.tick_to_pass_turn += 1
                        if cf.tick_to_pass_turn > cf.max_tick_to_pass_turn:
                            cf.tick_to_pass_turn = None
                            # do lidar scan to detect moving object
                            cf.run_lidar = True

                
                cf.angle = angle_from_center
            
            
            if cf.do_chuyen_lane == -1: # da chuyen sang lane trai (cf.lane = 'trai')
                if cf.segment_chuyen_lane == 1:
                    if len(cf.turned) == 5 and self.wait_pass_2_turn_to_chuyen_lane is False:
                        if cf.pass_object is True:
                            cf.do_chuyen_lane = 2 # chuyen phai
                        else: # chua vuot duoc vat
                            self.wait_pass_2_turn_to_chuyen_lane = True
                    elif len(cf.turned) == 6: # chuyen ve di thoi
                        if cf.pass_object is True:
                            cf.do_chuyen_lane = 2 # chuyen phai
                            self.wait_pass_2_turn_to_chuyen_lane = False # reset
                elif cf.segment_chuyen_lane == 2:
                    if len(cf.turned) == 6 and cf.pass_object is True: # qua doa va vuot vat
                        cf.do_chuyen_lane = 2 # chuyen phai
                elif cf.segment_chuyen_lane == 3:
                    if cf.pass_object is True:
                        cf.do_chuyen_lane = 2 # chuyen phai


            self.speedControl(cf.angle, self.fixed_speed)
            self.set_steer(cf.angle)


