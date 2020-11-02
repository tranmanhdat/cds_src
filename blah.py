            if cf.imu_angle > 45 and len(cf.turned) == 0: # turn first left
                cf.turned.append('left')
            if cf.imu_angle > 90 and len(cf.turned) == 1: # turn second left
                cf.turned.append('left')
            if cf.imu_angle > 135 and len(cf.turned) == 2: # turn third left
                cf.turned.append('left')
            if cf.imu_angle > 90 and len(cf.turned) == 3: # turn right at finish of segment 1
                cf.turned.append('right')

            ''' Get to second segment '''
            if cf.imu_angle > 45 and len(cf.turned) == 4: # turn right
                cf.turned.append('right')
            if cf.imu_angle > 90 and len(cf.turned) == 5: # turn left. It means we saw right sign previously. Change the route to route['phai']
                cf.turned.append('left')
                cf.current_route = 'phai'

            if cf.current_route == 'phai':
                if cf.imu_angle > 135 and len(cf.turned) == 6: # turn left
                    cf.turned.append('left')
                if cf.imu_angle > 180 and len(cf.turned) == 7: # turn left
                    cf.turned.append('left')

                if len(cf.turned) == 7: # we pass all the route. Now go straight to finish line
                    go_straight = True

            if cf.imu_angle > 0 and len(cf.turned) == 5: # turn right. It means we saw straight sign previously. Change the route to route['thang']
                cf.turned.append('left')
                cf.current_route = 'phai'
