import roslaunch
import rospy

process_generate_running = True

class ProcessListener(roslaunch.pmon.ProcessListener):
    global process_generate_running

    def process_died(self, name, exit_code):
        global process_generate_running
        process_generate_running = False
        rospy.logwarn("%s died with code %s", name, exit_code)


def init_launch(launchfile, process_listener):
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.parent.ROSLaunchParent(
        uuid,
        [launchfile],
        process_listeners=[process_listener],
    )
    return launch


rospy.init_node("dira_pca9685_controller")
launch_file = "/home/ubuntu/catkin_ws/src/dira_pca8266_controller/launch/run_car.launch"
launch = init_launch(launch_file, ProcessListener())
launch.start()

# while process_generate_running:
#     rospy.sleep(0.05)

launch.shutdown()
