https://github.com/fpt-corp/DiRa/tree/master/DiRa_Software/Jetson_TX2/Software/dira_gpio_controller?fbclid=IwAR1Js1jr6CyHGWj2HjT-ilA7NJDM0OJLrMBnmK8lByFUpMOOfKmntS_9pJI


cd ~/catkin_ws/src/mtapos/src
sudo chmod 666 /dev/ttyUSB0
roslaunch mtapos ros.launch
rosrun mtapos optimal.py
