# 1. Timed Out-and-Back
## 1.1. Timed Out-and-Back in the RViz Simulator
**[Remote PC]**
```
$ export TURTLEBOT3_MODEL=burger
$ roslaunch turtlebot3_fake turtlebot3_fake.launch  # 执行这条命令后 rviz 会自动打开
$ rosrun my_turtlebot3_navigation timed_out_and_back.py
```

## 1.2. Timed Out and Back using a Real Robot
**[TurtleBot]** Bring up basic packages to start TurtleBot3 applications.
```
$ roslaunch turtlebot3_bringup turtlebot3_core.launch
```
**[Remote PC]**
```
$ roslaunch turtlebot3_bringup turtlebot3_remote.launch
$ rosrun rviz rviz -d `rospack find turtlebot3_description`/rviz/model.rviz
$ rosrun my_turtlebot3_navigation odom_out_and_back.py
```

# 2. Out and Back Using Odometry
## 2.1. Odometry-Based Out and Back in the RViz Simulator
**[Remote PC]**
```
$ export TURTLEBOT3_MODEL=burger
$ roslaunch turtlebot3_fake turtlebot3_fake.launch  # 执行这条命令后 rviz 会自动打开
$ rosrun my_turtlebot3_navigation timed_out_and_back.py
```

## 2.2. Odometry-Based Out and Back Using a Real Robot
```
$ roslaunch turtlebot3_bringup turtlebot3_remote.launch
$ rosrun rviz rviz -d `rospack find turtlebot3_description`/rviz/model.rviz
$ rosrun my_turtlebot3_navigation odom_out_and_back.py
```

# 3. Navigating a Square using Odometry
## 3.1. Navigating a Square in the RViz Simulator
```
$ export TURTLEBOT3_MODEL=burger
$ roslaunch turtlebot3_fake turtlebot3_fake.launch  # 执行这条命令后 rviz 会自动打开
$ rosrun my_turtlebot3_navigation nav_square.py
```

## 3.2. Navigating a Square using a Real Robot
```
$ roslaunch turtlebot3_bringup turtlebot3_remote.launch
$ rosrun rviz rviz -d `rospack find turtlebot3_description`/rviz/model.rviz
$ rosrun my_turtlebot3_navigation nav_square.py
```