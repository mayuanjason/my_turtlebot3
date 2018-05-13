#!/usr/bin/env python
# coding=utf-8

import rospy
from geometry_msgs.msg import Twist, Point
import tf2_ros
from math import radians, sqrt, pow, pi

from my_turtlebot3_navigation.transform_utils import quat_to_angle, normalize_angle


class OutAndBack:

    def __init__(self):
        # Give the node a name
        rospy.init_node('out_and_back', anonymous=False)

        # Set rospy to execute a shutdown function when exiting
        rospy.on_shutdown(self.shutdown)

        # Publisher to control the robot's speed
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=5)

        # How fast will we update the robot's movement
        rate = 20

        # Set the qeuivalent ROS rate variable
        r = rospy.Rate(rate)

        # Set the forward linear speed to 0.15 meters per second
        linear_speed = 0.15

        # Set the travel distance in meters
        goal_distance = 1.0

        # Set the rotation speed in radians per second
        angular_speed = 0.5

        # Set the angular tolerance in degrees converted to radians
        angular_tolerance = radians(1.0)

        # Set the rotation angle to Pi radians (180 degrees)
        goal_angle = pi

        # Initialize the tf listener
        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)

        # Give tf some time to fill its buffer
        rospy.sleep(2)

        # Set the odom frame
        self.odom_frame = 'odom'

        # Find out if the robot uses /bas_link or /base_footprint
        self.base_frame = 'base_footprint'

        # Initialize the position variable as a Point type
        position = Point()

        # Loop once for each leg of the trip
        for i in range(2):
            # Initialize the movement command
            move_cmd = Twist()

            # Set the movement command to forward motion
            move_cmd.linear.x = linear_speed

            # Get the starting position values
            (position, rotation) = self.get_odom()

            x_start = position.x
            y_start = position.y

            # Keep track of the distance traveled
            distance = 0

            # Enter the loop to move along a side
            while distance < goal_distance and not rospy.is_shutdown():
                # Publish the Twist messgae and sleep 1 cycle
                self.cmd_vel.publish(move_cmd)
                r.sleep()

                # Get the current position
                (position, rotation) = self.get_odom()

                # Compute the Euclidean distance from the start
                distance = sqrt(pow((position.x - x_start), 2) +
                                pow((position.y - y_start), 2))

            # Stop the robot before the rotation
            move_cmd = Twist()
            self.cmd_vel.publish(move_cmd)
            rospy.sleep(1)

            # Set the movement command to a rotation
            move_cmd.angular.z = angular_speed

            # Track the last angle measured
            last_angle = rotation

            # Track how far we have turned
            turn_angle = 0

            while abs(turn_angle + angular_tolerance) < abs(goal_angle) and not rospy.is_shutdown():
                # Publish the Twist message and sleep 1 cycle
                self.cmd_vel.publish(move_cmd)
                r.sleep()

                # Get the current rotation
                (position, rotation) = self.get_odom()

                # Compute the amount of rotation since the last loop
                delta_angle = normalize_angle(rotation - last_angle)

                # Add to the running total
                turn_angle += delta_angle
                last_angle = rotation

            # Stop the robot before the next leg
            move_cmd = Twist()
            self.cmd_vel.publish(move_cmd)
            rospy.sleep(1)

        # Stop the robot for good
        self.cmd_vel.publish(Twist())

    def get_odom(self):
        # Get the current transform between the odom and base frames
        try:
            trans = self.tfBuffer.lookup_transform(
                self.odom_frame, self.base_frame, rospy.Time(0), rospy.Duration(1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.loginfo("TF Exception")
            return

        x = trans.transform.translation.x
        y = trans.transform.translation.y
        z = trans.transform.translation.z

        return (Point(*[x, y, z]), quat_to_angle(trans.transform.rotation))

    def shutdown(self):
        # Always stop the robot when shutting down the node.
        rospy.loginfo("Stopping the robot...")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)


if __name__ == '__main__':
    try:
        OutAndBack()
    except rospy.ROSInterruptException:
        rospy.loginfo("Out-and-Back node terminated.")
