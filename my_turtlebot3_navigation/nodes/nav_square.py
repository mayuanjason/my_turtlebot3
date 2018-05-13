#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist, Point
import tf2_ros
from math import radians, sqrt, pow

from my_turtlebot3_navigation.transform_utils import quat_to_angle, normalize_angle


class NavSquare:

    def __init__(self):
        # Give the node a name
        rospy.init_node('nav_square', anonymous=False)

        # Set rospy to execute a shutdown function when terminating the script
        rospy.on_shutdown(self.shutdown)

        # How fast will we update the robot's movement
        rate = 20

        # Set the equivalent ROS rate variable
        r = rospy.Rate(rate)

        # Set the parameters for the target square
        goal_diatance = rospy.get_param("~goal_distance", 1.0)  # meters

        # degrees converted to radians
        goal_angle = radians(rospy.get_param("~goal_angle", 90))
        linear_speed = rospy.get_param(
            "~linear_speed", 0.2)  # meters per second
        angular_speed = rospy.get_param(
            "~angular_speed", 0.7)  # radians per second
        angular_tolerance = radians(rospy.get_param(
            "~angular_tolerance", 2))  # degrees to radians

        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=5)

        # The base frame is base footprint for the TurtleBot
        self.base_frame = rospy.get_param("~base_frame", 'base_footprint')

        # The odom frame is usually just /odom
        self.odom_frame = rospy.get_param("~odom_frame", 'odom')

        # Initialize the tf listener
        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)

        # Give tf some time to fill its buffer
        rospy.sleep(2)

        # Initialize the position variable as a Point type
        position = Point()

        # Cycle through the four sides of the square
        for i in range(4):
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
            while distance < goal_diatance and not rospy.is_shutdown():
                # Publisher the Twist message and sleep 1 cycle
                self.cmd_vel.publish(move_cmd)
                r.sleep()

                # Get the current position
                (position, rotation) = self.get_odom()

                # Compute the Eculidean distance from the start
                distance = sqrt(pow((position.x - x_start), 2) +
                                pow((position.y - y_start), 2))

            # stop the robot before rotating
            move_cmd = Twist()
            self.cmd_vel.publish(move_cmd)
            rospy.sleep(1)

            # Set the movement command to a rotation
            move_cmd.angular.z = angular_speed

            # Track the last angle measured
            last_angle = rotation

            # Track how far we have turned
            turn_angle = 0

            # Begin the rotation
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
        NavSquare()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation terminated.")
