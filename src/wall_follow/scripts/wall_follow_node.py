#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import time
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Float64MultiArray


class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('wall_follow_node')

        self.debug = True

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # create subscribers and publishers
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            drive_topic,
            10
        )

        self.scan_subscription = self.create_subscription(
            LaserScan,
            lidarscan_topic,
            self.scan_callback,
            10
        )

        # set PID gains
        self.kp = 1.0
        self.kd = 0.1
        self.ki = 0.3
        self.alpha = 0.2

        # store history
        self.integral = 0.0
        self.prev_error = None
        self.error = None
        self.filtered_error = None
        self.prev_time = None
        self.time = None

        # store any necessary values you think you'll need
        self.theta = 45
        self.dist = 1.0 

        self.angle_min = - 2.35
        self.angle_increment = 0.004352

        self.vis_publisher = self.create_publisher(Marker, 'angle_speed', 10)
        self.plot_publisher = self.create_publisher(Float64MultiArray, 'pid_debug', 10)

    def get_range(self, range_data, angle):
        """
        Simple helper to return the corresponding range measurement at a given angle. Make sure you take care of NaNs and infs.

        Args:
            range_data: single range array from the LiDAR
            angle: between angle_min and angle_max of the LiDAR

        Returns:
            range: range measurement in meters at the given angle

        """

        angle = (angle / 180) * math.pi
        index = round((angle - self.angle_min) / self.angle_increment)
        range = range_data[index]

        return range

    def get_error(self, range_data, dist):
        """
        Calculates the error to the wall. Follow the wall to the left (going counter clockwise in the Levine loop). You potentially will need to use get_range()

        Args:
            range_data: single range array from the LiDAR
            dist: desired distance to the wall

        Returns:
            error: calculated error
        """
        a = self.get_range(range_data, self.theta)
        b = self.get_range(range_data, 90)

        alpha = math.atan((b - a * math.cos(self.theta)) / (a * math.sin(self.theta)))

        distance = b * math.cos(alpha)
        error = distance - dist
        return error

    def pid_control(self):
        """
        Based on the calculated error, publish vehicle control

        Returns:
            None
        """
        # Skip the first measurement
        if not self.prev_error:
            self.filtered_error = self.error
            self.prev_error = self.filtered_error
            self.prev_time = self.time
            return 

        # TODO: make d term more stable
        self.filtered_error = self.alpha * self.error + (1 - self.alpha) * self.prev_error

        dt = self.time - self.prev_time
        self.integral += self.error * dt

        angle = self.kp * self.error + self.kd * (self.filtered_error - self.prev_error) / dt + self.ki * self.integral

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'

        # compute desired velocity based on steering angle
        angle_deg = angle / math.pi * 180
        if abs(angle_deg) < 10:
            velocity = 1.5
        elif abs(angle_deg) < 20:
            velocity = 1.0
        else:
            velocity = 0.5
        self.get_logger().info(f"angle: {angle}, d:{self.kd * (self.filtered_error - self.prev_error) / dt}")

        # publish
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = angle
        self.drive_publisher.publish(drive_msg)

        # visualization
        if self.debug:
            self.draw_angle_speed(angle, velocity)
            plot_msg = Float64MultiArray()
            plot_msg.data = [
                self.error,
                self.kp * self.error,
                self.kd * (self.error - self.prev_error) / dt,
                self.ki * self.integral,
                angle
            ]
            self.plot_publisher.publish(plot_msg)

        self.prev_error = self.filtered_error
        self.prev_time = self.time

    def scan_callback(self, msg):
        """
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """
        range_data = np.array(msg.ranges)
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment
        self.time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        range_data[np.isnan(range_data)] = 1e-5
        range_data[np.isinf(range_data)] = msg.range_max

        self.error = self.get_error(range_data, self.dist)   # compute error
        self.pid_control() # actuate the car with PID

    def draw_angle_speed(self, angle, speed):
        arrow_length = speed

        arrow = Marker()
        arrow.header.stamp = self.get_clock().now().to_msg()
        arrow.header.frame_id = 'ego_racecar/laser'
        arrow.ns = 'ttc_arrow'
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD

        arrow.points = []
        start = Point()
        start.x = 0.0
        start.y = 0.0
        start.z = 0.0
        end = Point()
        end.x = arrow_length * np.cos(angle)
        end.y = arrow_length * np.sin(angle)
        end.z = 0.0
        arrow.points.append(start)
        arrow.points.append(end)

        arrow.scale.x = 0.05  # shaft diameter
        arrow.scale.y = 0.1   # head diameter

        arrow.color.r = 0.0
        arrow.color.g = 1.0
        arrow.color.b = 0.0
        arrow.color.a = 1.0

        self.vis_publisher.publish(arrow)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()