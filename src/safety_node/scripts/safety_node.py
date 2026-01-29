#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
# TODO: include needed ROS msg type headers and libraries
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


class SafetyNode(Node):
    """
    The class that handles emergency braking.
    """
    def __init__(self):
        super().__init__('safety_node')
        """
        One publisher should publish to the /drive topic with a AckermannDriveStamped drive message.

        You should also subscribe to the /scan topic to get the LaserScan messages and
        the /ego_racecar/odom topic to get the current speed of the vehicle.

        The subscribers should use the provided odom_callback and scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        """
        self.speed = 0.
        
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped,
            'drive',
            10
        )

        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )

        self.odom_subscription = self.create_subscription(
            Odometry,
            'ego_racecar/odom',
            self.odom_callback,
            10
        )


    def odom_callback(self, odom_msg):
        v_x = odom_msg.twist.twist.linear.x

        # omega_z = odom_msg.twist.twist.angular.z

        self.speed = v_x

    def scan_callback(self, scan_msg):
        # TODO: Add AEB when turning
        ranges = np.array(scan_msg.ranges)  # No nan or inf values in the simulation
        num_ranges = len(ranges)

        angle_min = scan_msg.angle_min
        angle_max = scan_msg.angle_max
        angle_increment = scan_msg.angle_increment

        range_rates = - self.speed * np.cos(np.arange(0, num_ranges) * angle_increment + angle_min)
        ttc = np.maximum(ranges - 0.148, 1e-5) / np.maximum(-range_rates, 1e-5)  # Account for car shape (lidar offset + car width + wheel radius)
        min_ttc = np.min(ttc)

        min_ttc_index = np.argmin(ttc)
        min_ttc_angle = angle_min + min_ttc_index * angle_increment
        min_ttc_angle_deg = np.degrees(min_ttc_angle)

        ttc_threshold = 0.25  # seconds
        if min_ttc < ttc_threshold:
            self.get_logger().info(f"Brake, ttc: {min_ttc}, angle: {min_ttc_angle_deg}")
            self.emergency_brake()
        else:
            self.get_logger().info(f"Safe, ttc: {min_ttc}, angle: {min_ttc_angle_deg}")

    def emergency_brake(self):
        msg = AckermannDriveStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0

        self.drive_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    rclpy.spin(safety_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()