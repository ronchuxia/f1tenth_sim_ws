#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
# TODO: include needed ROS msg type headers and libraries
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


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
        self.v_x = 0.
        self.omega_z = 0.
        
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

        self.marker_publisher = self.create_publisher(
            Marker,
            'ttc_marker',
            10
        )


    def odom_callback(self, odom_msg):
        v_x = odom_msg.twist.twist.linear.x
        omega_z = odom_msg.twist.twist.angular.z

        self.v_x = v_x
        self.omega_z = omega_z

    def scan_callback(self, scan_msg):
        ranges = np.array(scan_msg.ranges)  # No nan or inf values in the simulation
        num_ranges = len(ranges)

        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment
        angles = np.arange(0, num_ranges) * angle_increment + angle_min

        offsets = np.where((angles >= -90) & (angles <= 90), 0.148, 0.171)

        range_rates = - self.v_x * np.cos(angles) - self.omega_z * 0.275 * np.sin(angles)
        ttc = np.maximum(ranges - offsets, 1e-5) / np.maximum(-range_rates, 1e-5)  # Account for car shape (lidar offset + car width + wheel radius)
        min_ttc = np.min(ttc)

        min_ttc_index = np.argmin(ttc)
        min_ttc_angle = angle_min + min_ttc_index * angle_increment

        ttc_threshold = 0.25  # seconds
        is_braking = min_ttc < ttc_threshold

        self.publish_ttc_marker(min_ttc_angle, min_ttc, is_braking)

        if is_braking:
            self.get_logger().info(f"Brake, ttc: {min_ttc}")
            self.emergency_brake()
        else:
            self.get_logger().info(f"Safe, ttc: {min_ttc}")

    def publish_ttc_marker(self, angle, ttc, is_braking):
        arrow_length = 0.5

        # Arrow marker
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

        if is_braking:
            arrow.color.r = 1.0
            arrow.color.g = 0.0
        else:
            arrow.color.r = 0.0
            arrow.color.g = 1.0
        arrow.color.b = 0.0
        arrow.color.a = 1.0

        self.marker_publisher.publish(arrow)

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