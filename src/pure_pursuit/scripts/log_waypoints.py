#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from tf_transformations import euler_from_quaternion
from time import gmtime, strftime
from numpy import linalg as LA
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped


class PointLogger(Node):
    def __init__(self, mode='sim'):
        super().__init__('point_logger_node')

        if mode == 'pf':
            self.odom_subscription = self.create_subscription(Odometry, '/pf/pose/odom', self.save_waypoint_from_odometry, 10)  # real life, particle filter
        elif mode == 'sim':
            self.odom_subscription = self.create_subscription(Odometry, '/ego_racecar/odom', self.save_waypoint_from_odometry, 10)    # simulation, odom
        elif mode == 'rviz':
            self.odom_subscription = self.create_subscription(PointStamped, '/clicked_point', self.save_waypoint_from_pointstamped, 10)   # simulation, clicked point

        self.file_name = strftime('/sim_ws/src/pure_pursuit/waypoints/wp-%Y-%m-%d-%H-%M-%S', gmtime()) + '.csv'
        self.file = open(self.file_name, 'w')

    def save_waypoint_from_odometry(self, data):
        quaternion = np.array([data.pose.pose.orientation.x, 
                            data.pose.pose.orientation.y, 
                            data.pose.pose.orientation.z, 
                            data.pose.pose.orientation.w])
        euler = euler_from_quaternion(quaternion)

        speed = LA.norm(np.array([data.twist.twist.linear.x, 
                                data.twist.twist.linear.y, 
                                data.twist.twist.linear.z]),2)

        if data.twist.twist.linear.x > 1e-5:
            self.file.write('%f, %f, %f, %f\n' % (data.pose.pose.position.x,
                                            data.pose.pose.position.y,
                                            euler[2],
                                            speed))
        
            self.get_logger().info("Point logged.")
        
    def save_waypoint_from_pointstamped(self, data):
        self.file.write('%f, %f\n' % (data.point.x, data.point.y))
        self.get_logger().info("Point logged.")

    def destroy_node(self):
        self.file.close()
        super().destroy_node()


def main(args=None):
    mode = 'sim'

    rclpy.init(args=args)
    print("PointLogger Initialized")
    point_logger_node = PointLogger(mode)
    file_name = point_logger_node.file_name
    rclpy.spin(point_logger_node)

    point_logger_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()