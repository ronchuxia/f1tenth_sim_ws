#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import Point
from tf_transformations import euler_from_quaternion


class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        self.odom_subscription = self.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # read waypoints from file
        self.waypoints = np.loadtxt("/sim_ws/src/pure_pursuit/waypoints/waypoints.csv", delimiter=',')[:,:2]

        # publisher for all waypoints (publish once)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.waypoints_publisher = self.create_publisher(Marker, '/rviz_waypoints', qos)
        self.visualize_waypoints()

        # publisher for target waypoint
        self.target_publisher = self.create_publisher(Marker, '/rviz_target', 10)

        self.l = 0.5
        self.p = 1.0
        self.v = 1.0

    def pose_callback(self, pose_msg):
        self.timestamp = pose_msg.header.stamp
        quaternion = np.array([pose_msg.pose.pose.orientation.x, 
                            pose_msg.pose.pose.orientation.y, 
                            pose_msg.pose.pose.orientation.z, 
                            pose_msg.pose.pose.orientation.w])
        euler = euler_from_quaternion(quaternion)
        euler_z = euler[2]
        
        # find the current waypoint to track
        self.pos = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])
        dist = np.linalg.norm(self.waypoints - self.pos, axis=1)

        smaller = (dist <= self.l)
        larger = (dist > self.l)
        target = np.bitwise_and(smaller[:-1], larger[1:])
        target_idx = target.nonzero()[0][0]

        self.get_logger().info(f"{target_idx}")

        waypoint_smaller = self.waypoints[target_idx]
        waypoint_larger = self.waypoints[target_idx+1]

        self.target_x = waypoint_larger[0]
        self.target_y = waypoint_larger[1]
        l = dist[target_idx+1]
        
        # transform goal point to vehicle frame of reference
        y = (self.target_y - self.pos[1]) * np.cos(euler_z) - (self.target_x - self.pos[0]) * np.sin(euler_z)

        # calculate curvature/steering angle
        self.gamma = 2 * y / l ** 2
        self.angle = np.clip(self.p * self.gamma, -0.7854, 0.7854)

        # publish drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = self.v
        drive_msg.drive.steering_angle = self.angle
        self.drive_publisher.publish(drive_msg)

        # visualize goal point
        self.visualize_target()

    def visualize_waypoints(self):
        points = Marker()
        # points.header.stamp = self.get_clock().now().to_msg()
        points.header.frame_id = '/map'
        points.ns = 'waypoints'
        points.id = 0
        points.type = Marker.POINTS
        points.action = Marker.ADD

        for waypoint in self.waypoints:
            point = Point()
            point.x = waypoint[0]
            point.y = waypoint[1]
            point.z = 0.0
            points.points.append(point)

        points.scale.x = 0.1
        points.scale.y = 0.1
        points.scale.z = 0.1

        points.color.r = 1.0
        points.color.g = 0.0
        points.color.b = 0.0
        points.color.a = 1.0

        self.waypoints_publisher.publish(points)

    def visualize_target(self):
        sphere = Marker()
        sphere.header.stamp = self.timestamp
        sphere.header.frame_id = '/map'
        sphere.ns = 'target'
        sphere.id = 0
        sphere.type = Marker.SPHERE

        sphere.pose.position.x = self.target_x
        sphere.pose.position.y = self.target_y
        sphere.pose.position.z = 0.0

        sphere.scale.x = 0.2
        sphere.scale.y = 0.2
        sphere.scale.z = 0.2

        sphere.color.r = 0.0
        sphere.color.g = 1.0
        sphere.color.b = 0.0
        sphere.color.a = 1.0

        self.target_publisher.publish(sphere)


def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
