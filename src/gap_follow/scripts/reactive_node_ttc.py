#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

from collections import deque

class ReactiveFollowGap(Node):
    """ 
    Implement Wall Following on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('reactive_node')
        # Topics & Subs, Pubs
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # Subscribe to LIDAR
        self.lidarscan_subscription = self.create_subscription(LaserScan, lidarscan_topic, self.lidar_callback, 10)
        # Publish to drive
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        self.odom_subscription = self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 10)

        self.v_x = 0.
        self.omega_z = 0.

        # A window of ranges
        self.window_size = 3   # dt ~ 0.005s * window_size
        self.ranges_window = deque()

        self.bubble_radius = 0.25
        self.debug = True

        self.angle_speed_publisher = self.create_publisher(Marker, 'angle_speed', 10)
        self.closest_point_publisher = self.create_publisher(Marker, 'closest_point', 10)
        self.widest_gap_publisher = self.create_publisher(Marker, 'widest_gap', 10)
        self.lidar_timestamp = None

    def odom_callback(self, odom_msg):
        v_x = odom_msg.twist.twist.linear.x
        omega_z = odom_msg.twist.twist.angular.z

        self.v_x = v_x
        self.omega_z = omega_z

    def preprocess_lidar(self, data):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        ranges = np.array(data.ranges)
        ranges[np.isnan(ranges)] = data.range_min
        ranges[np.isinf(ranges)] = data.range_max

        # Maintain a window of ranges
        if len(self.ranges_window) >= self.window_size:
            self.ranges_window.popleft()
        self.ranges_window.append(ranges)
        
        # Compute mean over window and clip
        proc_ranges = np.mean(np.stack(self.ranges_window), axis=0)
        proc_ranges = np.clip(proc_ranges, data.range_min, data.range_max)

        num_ranges = len(ranges) 
        angle_min = data.angle_min
        angle_increment = data.angle_increment
        angles = np.arange(0, num_ranges) * angle_increment + angle_min

        idx_forward = np.abs(angles) < np.pi / 2.0
        proc_ranges = proc_ranges[idx_forward]
        angles = angles[idx_forward]

        return proc_ranges, angles

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        mask = ~np.isclose(free_space_ranges, 0.0)
        padded = np.concatenate(([False], mask, [False]))                                                           
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == 1)[0]                                                                            
        ends = np.where(diffs == -1)[0]                                                                           
        if len(starts) == 0:
            return 0, 0
        best = np.argmax(ends - starts)
        return starts[best], ends[best]
    
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        gap = ranges[start_i:end_i]
        best_point_idx = np.argmax(gap) + start_i
        return best_point_idx

    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        self.lidar_timestamp = data.header.stamp
        proc_ranges, angles = self.preprocess_lidar(data)

        # Find closest point (ttc)
        range_rates = - self.v_x * np.cos(angles) - self.omega_z * 0.275 * np.sin(angles)
        ttc = np.maximum(proc_ranges - 0.148, 1e-5) / np.maximum(-range_rates, 1e-5)

        closest_point_idx = np.argmin(ttc)

        # Eliminate all points inside 'bubble' (set them to zero) 
        dist_to_closest = proc_ranges[closest_point_idx] * np.sin(np.abs(angles - angles[closest_point_idx]))  
        points_in_bubble = dist_to_closest < self.bubble_radius
        free_space_ranges = np.where(points_in_bubble, 0.0, proc_ranges)   # shape (num_ranges, )

        # Find max length gap 
        widest_gap_start, widest_gap_end = self.find_max_gap(free_space_ranges)

        if widest_gap_end == widest_gap_start:  # No gap found
            return

        # Find the best point in the gap
        best_point_idx = self.find_best_point(widest_gap_start, widest_gap_end, free_space_ranges)
        best_point_angle = angles[best_point_idx]

        angle_deg = best_point_angle / math.pi * 180
        if abs(angle_deg) < 10:
            velocity = 1.5
        elif abs(angle_deg) < 20:
            velocity = 1.0
        else:
            velocity = 0.5

        if self.debug:
            self.draw_angle_speed(best_point_angle, velocity)
            self.draw_closest_point(proc_ranges[closest_point_idx], angles[closest_point_idx], self.bubble_radius)
            self.draw_widest_gap(proc_ranges[widest_gap_start:widest_gap_end], angles[widest_gap_start:widest_gap_end])

        # Publish Drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = velocity
        drive_msg.drive.steering_angle = best_point_angle
        self.drive_publisher.publish(drive_msg)
        
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

        self.angle_speed_publisher.publish(arrow) 

    def draw_closest_point(self, range, angle, radius):
        point = Marker()
        point.header.stamp = self.lidar_timestamp
        point.header.frame_id = 'ego_racecar/laser'
        point.ns = 'closest_point'
        point.id = 0
        point.type = Marker.SPHERE
        point.action = Marker.ADD

        point.pose.position.x = range * np.cos(angle)
        point.pose.position.y = range * np.sin(angle)
        point.pose.position.z = 0.0

        point.scale.x = radius
        point.scale.y = radius
        point.scale.z = 0.01

        point.color.r = 1.0
        point.color.g = 0.0
        point.color.b = 0.0
        point.color.a = 1.0

        self.closest_point_publisher.publish(point)
    
    def draw_widest_gap(self, ranges, angles):
        points = Marker()
        points.header.stamp = self.lidar_timestamp
        points.header.frame_id = 'ego_racecar/laser'
        points.ns = 'widest_gap'
        points.id = 0
        points.type = Marker.POINTS
        points.action = Marker.ADD

        for i in range(len(ranges)):
            point = Point()
            point.x = ranges[i] * np.cos(angles[i])
            point.y = ranges[i] * np.sin(angles[i])
            point.z = 0.0
            points.points.append(point)
        
        points.scale.x = 0.05
        points.scale.y = 0.05
        points.scale.z = 0.05

        points.color.r = 0.0
        points.color.g = 0.0
        points.color.b = 1.0
        points.color.a = 1.0

        self.widest_gap_publisher.publish(points)


def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()