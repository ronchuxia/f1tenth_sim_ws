#!/usr/bin/env python3
"""
This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf

TODO: 
- Implement a probabilistic occupancy grid.
    - Shift the occupancy grid according to odom.
    - Update the occupancy grid according to lidar scans.
"""
import numpy as np
from numpy import linalg as LA
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import OccupancyGrid

from scipy.ndimage import binary_dilation
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import Point
from tf_transformations import euler_from_quaternion


class RRTree(object):
    def __init__(self):
        self.pos = np.array([[0.0, 0.0]]) # shape (num_nodes, 2)
        self.parent = np.array([-1])
        self.cost = None # only used in RRT*


# class def for RRT
class RRT(Node):
    def __init__(self):
        super().__init__('rrt_node')

        self.initialize_parameters()

        self.pose_sub_ = self.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback, 1)
        self.scan_sub_ = self.create_subscription(LaserScan, '/scan', self.scan_callback, 1)

        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.occupancy_grid_pub_ = self.create_publisher(OccupancyGrid, '/rviz_occupancy_grid', 10)

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.waypoints_publisher = self.create_publisher(Marker, '/rviz_waypoints', qos)

        self.goal_publisher = self.create_publisher(Marker, '/rviz_goal', 10)
        self.target_publisher = self.create_publisher(Marker, '/rviz_target', 10)
        self.trajectory_publisher = self.create_publisher(Marker, '/rviz_trajectory', 10)
        self.tree_publisher = self.create_publisher(Marker, '/rviz_tree', 10)
        self.path_publisher = self.create_publisher(Marker, '/rviz_path', 10)

        # occupancy grid
        self.RRT = RRTree()
        self.occupancy_grid = np.zeros((self.grid_size, self.grid_size))
        self.lidar_timestamp = self.get_clock().now()
        self.pose_timestamp = self.get_clock().now()
        self.euler_z = 0.0
        self.pos = np.array([0.0, 0.0])
        self.waypoints = np.loadtxt("/sim_ws/src/pure_pursuit/waypoints/hallway_waypoints.csv", delimiter=',')[:,:2]
        self.visualize_waypoints()

    def initialize_parameters(self):
        # grid size
        self.declare_parameter('grid_size', 50)
        self.grid_size = self.get_parameter('grid_size').value

        # grid resolution, in meters
        self.declare_parameter('grid_resolution', 0.1)
        self.grid_resolution = self.get_parameter('grid_resolution').value

        # coordinate of /ego_racecar/odom in the occupancy grid
        self.declare_parameter('origin_x', 0)
        self.origin_x = self.get_parameter('origin_x').value
        self.declare_parameter('origin_y', self.grid_size // 2)
        self.origin_y = self.get_parameter('origin_y').value

        # position of lidar in the car frame, in meters
        self.declare_parameter('lidar_offset', 0.275)
        self.lidar_offset = self.get_parameter('lidar_offset').value

        # bubble radius, in meters
        self.declare_parameter('bubble_radius', 0.1016)
        self.bubble_radius = self.get_parameter('bubble_radius').value

        # maximum number of iterations for RRT
        self.declare_parameter('max_iterations', 1000)
        self.max_iterations = self.get_parameter('max_iterations').value

        # goal threshold, in meters
        self.declare_parameter('goal_threshold', 0.05)
        self.goal_threshold = self.get_parameter('goal_threshold').value

        # step size for steering, in meters
        self.declare_parameter('step_size', 0.1)
        self.step_size = self.get_parameter('step_size').value

        # look ahead distance for goal selection, in meters
        self.declare_parameter('l_goal', 3.0)
        self.l_goal = self.get_parameter('l_goal').value

        # look ahead distance for pure pursuit, in meters
        self.declare_parameter('l_pure_pursuit', 1.0)
        self.l_pure_pursuit = self.get_parameter('l_pure_pursuit').value

        # velocity for pure pursuit, in m/s
        self.declare_parameter('v', 1.0)
        self.v = self.get_parameter('v').value

        # visualization
        self.declare_parameter('vis', True)
        self.vis = self.get_parameter('vis').value


    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here

        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:

        """
        self.lidar_timestamp = scan_msg.header.stamp
        lidar_ranges = np.array(scan_msg.ranges)
        num_ranges = len(lidar_ranges)

        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment
        angles = np.arange(0, num_ranges) * angle_increment + angle_min

        # compute the index of the endpoint cells
        endpoint_x = lidar_ranges * np.cos(angles) + self.lidar_offset
        endpoint_y = lidar_ranges * np.sin(angles)

        endpoint_x_idx = endpoint_x // self.grid_resolution + self.origin_x
        endpoint_y_idx = endpoint_y // self.grid_resolution + self.origin_y
        in_grid = (
            (endpoint_x_idx >= 0) & (endpoint_x_idx < self.grid_size) &
            (endpoint_y_idx >= 0) & (endpoint_y_idx < self.grid_size)
        )

        # update occupancy grid
        self.occupancy_grid = np.zeros((self.grid_size, self.grid_size))
        self.occupancy_grid[endpoint_x_idx[in_grid].astype(int), endpoint_y_idx[in_grid].astype(int)] = 1

        # extend obstacles by bubble radius
        # TODO: use a circle bubble instead of a square bubble
        bubble_cells = int(self.bubble_radius / self.grid_resolution)
        self.occupancy_grid = binary_dilation(self.occupancy_grid, structure=np.ones((2*bubble_cells+1, 2*bubble_cells+1)))

        if self.vis:
            self.visualize_occupancy_grid()


    def pose_callback(self, pose_msg):
        """
        The pose callback when subscribed to particle filter's inferred pose
        Here is where the main RRT loop happens

        Args: 
            pose_msg (PoseStamped): incoming message from subscribed topic
        Returns:

        """

        self.process_pose_msg(pose_msg)

        # find the goal point
        (goal_x, goal_y), _ = self.find_target_waypoint(self.waypoints, self.pos, self.l_goal)

        # transform goal point to vehicle frame
        goal_x, goal_y = self.map_to_base_link(goal_x, goal_y)
        if self.vis:
            self.visualize_point((goal_x, goal_y), self.goal_publisher, frame='/ego_racecar/base_link', ns='goal', color=(0.0, 0.0, 1.0, 1.0))

        # RRT
        self.RRT = RRTree() # reset RRT
        for i in range(self.max_iterations):
            sampled_point = self.sample()
            nearest_node_idx = self.nearest(self.RRT, sampled_point)
            new_node_pos = self.steer(nearest_node_idx, sampled_point)

            if not self.check_collision(nearest_node_idx, new_node_pos):
                self.RRT.pos = np.vstack((self.RRT.pos, new_node_pos))
                self.RRT.parent = np.append(self.RRT.parent, nearest_node_idx)

                if self.is_goal(new_node_pos, goal_x, goal_y):
                    path = self.find_path(self.RRT, len(self.RRT.pos)-1)

                    if self.vis:
                        self.visualize_tree(self.RRT.pos, self.RRT.parent)
                        self.visualize_path(path)

                    self.pure_pursuit(path)

    def sample(self):
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            (x, y) (np.ndarray): a tuple representing the sampled point

        """
        # TODO: improve sampling method
        x_lower_bound = -self.origin_x * self.grid_resolution
        x_upper_bound = (self.grid_size - self.origin_x) * self.grid_resolution
        y_lower_bound = -self.origin_y * self.grid_resolution
        y_upper_bound = (self.grid_size - self.origin_y) * self.grid_resolution
        x = np.random.uniform(x_lower_bound, x_upper_bound)
        y = np.random.uniform(y_lower_bound, y_upper_bound)
        return np.array([x, y])

    def nearest(self, tree, sampled_point):
        """
        This method should return the nearest node on the tree to the sampled point

        Args:
            tree ([]): the current RRT tree
            sampled_point (tuple of (float, float)): point sampled in free space
        Returns:
            nearest_node_idx (int): index of neareset node on the tree
        """
        node_pos = tree.pos
        distances = LA.norm(node_pos - sampled_point, axis=1)   # shape (num_nodes,)
        nearest_node_idx = np.argmin(distances)
        return nearest_node_idx

    def steer(self, nearest_node_idx, sampled_point):
        """
        This method should return a point in the viable set such that it is closer 
        to the nearest_node than sampled_point is.

        Args:
            nearest_node_idx (int): index of the nearest node on the tree to the sampled point
            sampled_point (np.ndarray): sampled point
        Returns:
            new_node_pos (np.ndarray): position of the new node created from steering
        """
        nearest_node_pos = self.RRT.pos[nearest_node_idx]
        direction = sampled_point - nearest_node_pos
        distance = LA.norm(direction)
        if distance > self.step_size:
            direction = direction / distance * self.step_size
        new_node_pos = nearest_node_pos + direction
        return new_node_pos

    def check_collision(self, nearest_node_idx, new_node_pos):
        """
        This method should return whether the path between nearest and new_node is
        collision free.

        Args:
            nearest_node_idx (int): index of the nearest node on the tree
            new_node_pos (np.ndarray): position of the new node
        Returns:
            collision (bool): whether the path between the two nodes are in collision
                              with the occupancy grid
        """
        # TODO: improve collision checking by accounting for car dynamics
        nearest_node_pos = self.RRT.pos[nearest_node_idx]
        num_points = max(int(LA.norm(new_node_pos - nearest_node_pos) / self.grid_resolution), 1)
        t = np.linspace(0, 1, num_points)
        points_x = nearest_node_pos[0] + t * (new_node_pos[0] - nearest_node_pos[0])
        points_y = nearest_node_pos[1] + t * (new_node_pos[1] - nearest_node_pos[1])
        points_x_idx = (points_x // self.grid_resolution + self.origin_x).astype(int)
        points_y_idx = (points_y // self.grid_resolution + self.origin_y).astype(int)
        collision = np.any(self.occupancy_grid[points_x_idx, points_y_idx])
        return collision

    def is_goal(self, new_node_pos, goal_x, goal_y):
        """
        This method should return whether the latest added node is close enough
        to the goal.

        Args:
            new_node_pos (np.ndarray): position of the new node
            goal_x (double): x coordinate of the current goal
            goal_y (double): y coordinate of the current goal
        Returns:
            close_enough (bool): true if node is close enough to the goal
        """
        goal_pos = np.array([goal_x, goal_y])
        if LA.norm(new_node_pos - goal_pos) < self.goal_threshold:
            return True
        else:
            return False

    def find_path(self, tree, latest_added_node_idx):
        """
        This method returns a path as a list of Nodes connecting the starting point to
        the goal once the latest added node is close enough to the goal

        Args:
            tree ([]): current tree as a list of Nodes
            latest_added_node_idx (int): index of the latest added node in the tree
        Returns:
            path ([]): valid path as a list of Nodes
        """
        path = [latest_added_node_idx]
        while tree.parent[path[-1]] != -1:
            path.append(tree.parent[path[-1]])
        return self.RRT.pos[path[::-1]] # shape (num_nodes_in_path, 2)
    
    def pure_pursuit(self, path):
        """
        Args:
            - path (np.ndarray): shape (num_nodes_in_path, 2)
        """
        (target_x, target_y), l = self.find_target_waypoint(path, np.array([0.0, 0.0]), self.l_pure_pursuit)
        if self.vis:
            self.visualize_point((target_x, target_y), self.target_publisher, frame='/ego_racecar/base_link', ns='pure_pursuit_target', color=(0.0, 1.0, 0.0, 1.0))

        gamma = 2 * target_y / l ** 2
        # angle = np.clip(gamma, -0.7854, 0.7854)
        angle = gamma

        # publish drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = self.v
        drive_msg.drive.steering_angle = angle
        self.drive_pub_.publish(drive_msg)

        if self.vis:
            self.visualize_trajectory(angle)

    def process_pose_msg(self, pose_msg):
        self.pose_timestamp = pose_msg.header.stamp
        quaternion = np.array([pose_msg.pose.pose.orientation.x, 
                            pose_msg.pose.pose.orientation.y, 
                            pose_msg.pose.pose.orientation.z, 
                            pose_msg.pose.pose.orientation.w])
        euler = euler_from_quaternion(quaternion)
        self.euler_z = euler[2]
        self.pos = np.array([pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y])
        # NOTE: If using pf/pose/odom, position is the laser position, not the base_link position. You need to transform it to the base_link position by adding the laser offset (0.275m) in the x direction of the car frame.

    def find_target_waypoint(self, waypoints, current_pos, l):
        dist = np.linalg.norm(waypoints - current_pos, axis=1)

        smaller = (dist <= l)
        larger = (dist > l)
        targets = np.bitwise_and(smaller[:-1], larger[1:])
        target_idx = targets.nonzero()[0][0]

        target_waypoint = waypoints[target_idx+1]

        if self.vis:
            self.visualize_point(target_waypoint)

        return (target_waypoint[0], target_waypoint[1]), dist[target_idx+1]
    
    def map_to_base_link(self, x, y):
        base_link_x = (x - self.pos[0]) * np.cos(self.euler_z) + (y - self.pos[1]) * np.sin(self.euler_z)
        base_link_y = (y - self.pos[1]) * np.cos(self.euler_z) - (x - self.pos[0]) * np.sin(self.euler_z)
        return base_link_x, base_link_y

    # The following methods are needed for RRT* and not RRT
    def cost(self, tree, node):
        """
        This method should return the cost of a node

        Args:
            node (Node): the current node the cost is calculated for
        Returns:
            cost (float): the cost value of the node
        """
        return 0

    def line_cost(self, n1, n2):
        """
        This method should return the cost of the straight line between n1 and n2

        Args:
            n1 (Node): node at one end of the straight line
            n2 (Node): node at the other end of the straint line
        Returns:
            cost (float): the cost value of the line
        """
        return 0

    def near(self, tree, node):
        """
        This method should return the neighborhood of nodes around the given node

        Args:
            tree ([]): current tree as a list of Nodes
            node (Node): current node we're finding neighbors for
        Returns:
            neighborhood ([]): neighborhood of nodes as a list of Nodes
        """
        neighborhood = []
        return neighborhood
    
    def visualize_occupancy_grid(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.lidar_timestamp
        msg.header.frame_id = '/ego_racecar/base_link'
        msg.info.resolution = self.grid_resolution
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size
        msg.info.origin.position.x = -self.origin_x * self.grid_resolution
        msg.info.origin.position.y = -self.origin_y * self.grid_resolution
        msg.info.origin.position.z = 0.0

        msg.data = (self.occupancy_grid.T.flatten() * 100).astype(int).tolist()  # flatten and convert to percentage
        self.occupancy_grid_pub_.publish(msg)

    def visualize_waypoints(self):
        points = Marker()
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

    def visualize_point(self, point, point_publisher=None, frame='/map', ns='point', color=(0.0, 1.0, 0.0, 1.0)):
        sphere = Marker()
        sphere.header.stamp = self.pose_timestamp
        sphere.header.frame_id = frame
        sphere.ns = ns
        sphere.id = 0
        sphere.type = Marker.SPHERE

        sphere.pose.position.x = point[0]
        sphere.pose.position.y = point[1]
        sphere.pose.position.z = 0.0

        sphere.scale.x = 0.2
        sphere.scale.y = 0.2
        sphere.scale.z = 0.2

        sphere.color.r = color[0]
        sphere.color.g = color[1]
        sphere.color.b = color[2]
        sphere.color.a = color[3]

        if point_publisher:
            point_publisher.publish(sphere)

    def visualize_trajectory(self, angle, steps=20, arc_length=2.0):
        line = Marker()
        line.header.stamp = self.pose_timestamp
        line.header.frame_id = 'ego_racecar/base_link'
        line.ns = 'trajectory'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.05

        line.color.r = 0.0
        line.color.g = 0.0
        line.color.b = 1.0
        line.color.a = 1.0

        if abs(angle) < 1e-3:
            # straight line along +x
            for i in range(steps):
                p = Point(x=arc_length * i / (steps - 1), y=0.0, z=0.0)
                line.points.append(p)
        else:
            R = 1.0 / angle
            total_angle = arc_length / R
            for i in range(steps):
                a = -np.pi / 2 + total_angle * i / (steps - 1)
                p = Point(x=R * np.cos(a), y=R + R * np.sin(a), z=0.0)
                line.points.append(p)
                line.points.append(p)

        self.trajectory_publisher.publish(line)

    def visualize_tree(self, pos, parent):
        """
        Visualize the RRT tree as a LINE_LIST marker in the base_link frame.
        Each edge is represented as a (parent, child) point pair.

        Args:
            pos (np.ndarray): shape (num_nodes, 2), node positions in base_link frame
            parent (np.ndarray): shape (num_nodes,), parent index for each node (-1 for root)
        """
        marker = Marker()
        marker.header.stamp = self.pose_timestamp
        marker.header.frame_id = 'ego_racecar/base_link'
        marker.ns = 'rrt_tree'
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.02

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for i in range(1, len(pos)):
            p_idx = parent[i]
            p_parent = Point(x=pos[p_idx][0], y=pos[p_idx][1], z=0.0)
            p_child = Point(x=pos[i][0], y=pos[i][1], z=0.0)
            marker.points.append(p_parent)
            marker.points.append(p_child)

        self.tree_publisher.publish(marker)

    def visualize_path(self, path):
        """
        Visualize the found RRT path as a LINE_STRIP marker in the base_link frame.

        Args:
            path (np.ndarray): shape (num_nodes_in_path, 2), path node positions in base_link frame
        """
        marker = Marker()
        marker.header.stamp = self.pose_timestamp
        marker.header.frame_id = 'ego_racecar/base_link'
        marker.ns = 'rrt_path'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for node in path:
            marker.points.append(Point(x=node[0], y=node[1], z=0.0))

        self.path_publisher.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
