#!/usr/bin/env python3
"""
This file contains the class definition for tree nodes and RRT
Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf

TODO: 
- Implement a probabilistic occupancy grid.
    - Shift the occupancy grid according to odom.
    - Update the occupancy grid according to lidar scans.
- Implement RRT rebuild gate.
- Implement RRT*.
- Implement pure pursuit with odom.
- Replace particle filter with faster SLAM methods.
"""
import numpy as np
from numpy import linalg as LA
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import OccupancyGrid

from scipy.ndimage import binary_dilation
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf_transformations import euler_from_quaternion

from lab6_pkg.helper import find_target_waypoint, visualize_point, visualize_points, visualize_trajectory, visualize_tree, visualize_path, visualize_occupancy_grid
from collections import deque


class RRTree(object):
    def __init__(self):
        self.pos = np.array([[0.0, 0.0]]) # shape (num_nodes, 2)
        self.parent = np.array([-1])
        self.cost = np.array([0.0]) # only used in RRT*


# class def for RRT
class RRT(Node):
    def __init__(self):
        super().__init__('rrt_node')

        self.initialize_parameters()

        self.pose_sub_ = self.create_subscription(Odometry, '/ego_racecar/odom', self.pose_callback, 1)
        self.scan_sub_ = self.create_subscription(LaserScan, '/scan', self.scan_callback, 1)

        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.occupancy_grid_publisher = self.create_publisher(OccupancyGrid, '/rviz_occupancy_grid', 10)

        if self.vis:
            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.waypoints_publisher = self.create_publisher(Marker, '/rviz_waypoints', qos)

            self.marker_array = MarkerArray()
            self.marker_array_publisher = self.create_publisher(MarkerArray, '/rviz_markers', 10)

        # occupancy grid
        self.RRT = RRTree()
        self.occupancy_grid = np.zeros((self.grid_size, self.grid_size))
        self.lidar_timestamp = self.get_clock().now().to_msg()
        self.pose_timestamp = self.get_clock().now().to_msg()
        self.euler_z = 0.0
        self.pos = np.array([0.0, 0.0])
        self.waypoints = np.loadtxt("/sim_ws/src/pure_pursuit/waypoints/hallway_waypoints.csv", delimiter=',')[:,:2]
        self.goal_pos = np.array([0.0, 0.0])
        
        if self.vis:
            waypoints_marker = visualize_points(self.waypoints, self.pose_timestamp, frame_id='/map', ns='waypoints', id=0, color=(1.0, 0.0, 0.0, 1.0))
            self.waypoints_publisher.publish(waypoints_marker)

    def initialize_parameters(self):
        # grid size
        self.declare_parameter('grid_size', 80)
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
        self.declare_parameter('step_size', 0.25)
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

        # probability of sampling the goal point in the sampling function
        self.declare_parameter('goal_sample_rate', 0.1)
        self.goal_sample_rate = self.get_parameter('goal_sample_rate').value

        # gamma for RRT*
        self.declare_parameter('gamma', 10)
        self.gamma = self.get_parameter('gamma').value

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
            occupancy_grid_msg = visualize_occupancy_grid(self.occupancy_grid, self.lidar_timestamp, self.grid_resolution, self.grid_size, self.grid_size, self.origin_x, self.origin_y)
            self.occupancy_grid_publisher.publish(occupancy_grid_msg)

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
        goal, _ = find_target_waypoint(self.waypoints, self.pos, self.l_goal)

        # transform goal point to vehicle frame
        goal_pos = self.map_to_base_link(goal)
        self.goal_pos = goal_pos
        if self.vis:
            self.marker_array = MarkerArray()
            goal_marker = visualize_point(goal_pos, self.pose_timestamp, frame_id='/ego_racecar/base_link', ns='goal', color=(0.0, 0.0, 1.0, 1.0))
            self.marker_array.markers.append(goal_marker)

        # RRT
        self.RRT = RRTree() # reset RRT
        for i in range(self.max_iterations):
            sampled_point = self.sample()
            nearest_node_idx = self.nearest(self.RRT, sampled_point)
            nearest_node = self.RRT.pos[nearest_node_idx]
            new_node = self.steer(nearest_node, sampled_point)

            if not self.check_collision(nearest_node, new_node):
                neighborhood_idx = self.near(self.RRT, new_node)

                self.connect(self.RRT, new_node, nearest_node_idx, neighborhood_idx)
                
                self.rewire(self.RRT, new_node, neighborhood_idx)

                if self.is_goal(new_node, goal_pos):
                    path = self.find_path(self.RRT, len(self.RRT.pos)-1)

                    if self.vis:
                        tree_marker = visualize_tree(self.RRT.pos, self.RRT.parent, self.pose_timestamp, frame_id='/ego_racecar/base_link', ns='tree', id=0, color=(1.0, 0.0, 0.0, 1.0))
                        path_marker = visualize_path(path, self.pose_timestamp, frame_id='/ego_racecar/base_link', ns='path', id=0, color=(0.0, 1.0, 1.0, 1.0))
                        self.marker_array.markers.append(tree_marker)
                        self.marker_array.markers.append(path_marker)

                    self.pure_pursuit(path)

                    if self.vis:
                        self.marker_array_publisher.publish(self.marker_array)
                    
                    return
                
        self.get_logger().info("Path not found!")

    def sample(self):
        """
        This method should randomly sample the free space, and returns a viable point

        Args:
        Returns:
            (x, y) (np.ndarray): a tuple representing the sampled point

        """
        # TODO: improve sampling method
        if np.random.rand() < self.goal_sample_rate: # with 10% probability, sample the goal point to encourage goal bias
            return self.goal_pos
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

    def steer(self, nearest_node, sampled_point):
        """
        This method should return a point in the viable set such that it is closer 
        to the nearest_node than sampled_point is.

        Args:
            nearest_node (np.ndarray): position of the nearest node on the tree to the sampled point
            sampled_point (np.ndarray): position of the sampled point
        Returns:
            new_node (np.ndarray): position of the new node created from steering
        """
        direction = sampled_point - nearest_node
        distance = LA.norm(direction)
        if distance > self.step_size:
            direction = direction / distance * self.step_size
        new_node = nearest_node + direction
        return new_node

    def check_collision(self, nearest_node, new_node):
        """
        This method should return whether the path between nearest and new_node is
        collision free.

        Args:
            nearest_node (np.ndarray): position of the nearest node on the tree
            new_node (np.ndarray): position of the new node
        Returns:
            collision (bool): whether the path between the two nodes are in collision
                              with the occupancy grid
        """
        # TODO: improve collision checking by accounting for car dynamics
        num_points = max(int(LA.norm(new_node - nearest_node) / self.grid_resolution), 1)
        t = np.linspace(0, 1, num_points)
        points_x = nearest_node[0] + t * (new_node[0] - nearest_node[0])
        points_y = nearest_node[1] + t * (new_node[1] - nearest_node[1])
        points_x_idx = (points_x // self.grid_resolution + self.origin_x).astype(int)
        points_y_idx = (points_y // self.grid_resolution + self.origin_y).astype(int)
        collision = np.any(self.occupancy_grid[points_x_idx, points_y_idx])
        return collision

    def is_goal(self, new_node, goal):
        """
        This method should return whether the latest added node is close enough
        to the goal.

        Args:
            new_node (np.ndarray): position of the new node
            goal (np.ndarray): position of the goal
        Returns:
            close_enough (bool): true if node is close enough to the goal
        """
        if LA.norm(new_node - goal) < self.goal_threshold:
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
        target, l = find_target_waypoint(path, np.array([0.0, 0.0]), self.l_pure_pursuit)
        if self.vis:
            target_marker = visualize_point(target, self.pose_timestamp, frame_id='/ego_racecar/base_link', ns='target', color=(0.0, 1.0, 0.0, 1.0))
            self.marker_array.markers.append(target_marker)

        target_y = target[1]
        gamma = 2 * target_y / l ** 2
        angle = np.clip(gamma, -0.7854, 0.7854)

        # publish drive message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = self.v
        drive_msg.drive.steering_angle = angle
        self.drive_publisher.publish(drive_msg)

        if self.vis:
            trajectory_marker = visualize_trajectory(angle, self.pose_timestamp, frame_id='/ego_racecar/base_link', ns='trajectory', id=0, color=(0.0, 1.0, 0.0, 1.0), steps=20, arc_length=self.l_pure_pursuit)
            self.marker_array.markers.append(trajectory_marker)

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
    
    def map_to_base_link(self, map_pos):
        base_link_x = (map_pos[0] - self.pos[0]) * np.cos(self.euler_z) + (map_pos[1] - self.pos[1]) * np.sin(self.euler_z)
        base_link_y = (map_pos[1] - self.pos[1]) * np.cos(self.euler_z) - (map_pos[0] - self.pos[0]) * np.sin(self.euler_z)
        base_link_pos = np.array([base_link_x, base_link_y])
        return base_link_pos

    # The following methods are needed for RRT* and not RRT
    def connect(self, tree, new_node, nearest_node_idx, neighborhood_idx):
        """
        Connect the new node to the tree.
        """
        node_min = nearest_node_idx
        cost_min = tree.cost[nearest_node_idx] + self.line_cost(tree.pos[nearest_node_idx], new_node)
        for i in range(len(neighborhood_idx)):
            neighbor_idx = neighborhood_idx[i]
            collision = self.check_collision(tree.pos[neighbor_idx], new_node)
            cost = tree.cost[neighbor_idx] + self.line_cost(tree.pos[neighbor_idx], new_node)
            if not collision and cost < cost_min:
                node_min = neighbor_idx
                cost_min = cost
            
        tree.pos = np.vstack((tree.pos, new_node))
        tree.parent = np.append(tree.parent, node_min)
        tree.cost = np.append(tree.cost, cost_min)

    def rewire(self, tree, new_node, neighborhood_idx):
        """
        Rewire the neighborhood after connecting the new node to the tree.
        """
        for i in range(len(neighborhood_idx)):
            neighbor_idx = neighborhood_idx[i]
            collision = self.check_collision(new_node, tree.pos[neighbor_idx])
            cost = tree.cost[-1] + self.line_cost(new_node, tree.pos[neighbor_idx])
            if not collision and cost < tree.cost[neighbor_idx]:
                tree.parent[neighbor_idx] = len(tree.pos) - 1
                tree.cost[neighbor_idx] = cost

                parents = deque()
                parents.append(neighbor_idx)
                while parents:
                    parent = parents.popleft()
                    children = (tree.parent == parent).nonzero()[0]
                    for child in children:
                        tree.cost[child] = tree.cost[parent] + self.line_cost(tree.pos[parent], tree.pos[child])
                        parents.append(child)

    def line_cost(self, n1, n2):
        """
        This method should return the cost of the straight line between n1 and n2

        Args:
            n1 (np.ndarray): position of the first node
            n2 (np.ndarray): position of the second node
        Returns:
            cost (float): the cost value of the line
        """
        cost = LA.norm(n1 - n2)
        return cost

    def near(self, tree, node):
        """
        This method should return the neighborhood of nodes around the given node

        Args:
            tree ([]): current tree as a list of Nodes
            node_pos (np.ndarray): position of the current node we're finding neighbors for
        Returns:
            neighborhood (np.ndarray): neighborhood of nodes as an array of node indices
        """
        num_nodes = len(tree.pos)
        # radius = min(self.gamma * (math.sqrt(math.log(num_nodes) / num_nodes)), self.step_size)
        radius = 0.8
        nodes = tree.pos
        dist = LA.norm(nodes - node, axis=1)
        in_neighborhood = dist < radius
        neighborhood = in_neighborhood.nonzero()[0]
        return neighborhood


def main(args=None):
    rclpy.init(args=args)
    print("RRT Initialized")
    rrt_node = RRT()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
