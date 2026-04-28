# CMU 16-663 F1TENTH Autonomous Racing

ROS 2 workspace for the CMU 16-663 F1TENTH Autonomous Racing course.

## Labs

### Lab 2 — Automatic Emergency Braking
Safety node that computes Instantaneous Time to Collision (iTTC) from LiDAR scans and automatically brakes the car before impact. Introduces ROS 2 message types (`LaserScan`, `Odometry`, `AckermannDriveStamped`).

### Lab 3 — Wall Following
PID controller that keeps the car a fixed distance from the right wall by computing distance and angle from two LiDAR beams and feeding the error into a proportional-integral-derivative loop.

### Lab 4 — Follow the Gap
Reactive obstacle avoidance using two algorithms. Follow the Gap finds the nearest obstacle, zeros out a safety bubble around it, locates the widest free gap, and steers toward its center. Disparity Extender extends the edges of depth disparities in the LiDAR scan to account for car width, then steers toward the furthest point in the largest gap.

### Lab 5 — SLAM and Pure Pursuit
End-to-end autonomous lapping: build a map with `slam_toolbox`, localize with a particle filter, record waypoints, and track them using the Pure Pursuit geometric path-tracking algorithm.

### Lab 6 — Motion Planning (RRT / RRT*)
Sampling-based local motion planning for real-time obstacle avoidance on a race track. RRT builds a tree by randomly sampling the free space and connecting samples to the nearest node. RRT* extends this with rewiring: after each new node is added, nearby nodes are reconnected through it if doing so lowers their path cost, asymptotically converging to the optimal path. Both planners operate in an occupancy grid and replan each timestep within a window around the car.

### Lab 7 — Model Predictive Control
MPC controller that linearizes and discretizes a kinematic bicycle model, then solves a convex QP over a receding horizon to follow a reference trajectory while respecting actuator limits.

### Lab 8 — Vision
Camera pipeline on Jetson: access a Realsense camera via v4l2, perform intrinsic calibration with OpenCV, estimate ground-plane distances from pixel coordinates, then train and deploy an object detector with TensorRT.

## Build

```bash
source /opt/ros/humble/setup.bash
rosdep install -i --from-path src --rosdistro humble -y
colcon build
source install/local_setup.bash
```

## Run

```bash
# Simulator + RViz (run first in all labs)
ros2 launch f1tenth_gym_ros gym_bridge_launch.py

# Lab 2 — Automatic Emergency Braking
ros2 run safety_node safety_node.py

# Lab 3 — Wall Following
ros2 run wall_follow wall_follow_node.py

# Lab 4 — Follow the Gap / Disparity Extender
ros2 run gap_follow reactive_node.py
ros2 run gap_follow reactive_node_extender.py
ros2 run gap_follow reactive_node_ttc.py

# Lab 5 — Pure Pursuit
ros2 run pure_pursuit log_waypoints.py      # record waypoints first
ros2 run pure_pursuit pure_pursuit_node.py

# Lab 6 — RRT / RRT*
ros2 run lab6_pkg rrt_node.py
ros2 run lab6_pkg rrt_star_node.py
ros2 run lab6_pkg rrt_star_gate_node.py

# Lab 7 — MPC (template, not implemented)

# Lab 8 — Vision (standalone scripts, no ROS)
python3 src/lab8_pkg/camera.py       # camera capture
python3 src/lab8_pkg/calibrate.py    # camera calibration
python3 src/lab8_pkg/distance.py     # distance estimation
python3 src/lab8_pkg/convert_trt.py  # build TensorRT engine from ONNX
python3 src/lab8_pkg/detection.py    # TensorRT object detection
python3 src/lab8_pkg/integrated.py   # full pipeline
```
