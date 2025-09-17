import os

import numpy as np
import yaml
from ament_index_python import get_package_share_directory
from crazyflie_interfaces.msg._reference_trajectory import ReferenceTrajectory
import rclpy, heapq, random
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped
from crazyflie_interfaces.msg import LogDataGeneric, AttitudeSetpoint

from std_msgs.msg import Empty
import pandas as pd


class DataLogger(Node):
    def __init__(self) :
        super().__init__('data_logger')

        prefix = 'cf_1'

        self.position = [0.0, 0.0, 0.0]
        self.attitude = [1.0, 0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]
        self.angular_velocity = [0.0, 0.0, 0.0]
        self.ref_trajectory = np.zeros(13)

        self.position_history = []
        self.attitude_history = []
        self.velocity_history = []
        self.angular_velocity_history = []
        self.ref_trajectory_history = []
        self.time = []
        self.mpc_trajectory = []

        self.was_mpc_takeoff = False
        self.takeoff_time = None
        self.was_mpc_trajectory = False



        self.create_subscription(PoseStamped, f'{prefix}/pose', self._pose_msg_callback, 10)
        self.create_subscription(LogDataGeneric, f'{prefix}/velocity', self._velocity_msg_callback, 10)
        self.create_subscription(LogDataGeneric, f'{prefix}/angular_velocity', self._angular_velocity_msg_callback, 10)
        self.create_subscription(ReferenceTrajectory,  f'{prefix}/ref_trajectory',  self._ref_trajectory_callback, 10)

        self.takeoffService = self.create_subscription(Empty, f'/all/mpc_takeoff', self.takeoff, 10)
        self.trajectoryService = self.create_subscription(Empty, f'/all/mpc_trajectory', self.start_trajectory, 10)
        self.saveService = self.create_subscription(Empty, f'/logger/save_file', self.save_csv, 10)

        self.timer = self.create_timer(0.02, self.main_loop)




    def _pose_msg_callback(self, msg: PoseStamped):
        self.position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.attitude = [msg.pose.orientation.w,msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z]



    def _velocity_msg_callback(self, msg: LogDataGeneric):
        self.velocity = [v / 1000.0 for v in msg.values]

    def _angular_velocity_msg_callback(self, msg: LogDataGeneric):
        self.angular_velocity = [v / 1000.0 for v in msg.values]

    def _ref_trajectory_callback(self, msg: ReferenceTrajectory):
        if msg.points:
            traj_point = msg.points[0]
            self.ref_trajectory = [
                *traj_point.position,
                *traj_point.orientation,
                *traj_point.linear_velocity,
                *traj_point.angular_velocity
            ]
        else:
            self.ref_trajectory = np.zeros(13)

    def takeoff(self, msg: Empty):
        self.was_mpc_takeoff = True
        self.takeoff_time = self.get_clock().now()

    def start_trajectory(self, msg: Empty):
        self.was_mpc_trajectory = True

    def save_csv(self, msg: Empty=None):
        if not self.was_mpc_takeoff:
            return
        
        df = pd.DataFrame({
            'time': self.time,
            'position': self.position_history,
            'attitude': self.attitude_history,
            'velocity': self.velocity_history,
            'angular_velocity': self.angular_velocity_history,
            'ref_trajectory': self.ref_trajectory_history,
            'mpc_trajectory': self.mpc_trajectory
        })

        if not os.path.exists('logs'):
            os.makedirs('logs')
        df.to_csv('logs/logged_data.csv', index=False)
        self.get_logger().info('Zapisano dane do logs/logged_data.csv')


    def main_loop(self):

        if not self.was_mpc_takeoff:
            return
        
        self.position_history.append(self.position)
        self.attitude_history.append(self.attitude)
        self.velocity_history.append(self.velocity)
        self.angular_velocity_history.append(self.angular_velocity)
        self.ref_trajectory_history.append(self.ref_trajectory)
        self.time.append((self.get_clock().now() - self.takeoff_time).nanoseconds / 1e9)

        if self.was_mpc_trajectory:
            self.mpc_trajectory.append(1)
        else:
            self.mpc_trajectory.append(0)








def main():
    crazyflie_mpc_config_yaml = os.path.join(
        get_package_share_directory('crazyflie_mpc'),
        'config',
        'mpc.yaml')
    
    with open(crazyflie_mpc_config_yaml, 'r') as file:
        crazyflie_mpc_config = yaml.safe_load(file)


    rclpy.init()
    node = DataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_csv()
    finally:
        node.destroy_node()
        rclpy.shutdown()
