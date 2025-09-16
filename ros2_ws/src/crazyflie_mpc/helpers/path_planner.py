import os

import yaml
from ament_index_python import get_package_share_directory
import rclpy, heapq, random
from rclpy.node import Node
from rclpy.time import Time
import numpy as np

from geometry_msgs.msg import PoseStamped
from crazyflie_interfaces.msg import LogDataGeneric, AttitudeSetpoint, TrajectoryPoint, ReferenceTrajectory
from std_msgs.msg import Empty, Float32
from nav_msgs.msg import Odometry, Path


class PathPlanner(Node):
    def __init__(self, cf_name: str, mpc_N: int, mpc_tf: float,  rate: int, trajectory_type: str='lemniscate') :
        super().__init__('path_planner')

        prefix = cf_name

        self.rate = rate
        self.mpc_N = mpc_N
        self.mpc_tf = mpc_tf

        self.position = [0.,0.,0.]
        self.attitude = [1.,0.,0.,0.]

        self.trajectory_type =  trajectory_type
        self.flight_mode = 'idle'
        self.trajectory_changed = True
        self.trajectory_t0 = self.get_clock().now()
        self.takeoff_duration = 5.0
        self.land_duration = 5.0
        self.is_flying = False



        self.create_subscription( PoseStamped, f'{prefix}/pose_d',self._pose_msg_callback,10)
        self.takeoffService = self.create_subscription(Empty, f'/all/mpc_takeoff', self.takeoff, 10)
        self.landService = self.create_subscription(Empty, f'/all/mpc_land', self.land, 10)
        self.trajectoryService = self.create_subscription(Empty, f'/all/mpc_trajectory', self.start_trajectory, 10)
        self.hoverService = self.create_subscription(Empty, f'/all/mpc_hover', self.hover, 10)

        self.mpc_solution_path_pub = self.create_publisher(
            ReferenceTrajectory,
            f'{prefix}/ref_trajectory',
            10)
        
        self.create_timer(1./rate, self._main_loop)

        
    def _pose_msg_callback(self, msg: PoseStamped):
        self.position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.attitude = [msg.pose.orientation.w,msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z]

    def start_trajectory(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'trajectory'

    def takeoff(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'takeoff'
        self.go_to_position = np.array([self.position[0],
                                        self.position[1],
                                        1.0])
        
    def hover(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'hover'
        self.go_to_position = np.array([self.position[0],
                                        self.position[1],
                                        self.position[2]])

    def land(self, msg):
        self.trajectory_changed = True
        self.flight_mode = 'land'
        self.go_to_position = np.array([self.position[0],
                                        self.position[1],
                                        0.1])

    def trajectory_function(self, t):
        if self.trajectory_type == 'horizontal_circle':      
            a = 1.0
            omega = 0.75*np.tanh(0.1*t)
            pxr = self.trajectory_start_position[0] + a*np.cos(omega*t) - a
            pyr = self.trajectory_start_position[1] + a*np.sin(omega*t)
            pzr = self.trajectory_start_position[2]
            vxr = -a*omega*np.sin(omega*t)
            vyr = a*omega*np.cos(omega*t)
            vzr = 0.0
        elif self.trajectory_type == 'vertical_circle':
            a = 1.0
            omega = 0.75*np.tanh(0.1*t)
            pxr = self.trajectory_start_position[0] + a*np.sin(-omega*t + np.pi)
            pyr = self.trajectory_start_position[1]
            pzr = self.trajectory_start_position[2] + a*np.cos(-omega*t + np.pi) + a
            vxr = -a*omega*np.cos(-omega*t + np.pi)
            vyr = 0.0
            vzr = a*omega*np.sin(-omega*t + np.pi)
        elif self.trajectory_type == 'tilted_circle':
            a = 0.5
            c = 0.3
            omega = 0.75*np.tanh(0.1*t)
            pxr = self.trajectory_start_position[0] + a*np.cos(omega*t) - a
            pyr = self.trajectory_start_position[1] + a*np.sin(omega*t)
            pzr = self.trajectory_start_position[2] + c*np.sin(omega*t)
            vxr = -a*omega*np.sin(omega*t)
            vyr = a*omega*np.cos(omega*t)
            vzr = c*omega*np.cos(omega*t)
        elif self.trajectory_type == 'lemniscate':
            a = 1.0
            b = 0.8*np.tanh(0.1*t)
            pxr = self.trajectory_start_position[0] + a*np.sin(b*t)
            pyr = self.trajectory_start_position[1] + a*np.sin(b*t)*np.cos(b*t)
            pzr = self.trajectory_start_position[2]
            vxr = a*b*np.cos(b*t)
            vyr = a*b*np.cos(2*b*t)
            vzr = 0.0
        elif self.trajectory_type == 'helix':
            a = 1.0
            T_end = 10.0
            helix_velocity = 0.2
            omega = 0.75*np.tanh(0.1*t)
            pxr = self.trajectory_start_position[0] + a*np.cos(omega*t) - a
            pyr = self.trajectory_start_position[1] + a*np.sin(omega*t)
            vxr = -a*omega*np.sin(omega*t)
            vyr = a*omega*np.cos(omega*t)
            if t < T_end:
                pzr = self.trajectory_start_position[2] + helix_velocity*t
                vzr = helix_velocity
            else:
                pzr = self.trajectory_start_position[2] + helix_velocity*T_end
                vzr = 0.0
        elif self.trajectory_type == 'step':
            pxr = 0.2
            pyr = 0.0
            pzr = self.trajectory_start_position[2]
            vxr = 0.0
            vyr = 0.0
            vzr = 0.0

        return np.array([pxr,pyr,pzr,1.,0.,0.,0.,vxr,vyr,vzr,0.,0.,0.])

    def navigator(self, t):
        if self.flight_mode == 'takeoff':
            t_mpc_array = np.linspace(t, self.mpc_tf + t, self.mpc_N+1)
            yref = np.array([np.array([*((self.go_to_position - self.trajectory_start_position)*(1./(1. + np.exp(-(12.0 * (t_mpc - self.takeoff_duration) / self.takeoff_duration + 6.0)))) + self.trajectory_start_position),1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]) for t_mpc in t_mpc_array]).T
            # yref = np.repeat(np.array([[*self.go_to_position,0,0,0]]).T, self.mpc_N, axis=1)
        elif self.flight_mode == 'land':
            t_mpc_array = np.linspace(t, self.mpc_tf + t, self.mpc_N+1)
            yref = np.array([np.array([*((self.go_to_position - self.trajectory_start_position)*(1./(1. + np.exp(-(12.0 * (t_mpc - self.land_duration) / self.land_duration + 6.0)))) + self.trajectory_start_position),0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]) for t_mpc in t_mpc_array]).T
            # yref = np.repeat(np.array([[*self.go_to_position,0,0,0]]).T, self.mpc_N, axis=1)
        elif self.flight_mode == 'trajectory':
            t_mpc_array = np.linspace(t, self.mpc_tf + t, self.mpc_N+1)
            yref = np.array([self.trajectory_function(t_mpc) for t_mpc in t_mpc_array]).T

            # yref = np.array([np.array([0.2,0.,self.trajectory_start_position[2],1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]) for _ in range(self.mpc_N+1)]).T


        elif self.flight_mode == 'hover':
            yref = np.repeat(np.array([[*self.go_to_position,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]).T, self.mpc_N, axis=1)
        return yref
    
    def _main_loop(self):
        if self.flight_mode == 'idle':
            return
        
        if self.trajectory_changed:
            self.trajectory_start_position = self.position
            self.trajectory_t0 = self.get_clock().now()
            self.trajectory_changed = False

        ref_trajectory = ReferenceTrajectory()
        ref_trajectory.header.stamp = self.get_clock().now().to_msg()
        ref_trajectory.header.frame_id = 'world'

        t = (self.get_clock().now() - self.trajectory_t0).nanoseconds / 10.0**9
        trajectory = self.navigator(t)


        for i in range(self.mpc_N+1):
            traj_point = TrajectoryPoint()
            traj_point.position = trajectory[0:3,i].tolist()
            traj_point.orientation = trajectory[3:7,i].tolist()
            traj_point.linear_velocity = trajectory[7:10,i].tolist()
            traj_point.angular_velocity = trajectory[10:13,i].tolist()
            ref_trajectory.points.append(traj_point) # type: ignore

        self.mpc_solution_path_pub.publish(ref_trajectory)

        
        



def main():
    crazyflie_mpc_config_yaml = os.path.join(
        get_package_share_directory('crazyflie_mpc'),
        'config',
        'mpc.yaml')
    
    with open(crazyflie_mpc_config_yaml, 'r') as file:
        crazyflie_mpc_config = yaml.safe_load(file)

    n_agents = crazyflie_mpc_config['n_agents']

    mpc_tf = crazyflie_mpc_config['mpc']['horizon']
    mpc_N = crazyflie_mpc_config['mpc']['num_steps']
    control_update_rate = crazyflie_mpc_config['mpc']['control_update_rate']
    trajectory_type = crazyflie_mpc_config['path_planner']['trajectory_type']

    rclpy.init()


    node = PathPlanner('cf_1', mpc_N, mpc_tf, control_update_rate, trajectory_type)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
