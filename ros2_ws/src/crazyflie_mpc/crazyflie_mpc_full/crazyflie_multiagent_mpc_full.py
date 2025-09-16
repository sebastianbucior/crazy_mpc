import os

import pandas as pd

from crazyflie_interfaces.msg._reference_trajectory import ReferenceTrajectory

from .quadrotor_full_model import QuadrotorFull
from .trajectory_tracking_mpc_full import TrajectoryTrackingMpc

import rclpy
import rclpy.node
from rclpy import executors
# from crazyflie_py import *

from ament_index_python.packages import get_package_share_directory
from crazyflie_interfaces.msg import LogDataGeneric, AttitudeSetpoint
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty, Float32
import tf_transformations

import pathlib
from enum import Enum
from copy import copy
from collections import deque
import numpy as np
import yaml
import time
from builtin_interfaces.msg import Time

class Motors(Enum):
    MOTOR_CLASSIC = 1 # https://store.bitcraze.io/products/4-x-7-mm-dc-motor-pack-for-crazyflie-2 w/ standard props
    MOTOR_UPGRADE = 2 # https://store.bitcraze.io/collections/bundles/products/thrust-upgrade-bundle-for-crazyflie-2-x

class CrazyflieMPC(rclpy.node.Node):
    def __init__(self, cf_name: str, mpc_solver: TrajectoryTrackingMpc, quadrotor_dynamics: QuadrotorFull, mpc_N: int, mpc_tf: float, rate: int, plot_trajectory: bool = False, use_predictor: bool = False, delay_relay: bool = False) :
        super().__init__(node_name='crazyflie_mpc', namespace=cf_name)
        prefix = '/' + cf_name
        
        self.is_connected = True

        self.rate = rate

        self.odometry = Odometry()

        self.mpc_N = mpc_N
        self.mpc_tf = mpc_tf

        self.position = []
        self.position_stamp = Time()
        self.used_position_stamp = Time()
        self.velocity = []
        self.angular_velocity = []
        self.attitude = []

        self.flight_mode = 'idle'
        
        # TODO: Switch to parameters yaml?
        self.motors = Motors.MOTOR_CLASSIC # MOTOR_CLASSIC, MOTOR_UPGRADE

        self.takeoff_duration = 5.0
        self.land_duration = 5.0

        self.g = quadrotor_dynamics.gravity
        self.m = quadrotor_dynamics.mass

        self.mpc_solver = copy(mpc_solver)
        self.plot_trajectory = plot_trajectory
        self.control_queue = None
        self.get_logger().info('Initialization completed...')

        self.is_flying = False

        self.cnt = 0
        self.ref_trajectory = np.zeros((13, self.mpc_N + 1))

        self.use_predictor = use_predictor
        self.alpha = 0.1
        self.latency_cnt = 0
        self.latency_initialized = False
        self.latency = 0.0
        self.last_rpm = np.array([0.,0.,0.,0.])


        if delay_relay:
            self.create_subscription(PoseStamped, f'{prefix}/pose_d', self._pose_msg_callback, 10)
            self.create_subscription(LogDataGeneric, f'{prefix}/velocity_d', self._velocity_msg_callback, 10)
            self.create_subscription(LogDataGeneric, f'{prefix}/angular_velocity_d', self._angular_velocity_msg_callback, 10)
            self.attitude_setpoint_pub = self.create_publisher(AttitudeSetpoint, f'{prefix}/cmd_attitude_d', 10)
        else:
            self.create_subscription(PoseStamped, f'{prefix}/pose', self._pose_msg_callback, 10)
            self.create_subscription(LogDataGeneric, f'{prefix}/velocity', self._velocity_msg_callback, 10)
            self.create_subscription(LogDataGeneric, f'{prefix}/angular_velocity', self._angular_velocity_msg_callback, 10)
            self.attitude_setpoint_pub = self.create_publisher(AttitudeSetpoint, f'{prefix}/cmd_attitude', 10)




        self.create_subscription(LogDataGeneric, f'{prefix}/latency', self._latency_msg_callback, 10)

        self.takeoffService = self.create_subscription(Empty, f'/all/mpc_takeoff', self.takeoff, 10)
        self.landService = self.create_subscription(Empty, f'/all/mpc_land', self.land, 10)
        self.trajectoryService = self.create_subscription(Empty, f'/all/mpc_trajectory', self.start_trajectory, 10)
        self.hoverService = self.create_subscription(Empty, f'/all/mpc_hover', self.hover, 10)


        self.mpc_solution_path_pub = self.create_publisher(Path, f'{prefix}/mpc_solution_path', 10)
        self.mpc_time_pub = self.create_publisher(Float32,  f'{prefix}/mpc_time',10)
        self.create_subscription(ReferenceTrajectory,  f'{prefix}/ref_trajectory',  self._ref_trajectory_callback, 10)
        
        self.create_timer(1./rate, self._main_loop)
        self.create_timer(1./rate, self._mpc_solver_loop)


        # logging
        self.actual_state = []
        self.predicted_state = []
        self.controls = []

    def _pose_msg_callback(self, msg: PoseStamped):
        self.position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.attitude = [msg.pose.orientation.w,msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z]
        self.position_stamp = msg.header.stamp



    def _velocity_msg_callback(self, msg: LogDataGeneric):
        self.velocity = [v / 1000.0 for v in msg.values]

    def _angular_velocity_msg_callback(self, msg: LogDataGeneric):
        self.angular_velocity = [v / 1000.0 for v in msg.values]

    def _ref_trajectory_callback(self, msg: ReferenceTrajectory):
        self.ref_trajectory = np.array([
            [*traj_point.position,
            *traj_point.orientation,
            *traj_point.linear_velocity,
            *traj_point.angular_velocity]
            for traj_point in msg.points
        ]).T

    def _latency_msg_callback(self, msg: LogDataGeneric): 
        if not self.latency_initialized:
            if self.latency_cnt < 10:
                self.latency_cnt += 1
                return
            else:
                self.latency = msg.values[0]
                self.latency_initialized = True
        else:
            self.latency = (1.0 - self.alpha) * self.latency + self.alpha * msg.values[0]

        # self.get_logger().info(f"Latency: {self.latency} ms")

    def start_trajectory(self, msg):
        self.flight_mode = 'trajectory'

    def takeoff(self, msg):
        self.flight_mode = 'takeoff'
        
    def hover(self, msg):
        self.flight_mode = 'hover'

    def land(self, msg):
        self.flight_mode = 'land'
    
    def cmd_attitude_setpoint(self, roll, pitch, yaw_rate, thrust_pwm):
        setpoint = AttitudeSetpoint()
        setpoint.roll = roll
        setpoint.pitch = pitch
        setpoint.yaw_rate = yaw_rate
        setpoint.thrust = thrust_pwm
        setpoint.stamp_input = self.used_position_stamp
        self.attitude_setpoint_pub.publish(setpoint)

    def thrust_to_pwm(self, collective_thrust: float):
        # omega_per_rotor = 7460.8*np.sqrt((collective_thrust / 4.0))
        # pwm_per_rotor = 24.5307*(omega_per_rotor - 380.8359)
        collective_thrust = max(collective_thrust, 0.) #  make sure it's not negative
        if self.motors == Motors.MOTOR_CLASSIC:
            return int(max(min(24.5307*(7460.8*np.sqrt((collective_thrust / 4.0)) - 380.8359), 65535),0))
        elif self.motors == Motors.MOTOR_UPGRADE:
            return int(max(min(24.5307*(6462.1*np.sqrt((collective_thrust / 4.0)) - 380.8359), 65535),0))




    def _mpc_solver_loop(self):
        if not self.is_flying:
            return
        
        x0 = np.array([
            *self.position,
            *self.attitude,
            *self.velocity,
            *self.angular_velocity
        ])
        self.used_position_stamp = copy(self.position_stamp)

        trajectory = self.ref_trajectory

        yref = trajectory[:,:-1]
        yref_e = trajectory[:,-1]

        if self.latency_initialized and self.use_predictor:
            self.actual_state.append(x0)
            x = self.mpc_solver.predict_state(x0, self.last_rpm, round(self.latency) / 1000.0)
            self.predicted_state.append(x)
            self.controls.append(self.last_rpm)
            x[12]=x0[12] # use measured yaw rate  # type: ignore

        t0 = self.get_clock().now().nanoseconds
        status, x_mpc, u_mpc, rpm = self.mpc_solver.solve_mpc(x0, yref, yref_e) # type: ignore
        t1 = self.get_clock().now().nanoseconds

        # self.get_logger().info(f"MPC solve status: {status}")

        self.last_rpm = rpm[0,:]
        

        dt_ms = (t1 - t0) / 1e6
        mpc_time_msg = Float32()
        mpc_time_msg.data = dt_ms
        self.mpc_time_pub.publish(mpc_time_msg)


        self.control_queue = deque(u_mpc)

        if self.plot_trajectory:
            mpc_solution_path = Path()
            mpc_solution_path.header.frame_id = 'world'
            mpc_solution_path.header.stamp = self.get_clock().now().to_msg()

            for i in range(self.mpc_N):
                mpc_pose = PoseStamped()
                mpc_pose.pose.position.x = x_mpc[i,0]
                mpc_pose.pose.position.y = x_mpc[i,1]
                mpc_pose.pose.position.z = x_mpc[i,2]
                mpc_solution_path.poses.append(mpc_pose) # type: ignore

            self.mpc_solution_path_pub.publish(mpc_solution_path)


    def _main_loop(self):
        if self.flight_mode == 'idle':
            return

        if not self.position or not self.velocity or not self.attitude:
            self.get_logger().warning("Empty state message.")
            return
        
        if not self.is_flying:
            self.is_flying = True
            self.cmd_attitude_setpoint(0.,0.,0.,0)

        if self.control_queue is not None:
            control = self.control_queue.popleft()
            thrust_pwm = self.thrust_to_pwm(control[3])
            yawrate = 0.
            self.cmd_attitude_setpoint(control[0], 
                                       control[1], 
                                       control[2], 
                                       thrust_pwm)


def main():
    crazyflie_mpc_config_yaml = os.path.join(
        get_package_share_directory('crazyflie_mpc'),
        'config',
        'mpc.yaml')
    
    with open(crazyflie_mpc_config_yaml, 'r') as file:
        crazyflie_mpc_config = yaml.safe_load(file)
    
    n_agents = crazyflie_mpc_config['n_agents']
    build_acados = crazyflie_mpc_config['build_acados']

    rclpy.init()

    # Quadrotor Parameters
    mass = crazyflie_mpc_config['drone_properties']['mass']
    arm_length = crazyflie_mpc_config['drone_properties']['arm_length']
    Ixx = crazyflie_mpc_config['drone_properties']['Ixx']
    Iyy = crazyflie_mpc_config['drone_properties']['Iyy']
    Izz = crazyflie_mpc_config['drone_properties']['Izz']
    cm = crazyflie_mpc_config['drone_properties']['cm']
    tau = crazyflie_mpc_config['simple_model']['attitude_time_constant']
    motorConstant = crazyflie_mpc_config['drone_properties']['motorConstant']
    momentConstant = crazyflie_mpc_config['drone_properties']['momentConstant']

    # MPC Parameters
    mpc_tf = crazyflie_mpc_config['mpc']['horizon']
    mpc_N = crazyflie_mpc_config['mpc']['num_steps']
    control_update_rate = crazyflie_mpc_config['mpc']['control_update_rate']
    plot_trajectory = crazyflie_mpc_config['mpc']['plot_trajectory']
    kappa = crazyflie_mpc_config['full_model']['kappa']
    use_predictor = crazyflie_mpc_config['full_model']['use_predictor']
    delay_relay = crazyflie_mpc_config['delay_relay']['enabled']

    print(f'mass: {mass}, arm_length: {arm_length}, Ixx: {Ixx}, Iyy: {Iyy}, Izz: {Izz}, cm: {cm}, tau: {tau}, mpc_tf: {mpc_tf}, mpc_N: {mpc_N}, control_update_rate: {control_update_rate}, plot_trajectory: {plot_trajectory}')

    quadrotor_dynamics = QuadrotorFull(mass, arm_length, Ixx, Iyy, Izz, cm, tau, motorConstant, momentConstant)
    acados_c_generated_code_path = pathlib.Path(get_package_share_directory('crazyflie_mpc')).resolve() / 'acados_generated_files'
    mpc_solver = TrajectoryTrackingMpc('crazyflie', quadrotor_dynamics, mpc_tf, mpc_N, kappa, code_export_directory=acados_c_generated_code_path)
    # if build_acados:
    #     mpc_solver.generate_mpc()
    nodes = [CrazyflieMPC('cf_'+str(i), mpc_solver, quadrotor_dynamics, mpc_N, mpc_tf, control_update_rate, plot_trajectory, use_predictor, delay_relay) for i in np.arange(1, 1 + n_agents)]
    executor = executors.MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
    try:
        while rclpy.ok():
            node.get_logger().info('Beginning multiagent executor, shut down with CTRL-C')
            executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.\n')

        actual_state = np.array(nodes[0].actual_state)
        predicted_state = np.array(nodes[0].predicted_state)
        controls = np.array(nodes[0].controls)
        columns = (
            [f'actual_{i}' for i in range(actual_state.shape[1])] +
            [f'predicted_{i}' for i in range(predicted_state.shape[1])] +
            [f'control_{i}' for i in range(controls.shape[1])]
        )
        data = np.hstack([actual_state, predicted_state, controls])
        df = pd.DataFrame(data, columns=columns)
        df.to_csv('mpc_full_delay_states.csv', index=False)

    for node in nodes:
        node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
   main()