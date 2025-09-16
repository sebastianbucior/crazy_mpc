import os

import yaml
from ament_index_python import get_package_share_directory
import rclpy, heapq, random
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped
from crazyflie_interfaces.msg import LogDataGeneric, AttitudeSetpoint


class DelayRelay(Node):
    def __init__(self, cf_name: str, delay_ms: float, jitter_ms: float) :
        super().__init__('delay_relay')

        self.delay_ms = delay_ms
        self.jitter_ms = jitter_ms

        prefix = cf_name


        self.create_subscription(
            PoseStamped,
            f'{prefix}/pose',
            self._pose_msg_callback,
            10)
        
        self.create_subscription(
            LogDataGeneric,
            f'{prefix}/velocity',
            self._velocity_msg_callback,
            10)
        
        self.create_subscription(
            LogDataGeneric,
            f'{prefix}/angular_velocity',
            self._angular_velocity_msg_callback,
            10)

        self.create_subscription(
            AttitudeSetpoint,
            f'{prefix}/cmd_attitude_d',
            self._attitude_setpoint_msg_callback,
            10)
        
        self.pose_pub_d = self.create_publisher(PoseStamped, f'{prefix}/pose_d', 10)
        self.velocity_pub_d = self.create_publisher(LogDataGeneric, f'{prefix}/velocity_d', 10)
        self.angular_velocity_pub_d = self.create_publisher(LogDataGeneric, f'{prefix}/angular_velocity_d', 10)
        self.attitude_setpoint_pub_d = self.create_publisher(AttitudeSetpoint, f'{prefix}/cmd_attitude', 10)

        self.counter = 1

        self.pose_queue = []
        self.velocity_queue = []
        self.angular_velocity_queue = []
        self.attitude_setpoint_queue = []

        self.timer = self.create_timer(0.001, self._timer_callback)

        self.delay_ns = 0
        self.delay_timer = self.create_timer(0.1, self._compute_delay)
        
    def _compute_delay(self):
        mu = self.delay_ms
        jitter = random.gauss(0.0, 0.3) * self.jitter_ms
        self.delay_ns = int(max(0.0, (mu + jitter)) * 1e6)

    def _pose_msg_callback(self, msg: PoseStamped):
        # self._logger.info(f"Delay : {self.delay_ns} ns")

        t_due = self.get_clock().now().nanoseconds + self.delay_ns

        heapq.heappush(self.pose_queue, (t_due, self.counter, msg))
        self.counter += 1

    def _velocity_msg_callback(self, msg: LogDataGeneric):
        t_due = self.get_clock().now().nanoseconds + self.delay_ns

        heapq.heappush(self.velocity_queue, (t_due, self.counter, msg))
        self.counter += 1

    def _angular_velocity_msg_callback(self, msg: LogDataGeneric):
        t_due = self.get_clock().now().nanoseconds + self.delay_ns

        heapq.heappush(self.angular_velocity_queue, (t_due, self.counter, msg))
        self.counter += 1

    def _attitude_setpoint_msg_callback(self, msg: AttitudeSetpoint):
        t_due = self.get_clock().now().nanoseconds + self.delay_ns

        heapq.heappush(self.attitude_setpoint_queue, (t_due, self.counter, msg))
        self.counter += 1


    def _timer_callback(self):
        t_now = self.get_clock().now().nanoseconds

        while self.pose_queue and self.pose_queue[0][0] <= t_now:
            _, _, msg = heapq.heappop(self.pose_queue)
            self.pose_pub_d.publish(msg)

        while self.velocity_queue and self.velocity_queue[0][0] <= t_now:
            _, _, msg = heapq.heappop(self.velocity_queue)
            self.velocity_pub_d.publish(msg)

        while self.angular_velocity_queue and self.angular_velocity_queue[0][0] <= t_now:
            _, _, msg = heapq.heappop(self.angular_velocity_queue)
            self.angular_velocity_pub_d.publish(msg)

        while self.attitude_setpoint_queue and self.attitude_setpoint_queue[0][0] <= t_now:
            _, _, msg = heapq.heappop(self.attitude_setpoint_queue)
            self.attitude_setpoint_pub_d.publish(msg)

def main():
    crazyflie_mpc_config_yaml = os.path.join(
        get_package_share_directory('crazyflie_mpc'),
        'config',
        'mpc.yaml')
    
    with open(crazyflie_mpc_config_yaml, 'r') as file:
        crazyflie_mpc_config = yaml.safe_load(file)

    delay_ms = crazyflie_mpc_config['delay_relay']['delay_ms']
    jitter_ms = crazyflie_mpc_config['delay_relay']['jitter_ms']


    rclpy.init()
    node = DelayRelay('cf_1', delay_ms, jitter_ms)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
