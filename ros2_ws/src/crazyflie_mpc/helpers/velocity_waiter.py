import os
import subprocess
import time

import yaml
from ament_index_python import get_package_share_directory
from crazyflie_interfaces.msg._log_data_generic import LogDataGeneric
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import signal

class VelocityWaiter(Node):
    def __init__(self, timeout=5.0):
        super().__init__('velocity_waiter')
        self.timeout = timeout
        self.msg_received = False
        self.start_time = self.get_clock().now().nanoseconds
        self.create_subscription(LogDataGeneric, '/cf_1/velocity', self.cb, 10)

    def cb(self, msg):
        self.msg_received = True

    def wait_for_message(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            
            now = self.get_clock().now().nanoseconds
            if (now - self.start_time) / 1e9 > self.timeout:

                if self.msg_received:
                    return True
                else:
                    return False

def run_with_retry(save_csv=False):
    while True:
        # Start launch
        proc = subprocess.Popen([
            "ros2", "launch", "crazyflie_mpc", "mpc_config_launch.py",
        ])

        rclpy.init()
        waiter = VelocityWaiter(timeout=15.0)
        got_velocity = waiter.wait_for_message()
        rclpy.shutdown()

        if got_velocity:
            time.sleep(20)  # chwila przerwy przed startem eksperymentu
            print("✅ Velocity przyszło – startuję eksperyment")
            subprocess.run([
                "ros2", "topic", "pub", "-t", "1", "/all/mpc_takeoff", "std_msgs/msg/Empty"
            ])
            time.sleep(15)
            subprocess.run([
                "ros2", "topic", "pub", "-t", "1", "/all/mpc_trajectory", "std_msgs/msg/Empty"
            ])
            time.sleep(15)
            if save_csv:
                subprocess.run([
                    "ros2", "topic", "pub", "-t", "1", "/logger/save_file", "std_msgs/msg/Empty"
                ])
                time.sleep(10)  # chwila na zapisanie pliku
            print("✅ Eksperyment zakończony – zamykam launch")

            proc.send_signal(signal.SIGINT)  # ctrl+c dla całego launch
            proc.wait()
            proc.terminate()   # SIGTERM
            proc.kill()  
            subprocess.run(["pkill", "-9", "-f", "gz"])
            break
        else:
            print("⚠️ Brak velocity – restartuję launch")
            proc.send_signal(signal.SIGINT)
            proc.wait()
            proc.terminate()   # SIGTERM
            proc.kill()  
            subprocess.run(["pkill", "-9", "-f", "gz"])
            time.sleep(10)  # chwila przerwy przed restartem

if __name__ == '__main__':
    
    # kappa_values = [4,6,8,10,12,15]

    # crazyflie_mpc_config_yaml = os.path.join(
    # get_package_share_directory('crazyflie_mpc'),
    # 'config',
    # 'mpc.yaml'
    # )

    # filename = "logged_data_full_model_compensation_kappa"

    # for kappa in kappa_values:
    #     with open(crazyflie_mpc_config_yaml, 'r') as file:
    #         crazyflie_mpc_config = yaml.safe_load(file)

    #     crazyflie_mpc_config['full_model']['kappa'] = kappa
    #     with open(crazyflie_mpc_config_yaml, 'w') as file:
    #         yaml.dump(crazyflie_mpc_config, file, sort_keys=False)

    #     print(f"Uruchamiam eksperyment z kappa {kappa}")
    #     run_with_retry(save_csv=True)

    #     old_name = "logs/logged_data.csv"
    #     new_name = "logs/" + filename + "_" + str(kappa).replace(".", "_") + ".csv"

    #     os.rename(old_name, new_name)

    delay_values = [0,5,10,15,20,25]

    crazyflie_mpc_config_yaml = os.path.join(
    get_package_share_directory('crazyflie_mpc'),
    'config',
    'mpc.yaml'
    )

    filename = "logged_data_full_model_delay"

    for delay in delay_values:
        with open(crazyflie_mpc_config_yaml, 'r') as file:
            crazyflie_mpc_config = yaml.safe_load(file)

        crazyflie_mpc_config['delay_relay']['delay_ms'] = delay
        with open(crazyflie_mpc_config_yaml, 'w') as file:
            yaml.dump(crazyflie_mpc_config, file, sort_keys=False)

        print(f"Uruchamiam eksperyment z opóźnieniem {delay}")
        run_with_retry(save_csv=True)

        old_name = "logs/logged_data.csv"
        new_name = "logs/" + filename + "_" + str(delay).replace(".", "_") + ".csv"
        try:
            os.rename(old_name, new_name)
        except FileNotFoundError:
            print(f"⚠️ Plik {old_name} nie został znaleziony")
        except Exception as e:
            print(f"⚠️ Wystąpił błąd podczas zmiany nazwy pliku: {e}")

    filename = "logged_data_full_model_compensation_delay"

    for delay in delay_values:
        with open(crazyflie_mpc_config_yaml, 'r') as file:
            crazyflie_mpc_config = yaml.safe_load(file)

        crazyflie_mpc_config['delay_relay']['delay_ms'] = delay
        crazyflie_mpc_config['full_model']['use_predictor'] = True
        with open(crazyflie_mpc_config_yaml, 'w') as file:
            yaml.dump(crazyflie_mpc_config, file, sort_keys=False)

        print(f"Uruchamiam eksperyment z opóźnieniem {delay}")
        run_with_retry(save_csv=True)

        old_name = "logs/logged_data.csv"
        new_name = "logs/" + filename + "_" + str(delay).replace(".", "_") + ".csv"

        try:
            os.rename(old_name, new_name)
        except FileNotFoundError:
            print(f"⚠️ Plik {old_name} nie został znaleziony")
        except Exception as e:
            print(f"⚠️ Wystąpił błąd podczas zmiany nazwy pliku: {e}")