# mpc_bringup.launch.py
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, TimerAction, GroupAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os
from launch_ros.actions import Node

def generate_launch_description():
    
    crazyflie_mpc_config_yaml = os.path.join(
        get_package_share_directory('crazyflie_mpc'),
        'config',
        'mpc.yaml')
    
    with open(crazyflie_mpc_config_yaml, 'r') as file:
        crazyflie_mpc_config = yaml.safe_load(file)
    
    n_agents = str(crazyflie_mpc_config['n_agents'])
    backend        = "cflib"


    crazy_mpc_root = os.path.abspath(os.path.join(get_package_share_directory('crazyflie_mpc'), '..', '..', '..', '..', '..'))
    sitl_script = os.path.join(
        crazy_mpc_root,
        'crazyflie-firmware',
        'tools',
        'crazyflie-simulation',
        'simulator_files',
        'gazebo',
        'launch',
        'sitl_multiagent_square.sh'
    )
    
    
    crazyflies_yaml = os.path.join(
        get_package_share_directory('crazyflie_mpc'),
        'config',
        'crazyswarm2_full_sim.yaml')


    # 1) SITL (bash)
    sitl = ExecuteProcess(
        cmd=['bash', sitl_script, '-n', n_agents, '-m', 'crazyflie'],
        output='screen'
    )

    # 2) crazyflie launch (backend:=cflib)
    crazyflie_launch_path = os.path.join(
        get_package_share_directory('crazyflie'),
        'launch', 'launch.py'
    )
    crazyflie = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(crazyflie_launch_path),
        launch_arguments={'crazyflies_yaml_file': crazyflies_yaml, 'backend': backend, 'mocap': 'False', 'gui': 'False', 'rviz': 'False'}.items()
    )

    delay_relay_node = Node(
        package='crazyflie_mpc',
        executable='delay_relay',
    )

    path_planner_node = Node(
        package='crazyflie_mpc',
        executable='path_planner'
    )

    # 4) MPC node
    # Uwaga: Twój kod i tak czyta n_agents z YAML; zostawiamy argument CLI, jeśli wspierasz oba.
    crazyflie_mpc_node = Node(
        package='crazyflie_mpc',
        executable='crazyflie_multiagent_mpc_full_delay',
    )

    # Kolejność startu: SITL -> (po 3s) crazyflie -> (po kolejnych 3s) MPC
    staged = [
        GroupAction([sitl]),
        TimerAction(period=10.0, actions=[crazyflie]),
        TimerAction(period=11.0, actions=[delay_relay_node]),
        TimerAction(period=12.0, actions=[path_planner_node]),
        TimerAction(period=13.0, actions=[crazyflie_mpc_node]),
    ]

    return LaunchDescription(staged)
