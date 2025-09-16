from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'crazyflie_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sebastian',
    maintainer_email='sebastian11259@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'crazyflie_multiagent_mpc = crazyflie_mpc.crazyflie_multiagent_mpc:main',
        'crazyflie_multiagent_mpc_full = crazyflie_mpc_full.crazyflie_multiagent_mpc_full:main',
        'crazyflie_multiagent_mpc_full_delay = crazyflie_mpc_full_delay.crazyflie_multiagent_mpc_full_delay:main',
        'delay_relay = helpers.delay_relay:main',
        'path_planner = helpers.path_planner:main',
    ],
},
)
