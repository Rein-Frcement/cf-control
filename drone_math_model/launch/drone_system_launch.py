import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    gazebo_launch = LaunchConfiguration('gazebo_launch')
    pkg_ros_gz_bringup = get_package_share_directory('ros_gz_crazyflie_bringup')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_bringup, 'launch', 'crazyflie_simulation.launch.py')
        ),
        launch_arguments={'gazebo_launch': gazebo_launch}.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'gazebo_launch', default_value='True', description='Launch Gazebo simulation'
            ),
            gz_sim,
            Node(
                package='drone_math_model',
                executable='drone_trajectory_publisher',
                name='drone_trajectory_publisher',
                output='screen',
            ),
            Node(
                package='drone_math_model',
                executable='drone_controller',
                name='drone_controller',
                output='screen',
            ),
            Node(
                package='cf_control',
                executable='mixer',
                name='mixer',
                output='screen',
            ),
            Node(
                package='drone_math_model',
                executable='drone_model',
                name='drone_model',
                output='screen',
            ),
        ]
    )
