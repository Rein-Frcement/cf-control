from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
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
                package='drone_math_model',
                executable='drone_model',
                name='drone_model',
                output='screen',
            ),
        ]
    )
