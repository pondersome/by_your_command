#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('byc_'),
        'config',
        'config.yaml'
    )
    return LaunchDescription([
        Node(
            package='byc_',
            executable='silero_vad_node.py',
            name='silero_vad_node',
            output='screen',
            parameters=[config]
        ),
        Node(
            package='byc_',
            executable='interaction_node.py',
            name='interaction_node',
            output='screen',
            parameters=[config]
        ),
    ])
