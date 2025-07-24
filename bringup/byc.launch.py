#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('by_your_command'),
        'config',
        'config.yaml'
    )
    return LaunchDescription([
        Node(
            package='audio_common',
            executable='audio_capturer_node',
            name='audio_capturer_node',
            output='screen'
        ),
        Node(
            package='by_your_command',
            executable='silero_vad_node',
            name='silero_vad_node',
            output='screen',
            parameters=[config]
        ),
        Node(
            package='by_your_command',
            executable='speech_chunk_recorder',
            name='speech_chunk_recorder',
            output='screen'
        ),
    ])
