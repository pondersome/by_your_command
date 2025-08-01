#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare namespace and prefix arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace', 
        default_value='',
        description='Namespace for all nodes'
    )
    prefix_arg = DeclareLaunchArgument(
        'prefix', 
        default_value='',
        description='Prefix/group for all nodes'
    )
    
    # Get config path
    config = os.path.join(
        get_package_share_directory('by_your_command'),
        'config',
        'config.yaml'
    )
    
    # Define nodes
    audio_capturer = Node(
        package='audio_common',
        executable='audio_capturer_node',
        name='audio_capturer_node',
        output='screen',
        parameters=[{
            'chunk': 480,  # 30ms @ 16kHz = exactly 3 WebRTC frames (160 samples each)
            'rate': 16000,
            'device': 14  # Use PulseAudio
        }]
    )
    
    silero_vad = Node(
        package='by_your_command',
        executable='silero_vad_node',
        name='silero_vad_node',
        output='screen',
        parameters=[config]
    )
    
    voice_recorder = Node(
        package='by_your_command',
        executable='voice_chunk_recorder',
        name='voice_chunk_recorder',
        output='screen'
    )
    
    # Group all nodes with namespace handling
    # PushRosNamespace handles empty strings correctly
    nodes_group = GroupAction([
        PushRosNamespace(LaunchConfiguration('namespace')),
        PushRosNamespace(LaunchConfiguration('prefix')),
        audio_capturer,
        silero_vad,
        voice_recorder
    ])
    
    return LaunchDescription([
        namespace_arg,
        prefix_arg,
        nodes_group
    ])
