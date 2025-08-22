#!/usr/bin/env python3
"""
Launch file for Gemini Live Agent with Vision Support

This launches the complete system with Gemini Live agent configured for
video streaming. Note that video reduces session limit to 2 minutes!

Author: Karim Virani
Date: August 2025
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    """Generate launch description for Gemini with vision"""
    
    # Get package directory
    pkg_dir = get_package_share_directory('by_your_command')
    
    # Configuration files
    bridge_config = os.path.join(pkg_dir, 'config', 'bridge_gemini_vision.yaml')
    gemini_agent_config = os.path.join(pkg_dir, 'config', 'gemini_live_agent.yaml')
    
    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Namespace for all nodes'
    )
    
    prefix_arg = DeclareLaunchArgument(
        'prefix',
        default_value='',
        description='Prefix for topics within namespace'
    )
    
    enable_video_arg = DeclareLaunchArgument(
        'enable_video',
        default_value='true',
        description='Enable video streaming (reduces session to 2 minutes)'
    )
    
    # Nodes
    audio_capturer = Node(
        package='audio_common',
        executable='audio_capturer_node',
        name='audio_capturer_node',
        namespace=LaunchConfiguration('namespace'),
        parameters=[{
            'format': 1,  # S16LE
            'channels': 1,
            'rate': 16000,
            'chunk': 512
        }],
        remappings=[
            ('audio', 'audio_in'),
            ('audio_info', 'audio_info')
        ]
    )
    
    silero_vad = Node(
        package='by_your_command',
        executable='silero_vad_node',
        name='silero_vad_node',
        namespace=LaunchConfiguration('namespace'),
        parameters=[bridge_config],  # VAD params from bridge config
        remappings=[
            ('audio_in', 'audio_in'),
            ('voice_active', 'voice_active'),
            ('voice_chunks', 'voice_chunks')
        ]
    )
    
    ros_ai_bridge = Node(
        package='by_your_command',
        executable='ros_ai_bridge',
        name='ros_ai_bridge',
        namespace=LaunchConfiguration('namespace'),
        parameters=[
            bridge_config,
            {'namespace': LaunchConfiguration('namespace')},
            {'prefix': LaunchConfiguration('prefix')}
        ],
        remappings=[
            ('voice_chunks', 'voice_chunks'),
            ('audio_out', 'audio_out'),
            ('llm_transcript', 'llm_transcript'),
            ('command_transcript', 'command_transcript'),
            ('interruption_signal', 'interruption_signal')
        ],
        output='screen'
    )
    
    simple_audio_player = Node(
        package='by_your_command',
        executable='simple_audio_player',
        name='simple_audio_player',
        namespace=LaunchConfiguration('namespace'),
        remappings=[
            ('audio_out', 'audio_out'),
            ('interruption_signal', 'interruption_signal')
        ]
    )
    
    # Gemini Live Agent with vision config
    gemini_agent = Node(
        package='by_your_command',
        executable='gemini_live_agent',
        name='gemini_live_agent',
        namespace=LaunchConfiguration('namespace'),
        parameters=[
            {'config': gemini_agent_config},
            {'enable_video': LaunchConfiguration('enable_video')}
        ],
        output='screen',
        arguments=['--config', gemini_agent_config]
    )
    
    # Log information about video mode
    video_info = LogInfo(
        msg="⚠️  WARNING: Video mode enabled - Gemini sessions limited to 2 MINUTES!"
    )
    
    return LaunchDescription([
        namespace_arg,
        prefix_arg,
        enable_video_arg,
        video_info,
        audio_capturer,
        silero_vad,
        ros_ai_bridge,
        simple_audio_player,
        gemini_agent
    ])