#!/usr/bin/env python3
"""
Launch file for Gemini Live Agent with Pipecat

Launches the Gemini Live agent for multimodal interaction with voice and vision.

Author: Karim Virani
Version: 1.0
Date: August 2025
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, GroupAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directories
    pkg_dir = get_package_share_directory('by_your_command')
    config_dir = os.path.join(pkg_dir, 'config')
    
    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='/grunt1/agent',
        description='Namespace for the agent nodes'
    )
    
    prefix_arg = DeclareLaunchArgument(
        'prefix',
        default_value='grunt1/',
        description='Prefix for joint names'
    )
    
    agent_type_arg = DeclareLaunchArgument(
        'agent_type',
        default_value='multimodal',
        description='Agent type: conversation|command|visual|multimodal'
    )
    
    enable_bridge_arg = DeclareLaunchArgument(
        'enable_bridge',
        default_value='true',
        description='Enable ROS AI Bridge'
    )
    
    enable_vad_arg = DeclareLaunchArgument(
        'enable_vad',
        default_value='true',
        description='Enable VAD node for voice detection'
    )
    
    enable_camera_arg = DeclareLaunchArgument(
        'enable_camera',
        default_value='true',
        description='Enable camera input for visual processing'
    )
    
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='gemini_live_agent.yaml',
        description='Configuration file for Gemini agent'
    )
    
    # ROS AI Bridge node
    bridge_node = Node(
        package='by_your_command',
        executable='ros_ai_bridge',
        name='ros_ai_bridge',
        parameters=[
            PathJoinSubstitution([config_dir, 'gemini_live_agent.yaml'])
        ],
        condition=IfCondition(LaunchConfiguration('enable_bridge')),
        output='screen'
    )
    
    # Audio capture node
    audio_capture_node = Node(
        package='audio_common',
        executable='audio_capturer_node',
        name='audio_capturer_node',
        parameters=[{
            'format': 1,  # S16_LE
            'channels': 1,
            'rate': 16000,
            'chunk': 512
        }],
        output='screen'
    )
    
    # VAD node
    vad_node = Node(
        package='by_your_command',
        executable='silero_vad_node',
        name='silero_vad',
        parameters=[
            PathJoinSubstitution([config_dir, 'config.yaml'])
        ],
        condition=IfCondition(LaunchConfiguration('enable_vad')),
        output='screen'
    )
    
    # Echo suppressor node
    echo_suppressor_node = Node(
        package='by_your_command',
        executable='echo_suppressor',
        name='echo_suppressor',
        output='screen'
    )
    
    # Simple audio player for responses
    audio_player_node = Node(
        package='by_your_command',
        executable='simple_audio_player',
        name='simple_audio_player',
        parameters=[{
            'sample_rate': 24000,  # Gemini outputs at 24kHz
            'channels': 1
        }],
        output='screen'
    )
    
    # Command processor (for routing commands to robot subsystems)
    command_processor_node = Node(
        package='by_your_command',
        executable='command_processor',
        name='command_processor',
        parameters=[{
            'command_transcript_topic': 'command_transcript',
            'arm_preset_topic': '/grunt1/arm_preset',
            'behavior_command_topic': '/grunt1/behavior_command'
        }],
        output='screen'
    )
    
    # Gemini Live Agent (Python process, not a ROS node)
    gemini_agent_process = ExecuteProcess(
        cmd=[
            'python3', '-m', 'agents.gemini_live.gemini_live_agent'
        ],
        cwd='/home/karim/ros2_ws/src/by_your_command',
        env={
            'PYTHONPATH': os.path.join(pkg_dir, '..', '..', 'src', 'by_your_command'),
            'GEMINI_API_KEY': os.environ.get('GEMINI_API_KEY', ''),
            'AGENT_TYPE': LaunchConfiguration('agent_type'),
            'CONFIG_FILE': PathJoinSubstitution([config_dir, LaunchConfiguration('config_file')])
        },
        output='screen',
        name='gemini_live_agent'
    )
    
    # Log configuration info
    log_info = LogInfo(
        msg=['Launching Gemini Live agent with type: ', LaunchConfiguration('agent_type')]
    )
    
    # Group all nodes with namespace handling
    nodes_group = GroupAction([
        PushRosNamespace(LaunchConfiguration('namespace')),
        # Note: prefix is for joint names, not node namespace
        bridge_node,
        audio_capture_node,
        vad_node,
        echo_suppressor_node,
        audio_player_node,
        command_processor_node
    ])
    
    # Agent runs as separate process
    agent_group = GroupAction([
        gemini_agent_process
    ])
    
    return LaunchDescription([
        # Arguments
        namespace_arg,
        prefix_arg,
        agent_type_arg,
        enable_bridge_arg,
        enable_vad_arg,
        enable_camera_arg,
        config_file_arg,
        
        # Logging
        log_info,
        
        # Node groups
        nodes_group,
        agent_group
    ])