#!/usr/bin/env python3
"""
OpenAI Realtime Agent Launch File

Coordinates all nodes needed for OpenAI Realtime API integration:
- Audio capture and VAD processing
- ROS AI Bridge for data transport  
- OpenAI Realtime Agent with session management

Author: Karim Virani
Version: 1.0
Date: July 2025
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, EnvironmentVariable, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('by_your_command')
    
    # Configuration paths
    bridge_config = os.path.join(pkg_dir, 'config', 'config.yaml')
    agent_config = os.path.join(pkg_dir, 'config', 'oai_realtime_agent.yaml')
    
    # Launch arguments
    openai_api_key_arg = DeclareLaunchArgument(
        'openai_api_key',
        default_value='',
        description='OpenAI API key (or set OPENAI_API_KEY environment variable)'
    )
    
    pause_timeout_arg = DeclareLaunchArgument(
        'pause_timeout',
        default_value='10.0',
        description='Session pause timeout in seconds'
    )
    
    model_arg = DeclareLaunchArgument(
        'model',
        default_value='gpt-4o-realtime-preview',
        description='OpenAI model to use'
    )
    
    voice_arg = DeclareLaunchArgument(
        'voice',
        default_value='alloy',
        description='OpenAI voice (alloy, echo, fable, onyx, nova, shimmer)'
    )
    
    verbose_arg = DeclareLaunchArgument(
        'verbose',
        default_value='false',
        description='Enable verbose logging'
    )
    
    enable_voice_recorder_arg = DeclareLaunchArgument(
        'enable_voice_recorder',
        default_value='false',
        description='Enable voice chunk recorder for debugging'
    )
    
    # Audio capture node
    audio_capturer = Node(
        package='audio_common',
        executable='audio_capturer_node',
        name='audio_capturer_node',
        output='screen'
    )
    
    # Simple audio player for OpenAI response playback
    audio_player = Node(
        package='by_your_command',
        executable='simple_audio_player',
        name='simple_audio_player',
        output='screen',
        parameters=[{
            'topic': '/audio_out',
            'sample_rate': 24000,
            'channels': 1,
            'device': -1    # Default output device
        }]
    )
    
    # Echo suppressor - mutes mic when assistant speaks
    echo_suppressor = Node(
        package='by_your_command',
        executable='echo_suppressor',
        name='echo_suppressor',
        output='screen'
    )
    
    # Silero VAD node for speech detection
    silero_vad = Node(
        package='by_your_command',
        executable='silero_vad_node',
        name='silero_vad_node',
        output='screen',
        parameters=[bridge_config],
        remappings=[
            ('/audio', '/audio_filtered')  # Listen to filtered audio
        ]
    )
    
    # ROS AI Bridge for data transport with WebSocket enabled
    ros_ai_bridge = Node(
        package='by_your_command',
        executable='ros_ai_bridge', 
        name='ros_ai_bridge',
        output='screen',
        parameters=[{
            'config_file': agent_config,
            'websocket_server.enabled': True,
            'websocket_server.host': '0.0.0.0',
            'websocket_server.port': 8765
        }]
    )
    
    # OpenAI Realtime Agent (standalone process)
    openai_agent = ExecuteProcess(
        cmd=[
            '/home/karim/ros2_ws/install/by_your_command/lib/by_your_command/oai_realtime_agent',
            '--config', agent_config,
            '--pause-timeout', LaunchConfiguration('pause_timeout')
        ],
        output='screen',
        additional_env={
            'OPENAI_MODEL': LaunchConfiguration('model'),
            'OPENAI_VOICE': LaunchConfiguration('voice'),
            'PAUSE_TIMEOUT': LaunchConfiguration('pause_timeout')
        }
    )
    
    # Optional: Voice chunk recorder for debugging
    voice_recorder = Node(
        package='by_your_command',
        executable='voice_chunk_recorder',
        name='voice_chunk_recorder',
        output='screen',
        parameters=[{
            'output_dir': '/tmp/voice_chunks',
            'input_mode': 'audio_data',
            'input_topic': '/audio_out',
            'input_sample_rate': 24000,
            'audio_timeout': 10.0
        }],
        condition=IfCondition(LaunchConfiguration('enable_voice_recorder'))
    )
    
    # Startup message
    startup_message = LogInfo(
        msg=[
            '🚀 Starting OpenAI Realtime Agent System\n',
            '📡 Model: ', LaunchConfiguration('model'), '\n',
            '🎙️  Voice: ', LaunchConfiguration('voice'), '\n', 
            '⏱️  Pause timeout: ', LaunchConfiguration('pause_timeout'), 's\n',
            '🔊 Listening for speech...'
        ]
    )
    
    return LaunchDescription([
        # Launch arguments
        openai_api_key_arg,
        pause_timeout_arg,
        model_arg,
        voice_arg,
        verbose_arg,
        enable_voice_recorder_arg,
        
        # Startup message
        startup_message,
        
        # Nodes
        audio_capturer,
        echo_suppressor,
        audio_player,
        silero_vad,
        ros_ai_bridge,
        openai_agent,
        voice_recorder
    ])