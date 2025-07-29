#!/usr/bin/env python3
"""
OpenAI Dual Agent Launch File

Runs both conversational and command extraction agents simultaneously:
- Conversational agent: Natural dialogue and Q&A
- Command agent: Dedicated robot command extraction

Both agents process the same voice input but with different purposes.

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
    # Both agents use their specific configs with different agent_ids and topics
    conv_agent_config = os.path.join(pkg_dir, 'config', 'oai_realtime_agent.yaml')
    cmd_agent_config = os.path.join(pkg_dir, 'config', 'oai_command_agent.yaml')
    
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
    
    conv_model_arg = DeclareLaunchArgument(
        'conv_model',
        default_value='gpt-4o-realtime-preview',
        description='OpenAI model for conversation'
    )
    
    cmd_model_arg = DeclareLaunchArgument(
        'cmd_model',
        default_value='gpt-4o-realtime-preview',
        description='OpenAI model for command extraction (could use cheaper model)'
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
            'config_file': conv_agent_config,  # Use conversational config for bridge
            'websocket_server.enabled': True,
            'websocket_server.host': '0.0.0.0',
            'websocket_server.port': 8765,
            'websocket_server.max_connections': 10  # Support multiple agents
        }]
    )
    
    # Conversational OpenAI Agent
    conversational_agent = ExecuteProcess(
        cmd=[
            '/home/karim/ros2_ws/install/by_your_command/lib/by_your_command/oai_realtime_agent',
            '--config', conv_agent_config,
            '--pause-timeout', LaunchConfiguration('pause_timeout'),
            '--prompt-id', 'barney_conversational'
        ],
        output='screen',
        additional_env={
            'OPENAI_MODEL': LaunchConfiguration('conv_model'),
            'OPENAI_VOICE': LaunchConfiguration('voice'),
            'PAUSE_TIMEOUT': LaunchConfiguration('pause_timeout')
        }
    )
    
    # Command Extraction OpenAI Agent
    command_agent = ExecuteProcess(
        cmd=[
            '/home/karim/ros2_ws/install/by_your_command/lib/by_your_command/oai_realtime_agent',
            '--config', cmd_agent_config,
            '--pause-timeout', LaunchConfiguration('pause_timeout'),
            '--prompt-id', 'barney_command_extractor'
        ],
        output='screen',
        additional_env={
            'OPENAI_MODEL': LaunchConfiguration('cmd_model'),
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
    
    # Command transcript monitor (optional debug tool)
    command_monitor = LogInfo(
        msg=['Command extractor will publish to /command_transcript and /command_detected']
    )
    
    # Startup message
    startup_message = LogInfo(
        msg=[
            '🚀 Starting Dual Agent System\n',
            '🗣️  Conversational Agent:\n',
            '    Model: ', LaunchConfiguration('conv_model'), '\n',
            '    Voice: ', LaunchConfiguration('voice'), '\n', 
            '🤖 Command Extraction Agent:\n',
            '    Model: ', LaunchConfiguration('cmd_model'), '\n',
            '    Topics: /command_transcript, /command_detected\n',
            '⏱️  Pause timeout: ', LaunchConfiguration('pause_timeout'), 's\n',
            '🔊 Both agents listening for speech...'
        ]
    )
    
    return LaunchDescription([
        # Launch arguments
        openai_api_key_arg,
        pause_timeout_arg,
        conv_model_arg,
        cmd_model_arg,
        voice_arg,
        verbose_arg,
        enable_voice_recorder_arg,
        
        # Startup message
        startup_message,
        command_monitor,
        
        # Core audio pipeline
        audio_capturer,
        echo_suppressor,
        audio_player,
        silero_vad,
        
        # Bridge and agents
        ros_ai_bridge,
        conversational_agent,
        command_agent,
        
        # Optional debugging
        voice_recorder
    ])