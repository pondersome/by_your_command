#!/usr/bin/env python3
"""
Gemini Single Agent Launch File

Launches a single Gemini Live agent for testing.
Can be used for conversation, command extraction, or scene description modes.

Author: Karim Virani
Version: 1.0
Date: August 2025
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, EnvironmentVariable, PythonExpression
from launch_ros.actions import Node, PushRosNamespace
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('by_your_command')
    
    # Configuration paths - default to conversation agent
    default_config = os.path.join(pkg_dir, 'config', 'gemini_conversation_agent.yaml')
    bridge_config = os.path.join(pkg_dir, 'config', 'config.yaml')  # Standard bridge config
    
    # Namespace and prefix arguments
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
    
    # Launch arguments
    config_arg = DeclareLaunchArgument(
        'config',
        default_value=default_config,
        description='Path to agent configuration file'
    )
    
    agent_type_arg = DeclareLaunchArgument(
        'agent_type',
        default_value='conversation',
        description='Agent type: conversation, command, or scene'
    )
    
    prompt_id_arg = DeclareLaunchArgument(
        'prompt_id',
        default_value='barney_conversational_gemini',
        description='Prompt ID from prompts.yaml'
    )
    
    gemini_api_key_arg = DeclareLaunchArgument(
        'gemini_api_key',
        default_value=EnvironmentVariable('GEMINI_API_KEY', default_value=''),
        description='Gemini API key (defaults to GEMINI_API_KEY environment variable)'
    )
    
    model_arg = DeclareLaunchArgument(
        'model',
        default_value='models/gemini-2.0-flash-exp',
        description='Gemini model to use'
    )
    
    voice_arg = DeclareLaunchArgument(
        'voice',
        default_value='Kore',
        description='Gemini voice ID'
    )
    
    pause_timeout_arg = DeclareLaunchArgument(
        'pause_timeout',
        default_value='10.0',
        description='Session pause timeout in seconds'
    )
    
    video_fps_arg = DeclareLaunchArgument(
        'video_fps',
        default_value='1.0',
        description='Video frame rate (fps) - only for conversation agent'
    )
    
    verbose_arg = DeclareLaunchArgument(
        'verbose',
        default_value='false',
        description='Enable verbose logging'
    )
    
    
    # Audio capture node
    audio_capturer = Node(
        package='audio_common',
        executable='audio_capturer_node',
        name='audio_capturer_node',
        output='screen',
        parameters=[{
            'chunk': 512,  # 32ms @ 16kHz
            'rate': 16000,
            'device': -1
        }]
    )
    
    # Simple audio player
    audio_player = Node(
        package='by_your_command',
        executable='simple_audio_player',
        name='simple_audio_player',
        output='screen',
        parameters=[{
            'topic': 'response_voice',
            'sample_rate': 24000,  # Gemini output rate
            'channels': 1,
            'device': -1
        }]
    )
    
    # Silero VAD node
    silero_vad = Node(
        package='by_your_command',
        executable='silero_vad_node',
        name='silero_vad_node',
        output='screen',
        parameters=[bridge_config],
        remappings=[
            ('audio', 'audio')
        ]
    )
    
    # ROS AI Bridge
    ros_ai_bridge = Node(
        package='by_your_command',
        executable='ros_ai_bridge', 
        name='ros_ai_bridge',
        output='screen',
        parameters=[{
            'namespace': LaunchConfiguration('namespace'),
            'prefix': LaunchConfiguration('prefix'),
            'config_file': LaunchConfiguration('config'),
            'websocket_server.enabled': True,
            'websocket_server.host': '0.0.0.0',
            'websocket_server.port': 8765,
            'websocket_server.max_connections': 5
        }],
        remappings=[
            ('camera/image_raw', '/grunt1/arm1/cam_live/color/image_raw')
        ]
    )
    
    # Gemini Live Agent
    gemini_agent = ExecuteProcess(
        cmd=[
            '/home/karim/ros2_ws/install/by_your_command/lib/by_your_command/gemini_live_agent',
            '--config', LaunchConfiguration('config'),
            '--pause-timeout', LaunchConfiguration('pause_timeout'),
            '--prompt-id', LaunchConfiguration('prompt_id')
        ],
        output='screen',
        additional_env={
            'GEMINI_API_KEY': LaunchConfiguration('gemini_api_key'),
            'GEMINI_MODEL': LaunchConfiguration('model'),
            'VIDEO_FPS': LaunchConfiguration('video_fps')
        }
    )
    
    # Command processor (only for command agent)
    command_processor = Node(
        package='by_your_command',
        executable='command_processor',
        name='command_processor',
        output='screen',
        parameters=[{
            'command_transcript_topic': 'response_cmd',
            'arm_preset_topic': '/grunt1/arm_preset',
            'behavior_command_topic': '/grunt1/behavior_command'
        }],
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('agent_type'), "' == 'command'"
        ]))
    )
    
    # Voice recorder for debugging
    voice_recorder = Node(
        package='by_your_command',
        executable='voice_chunk_recorder',
        name='voice_recorder_output',
        output='screen',
        parameters=[{
            'output_dir': '/tmp/prompt_voice/gemini_single',
            'input_mode': 'audio_data',
            'input_topic': 'response_voice',
            'input_sample_rate': 24000,
            'audio_timeout': 10.0
        }]
    )
    
    # Startup message
    startup_message = LogInfo(
        msg=[
            '🚀 Starting Gemini Single Agent\n',
            '🤖 Type: ', LaunchConfiguration('agent_type'), '\n',
            '📋 Config: ', LaunchConfiguration('config'), '\n',
            '🎯 Prompt: ', LaunchConfiguration('prompt_id'), '\n',
            '🎙️  Model: ', LaunchConfiguration('model'), '\n',
            '🔊 Ready for input...'
        ]
    )
    
    # Group nodes with namespace
    nodes_group = GroupAction([
        PushRosNamespace(LaunchConfiguration('namespace')),
        PushRosNamespace(LaunchConfiguration('prefix')),
        audio_capturer,
        audio_player,
        silero_vad,
        ros_ai_bridge,
        command_processor,
        voice_recorder
    ])
    
    return LaunchDescription([
        # Arguments
        namespace_arg,
        prefix_arg,
        config_arg,
        agent_type_arg,
        prompt_id_arg,
        gemini_api_key_arg,
        model_arg,
        voice_arg,
        pause_timeout_arg,
        video_fps_arg,
        verbose_arg,
        
        # Startup
        startup_message,
        
        # Nodes
        nodes_group,
        
        # Agent
        gemini_agent
    ])