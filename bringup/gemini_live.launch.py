#!/usr/bin/env python3
"""
Gemini Live Agent Launch File

Launches the Gemini Live multimodal agent for voice and vision processing.
Based on the dual agent architecture but initially with single Gemini agent.

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
    
    # Configuration paths
    bridge_config = os.path.join(pkg_dir, 'config', 'bridge_dual_agent.yaml')  # Combined bridge config
    gemini_agent_config = os.path.join(pkg_dir, 'config', 'gemini_live_agent.yaml')
    # Keep command agent config for future use (commented out)
    # cmd_agent_config = os.path.join(pkg_dir, 'config', 'gemini_command_agent.yaml')
    
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
    gemini_api_key_arg = DeclareLaunchArgument(
        'gemini_api_key',
        default_value=EnvironmentVariable('GEMINI_API_KEY', default_value=''),
        description='Gemini API key (defaults to GEMINI_API_KEY environment variable)'
    )
    
    agent_type_arg = DeclareLaunchArgument(
        'agent_type',
        default_value='multimodal',
        description='Agent type: multimodal|conversation|command|visual'
    )
    
    pause_timeout_arg = DeclareLaunchArgument(
        'pause_timeout',
        default_value='30.0',
        description='Session pause timeout in seconds'
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
    
    save_mic_arg = DeclareLaunchArgument(
        'save_mic',
        default_value='false',
        description='Save raw microphone input (post echo suppression) for debugging'
    )
    
    
    # Audio capture node
    audio_capturer = Node(
        package='audio_common',
        executable='audio_capturer_node',
        name='audio_capturer_node',
        output='screen',
        parameters=[{
            'chunk': 512,  # 32ms @ 16kHz - standard chunk size
            'rate': 16000,
            'device': -1  # Use default device
        }]
    )
    
    # Simple audio player for Gemini response playback
    audio_player = Node(
        package='by_your_command',
        executable='simple_audio_player',
        name='simple_audio_player',
        output='screen',
        parameters=[{
            'topic': 'response_voice',  # Relative topic for namespacing
            'sample_rate': 24000,  # Gemini outputs at 24kHz PCM16 (confirmed by tests)
            'channels': 1,
            'device': -1    # Default output device
        }]
    )
    
    
    # Silero VAD node for speech detection
    silero_vad = Node(
        package='by_your_command',
        executable='silero_vad_node',
        name='silero_vad_node',
        output='screen',
        parameters=[bridge_config],
        remappings=[
            ('audio', 'audio')  # Listen to raw audio directly (no AEC needed)
        ]
    )
    
    # ROS AI Bridge for data transport with WebSocket enabled
    ros_ai_bridge = Node(
        package='by_your_command',
        executable='ros_ai_bridge', 
        name='ros_ai_bridge',
        output='screen',
        parameters=[{
            'namespace': LaunchConfiguration('namespace'),
            'prefix': LaunchConfiguration('prefix'),
            'config_file': bridge_config,  # Use bridge config for topics
            'websocket_server.enabled': True,
            'websocket_server.host': '0.0.0.0',
            'websocket_server.port': 8765,
            'websocket_server.max_connections': 10  # Support multiple agents
        }],
        remappings=[
            ('camera/image_raw', '/grunt1/arm1/cam_live/color/image_raw')  # Map camera topic to actual source
        ]
    )
    
    # Gemini Live Agent (primary multimodal agent)
    gemini_agent = ExecuteProcess(
        cmd=[
            '/home/karim/ros2_ws/install/by_your_command/lib/by_your_command/gemini_live_agent',
            '--config', gemini_agent_config
            # Don't pass pause-timeout so it uses config file value
        ],
        output='screen',
        additional_env={
            'GEMINI_API_KEY': LaunchConfiguration('gemini_api_key'),
            'PAUSE_TIMEOUT': LaunchConfiguration('pause_timeout')
        }
    )
    
    # Command Extraction Agent (placeholder for future - commented out)
    # command_agent = ExecuteProcess(
    #     cmd=['python3', '-m', 'agents.gemini_live.gemini_live_agent'],
    #     cwd='/home/karim/ros2_ws/src/by_your_command',
    #     output='screen',
    #     additional_env={
    #         'GEMINI_API_KEY': LaunchConfiguration('gemini_api_key'),
    #         'PYTHONPATH': '/home/karim/ros2_ws/src/by_your_command',
    #         'CONFIG_FILE': cmd_agent_config,
    #         'AGENT_TYPE': 'command'
    #     }
    # )
    
    # Voice chunk recorder for assistant output (always enabled for debugging)
    voice_recorder_output = Node(
        package='by_your_command',
        executable='voice_chunk_recorder',
        name='voice_recorder_output',
        output='screen',
        parameters=[{
            'output_dir': '/tmp/prompt_voice/assistant_output',
            'input_mode': 'audio_data',
            'input_topic': 'response_voice',  # Assistant voice
            'input_sample_rate': 16000,  # Gemini outputs at 16kHz
            'audio_timeout': 10.0
        }]
    )
    
    # Recorder for raw microphone input (post echo suppression)
    voice_recorder_raw = Node(
        package='by_your_command',
        executable='voice_chunk_recorder',
        name='voice_recorder_raw',
        output='screen',
        parameters=[{
            'output_dir': '/tmp/prompt_voice/mic_raw',
            'input_mode': 'audio_stamped',
            'input_topic': 'audio_filtered',  # Post echo suppression
            'input_sample_rate': 16000,
            'audio_timeout': 10.0
        }],
        condition=IfCondition(LaunchConfiguration('save_mic'))
    )
    
    
    # Command processor node
    command_processor = Node(
        package='by_your_command',
        executable='command_processor',
        name='command_processor',
        output='screen',
        parameters=[{
            'command_transcript_topic': 'response_cmd',
            'arm_preset_topic': '/grunt1/arm_preset',  # Absolute path outside namespace
            'behavior_command_topic': '/grunt1/behavior_command'  # Absolute path outside namespace
        }]
    )
    
    # Command transcript monitor (optional debug tool)
    command_monitor = LogInfo(
        msg=['Command processor will route commands from /response_cmd to robot subsystems']
    )
    
    # Startup message
    startup_message = LogInfo(
        msg=[
            '🚀 Starting Gemini Live Agent System\n',
            '🤖 Agent Type: ', LaunchConfiguration('agent_type'), '\n',
            '🎙️  Audio: 16kHz input/output\n',
            '📷 Vision: Ready for camera input\n',
            '⏱️  Timeout: ', LaunchConfiguration('pause_timeout'), 's\n',
            '🔊 Listening for multimodal input...'
        ]
    )
    
    # Group all ROS nodes with namespace handling
    nodes_group = GroupAction([
        PushRosNamespace(LaunchConfiguration('namespace')),
        PushRosNamespace(LaunchConfiguration('prefix')),
        audio_capturer,
        audio_player,
        silero_vad,
        ros_ai_bridge,
        command_processor,
        voice_recorder_output,
        voice_recorder_raw
    ])
    
    # Agents run as separate processes and handle namespace differently
    agents_group = GroupAction([
        gemini_agent,
        # command_agent  # Uncomment when ready for dual agent
    ])
    
    return LaunchDescription([
        # Namespace arguments
        namespace_arg,
        prefix_arg,
        
        # Launch arguments
        gemini_api_key_arg,
        agent_type_arg,
        pause_timeout_arg,
        verbose_arg,
        enable_voice_recorder_arg,
        save_mic_arg,
        
        # Startup message
        startup_message,
        command_monitor,
        
        # Nodes and agents
        nodes_group,
        agents_group
    ])