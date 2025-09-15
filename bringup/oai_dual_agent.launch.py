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
    # Both agents use their specific configs with different agent_ids and topics
    conv_agent_config = os.path.join(pkg_dir, 'config', 'oai_realtime_agent.yaml')
    cmd_agent_config = os.path.join(pkg_dir, 'config', 'oai_command_agent.yaml')
    
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
    openai_api_key_arg = DeclareLaunchArgument(
        'openai_api_key',
        default_value=EnvironmentVariable('OPENAI_API_KEY', default_value=''),
        description='OpenAI API key (defaults to OPENAI_API_KEY environment variable)'
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
    
    # Simple audio player for OpenAI response playback
    audio_player = Node(
        package='by_your_command',
        executable='simple_audio_player',
        name='simple_audio_player',
        output='screen',
        parameters=[{
            'topic': 'response_voice',  # Relative topic for namespacing
            'sample_rate': 16000,  # Standardized from 24kHz
            'channels': 1,
            'device': -1    # Default output device
        }]
    )
    
    # Clap detector node for wake-up detection
    clap_detector = Node(
        package='by_your_command',
        executable='clap_detector_node',
        name='clap_detector_node',
        output='screen',
        parameters=[{
            'audio_topic': 'audio',
            'wake_cmd_topic': 'wake_cmd',
            'enabled': True
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
            'config_file': bridge_config,  # Use combined bridge config for dual agents
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
            'OPENAI_API_KEY': LaunchConfiguration('openai_api_key'),
            'OPENAI_MODEL': LaunchConfiguration('conv_model'),
            # Voice is configured in the YAML file, not via env var
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
            'OPENAI_API_KEY': LaunchConfiguration('openai_api_key'),
            'OPENAI_MODEL': LaunchConfiguration('cmd_model'),
            # Voice is configured in the YAML file (or defaults to alloy)
            'PAUSE_TIMEOUT': LaunchConfiguration('pause_timeout')
        }
    )
    
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
            'input_sample_rate': 16000,  # Standardized from 24kHz
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
        msg=['Command extractor will publish to /response_cmd and /command_detected']
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
            '    Topics: /response_cmd, /command_detected\n',
            '⏱️  Pause timeout: ', LaunchConfiguration('pause_timeout'), 's\n',
            '🔊 Both agents listening for speech...'
        ]
    )
    
    # Group all ROS nodes with namespace handling
    nodes_group = GroupAction([
        PushRosNamespace(LaunchConfiguration('namespace')),
        PushRosNamespace(LaunchConfiguration('prefix')),
        audio_capturer,
        audio_player,
        clap_detector,
        silero_vad,
        ros_ai_bridge,
        command_processor,
        voice_recorder_output,
        voice_recorder_raw
    ])
    
    # Agents run as separate processes and handle namespace differently
    agents_group = GroupAction([
        conversational_agent,
        command_agent
    ])
    
    return LaunchDescription([
        # Namespace arguments
        namespace_arg,
        prefix_arg,
        
        # Launch arguments
        openai_api_key_arg,
        pause_timeout_arg,
        conv_model_arg,
        cmd_model_arg,
        voice_arg,
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