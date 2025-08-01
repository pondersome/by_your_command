#!/usr/bin/env python3
"""
AEC Test Launch File

Tests acoustic echo cancellation with generated tones to:
1. Measure actual system delay
2. Verify echo cancellation effectiveness

Author: Assistant
Date: July 2025
"""

import os
from launch import LaunchDescription
from launch.actions import GroupAction, DeclareLaunchArgument
from launch_ros.actions import Node, PushRosNamespace
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


def generate_launch_description():
    
    # Launch arguments
    aec_method = LaunchConfiguration('aec_method')
    
    # Audio capture node
    audio_capturer = Node(
        package='audio_common',
        executable='audio_capturer_node',
        name='audio_capturer_node',
        output='screen',
        parameters=[{
            'device': 14,  # Use pulse audio for automatic resampling
            'rate': 16000,
            'chunk': 480  # 30ms @ 16kHz = exactly 3 WebRTC frames (160 samples each)
        }]
    )
    
    # Simple audio player for test tone playback
    audio_player = Node(
        package='by_your_command',
        executable='simple_audio_player',
        name='simple_audio_player',
        output='screen',
        parameters=[{
            'topic': 'test_tone_out',  # Subscribe to test tone topic
            'sample_rate': 24000,
            'channels': 1,
            'device': 14  # Use pulse audio device
        }]
    )
    
    # AEC node with test mode enabled
    aec_node = Node(
        package='by_your_command',
        executable='aec_node',
        name='aec_node',
        output='screen',
        parameters=[{
            'mic_sample_rate': 16000,
            'speaker_sample_rate': 24000,
            'aec_method': aec_method,  # Use launch parameter
            'debug_logging': True,
            'bypass': False,
            'test_mode': True,  # Enable tone generation
            'frame_size': 160,  # 10ms at 16kHz for Speex
            'filter_length': 3200  # 200ms at 16kHz
        }],
        remappings=[
            ('audio', 'audio'),  # Raw mic input
            ('audio_out', 'test_tone_out'),  # Remap to test tone topic
            ('audio_filtered', 'audio_filtered')  # Echo-cancelled output
        ]
    )
    
    # Voice recorders for analysis
    voice_recorder_tone = Node(
        package='by_your_command',
        executable='voice_chunk_recorder',
        name='voice_recorder_tone',
        output='screen',
        parameters=[{
            'output_dir': '/tmp/voice_chunks/test_tone',
            'input_mode': 'audio_data',
            'input_topic': 'test_tone_out',  # Record generated tones
            'input_sample_rate': 24000,
            'audio_timeout': 10.0
        }]
    )
    
    voice_recorder_raw = Node(
        package='by_your_command',
        executable='voice_chunk_recorder',
        name='voice_recorder_raw',
        output='screen',
        parameters=[{
            'output_dir': '/tmp/voice_chunks/test_raw',
            'input_mode': 'audio_stamped',
            'input_topic': 'audio',  # Raw microphone
            'input_sample_rate': 16000,
            'audio_timeout': 10.0
        }]
    )
    
    voice_recorder_filtered = Node(
        package='by_your_command',
        executable='voice_chunk_recorder',
        name='voice_recorder_filtered',
        output='screen',
        parameters=[{
            'output_dir': '/tmp/voice_chunks/test_filtered',
            'input_mode': 'audio_stamped',
            'input_topic': 'audio_filtered',  # After AEC
            'input_sample_rate': 16000,
            'audio_timeout': 10.0
        }]
    )
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'aec_method',
            default_value='webrtc',
            description='AEC method to use: webrtc or speex'
        ),
        audio_capturer,
        audio_player,
        aec_node,
        voice_recorder_tone,
        voice_recorder_raw,
        voice_recorder_filtered
    ])