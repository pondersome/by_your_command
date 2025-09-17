"""
Test launch file for WebSocket reflection proxy.

This minimal launch file starts:
1. WebSocket reflection proxy
2. ROS AI Bridge
3. Simple test to verify proxy operation

Use this to test proxy functionality before full dual-agent deployment.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate test launch description"""

    # Get package paths
    pkg_dir = get_package_share_directory('by_your_command')
    config_file = os.path.join(pkg_dir, 'config', 'bridge_dual_agent.yaml')

    # Declare launch arguments
    proxy_port = DeclareLaunchArgument(
        'proxy_port',
        default_value='8766',
        description='Port for reflection proxy'
    )

    bridge_port = DeclareLaunchArgument(
        'bridge_port',
        default_value='8765',
        description='Port for ROS AI bridge'
    )

    # 1. WebSocket Reflection Proxy
    reflection_proxy = ExecuteProcess(
        cmd=['ros2', 'run', 'by_your_command', 'websocket_reflection_proxy'],
        name='websocket_reflection_proxy',
        output='screen'
    )

    # 2. ROS AI Bridge (using actual config file with full path)
    ros_ai_bridge = Node(
        package='by_your_command',
        executable='ros_ai_bridge',
        name='ros_ai_bridge',
        output='screen',
        parameters=[{
            'config_file': config_file
        }]
    )

    # 3. Test message publisher (delayed to allow startup)
    test_publisher = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'topic', 'pub', '--once',
                    '/prompt_text', 'std_msgs/String',
                    '{data: "Test message through proxy"}'
                ],
                output='screen'
            ),
            LogInfo(msg='📤 Published test message to /prompt_text')
        ]
    )

    # Info messages
    startup_info = LogInfo(msg=[
        '\n🧪 Testing WebSocket Reflection Proxy\n',
        '   Proxy port: ', LaunchConfiguration('proxy_port'), '\n',
        '   Bridge port: ', LaunchConfiguration('bridge_port'), '\n',
        '   \n',
        '   To test manually:\n',
        '   1. Check proxy is running on port 8766\n',
        '   2. Check bridge is running on port 8765\n',
        '   3. Connect test agents to port 8766\n',
        '   4. Verify cross-agent reflection works\n'
    ])

    return LaunchDescription([
        # Arguments
        proxy_port,
        bridge_port,

        # Components
        startup_info,
        reflection_proxy,
        ros_ai_bridge,
        test_publisher,

        # Status check
        TimerAction(
            period=3.0,
            actions=[
                LogInfo(msg='✅ Proxy and bridge should be running. Check logs above for any errors.')
            ]
        )
    ])