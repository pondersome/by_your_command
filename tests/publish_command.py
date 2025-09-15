#!/usr/bin/env python3
"""
Simple command publisher for testing the command processor.

Usage:
    ros2 run by_your_command publish_command "lookup"
    ros2 run by_your_command publish_command "move@forward"
    ros2 run by_your_command publish_command "tenhut@rightish"
"""

import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def main(args=None):
    if len(sys.argv) < 2:
        print("Usage: ros2 run by_your_command publish_command <command>")
        print("Examples:")
        print("  ros2 run by_your_command publish_command lookup")
        print("  ros2 run by_your_command publish_command tenhut@rightish")
        print("  ros2 run by_your_command publish_command move@forward")
        return
    
    command = sys.argv[1]
    
    rclpy.init(args=args)
    node = Node('command_publisher')
    
    pub = node.create_publisher(String, 'response_cmd', 10)
    
    # Wait for publisher to connect
    import time
    time.sleep(0.5)
    
    msg = String()
    msg.data = command
    pub.publish(msg)
    
    node.get_logger().info(f"Published command: '{command}'")
    
    # Give time for message to be sent
    time.sleep(0.5)
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()