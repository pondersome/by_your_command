#!/usr/bin/env python3
"""Test external conversation reset by publishing to /conversation_id"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from datetime import datetime
import time

class ConversationResetPublisher(Node):
    def __init__(self):
        super().__init__('test_conversation_reset')
        self.publisher = self.create_publisher(String, '/conversation_id', 10)
        self.get_logger().info('Created publisher for /conversation_id')
        
    def publish_reset(self):
        """Publish a new conversation ID to trigger external reset"""
        msg = String()
        # Generate new conversation ID with "external_" prefix to distinguish it
        msg.data = f"conv_external_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.publisher.publish(msg)
        self.get_logger().info(f'Published external reset: {msg.data}')

def main():
    rclpy.init()
    node = ConversationResetPublisher()
    
    # Wait a moment for setup
    time.sleep(1)
    
    # Publish external reset
    node.publish_reset()
    
    # Give time for the message to be sent
    rclpy.spin_once(node, timeout_sec=1.0)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()