#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class ConversationIDPublisher(Node):
    def __init__(self):
        super().__init__('test_conversation_id_publisher')
        self.publisher = self.create_publisher(String, '/conversation_id', 10)
        self.get_logger().info('Created publisher for /conversation_id')
        
    def publish_test_id(self):
        msg = String()
        msg.data = f"test_conv_{int(time.time())}"
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')

def main():
    rclpy.init()
    node = ConversationIDPublisher()
    
    # Publish a test ID
    node.publish_test_id()
    
    # Give time for the message to be sent
    rclpy.spin_once(node, timeout_sec=1.0)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()