#!/usr/bin/env python3
"""
Test the Gemini agent with test image
Sends a text message to trigger image+text processing
"""

import asyncio
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class GeminiImageTester(Node):
    def __init__(self):
        super().__init__('gemini_image_tester')
        
        # Publisher for text input
        self.text_pub = self.create_publisher(String, 'prompt_text', 10)
        
        # Subscriber for LLM transcript
        self.transcript_sub = self.create_subscription(
            String, 'response_text', 
            self.transcript_callback, 10
        )
        
        self.get_logger().info("Gemini Image Tester initialized")
        self.responses = []
        
    def transcript_callback(self, msg):
        """Capture assistant responses"""
        self.get_logger().info(f"📥 Response: {msg.data[:200]}")
        self.responses.append(msg.data)
        
    def send_text(self, text):
        """Send text to trigger image processing"""
        msg = String()
        msg.data = text
        self.text_pub.publish(msg)
        self.get_logger().info(f"📤 Sent: {text}")

def main():
    rclpy.init()
    node = GeminiImageTester()
    
    # Wait for connections
    time.sleep(2)
    
    # Send test message to trigger image+text
    node.send_text("What objects do you see?")
    
    # Spin for responses
    start_time = time.time()
    while time.time() - start_time < 15:  # Wait up to 15 seconds
        rclpy.spin_once(node, timeout_sec=0.1)
        
        # Check if we got visual response
        for response in node.responses:
            if any(word in response.lower() for word in ['sword', 'coin', 'hammer', 'book', 'table', 'wooden', 'karate']):
                node.get_logger().warning("🎯 SUCCESS! Got visual description!")
                node.get_logger().info(f"Response mentions objects from tablethings.jpg")
                break
    
    # Check results
    if node.responses:
        node.get_logger().info(f"Got {len(node.responses)} responses")
        
        # Check for visual content
        visual_found = False
        for resp in node.responses:
            if any(word in resp.lower() for word in ['sword', 'coin', 'hammer', 'book', 'wooden']):
                visual_found = True
                break
        
        if visual_found:
            node.get_logger().warning("✅ TEST PASSED: Agent can see and describe images!")
        else:
            node.get_logger().error("❌ TEST FAILED: No visual description found")
            node.get_logger().info("Responses were generic/non-visual")
    else:
        node.get_logger().error("❌ No responses received")
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()