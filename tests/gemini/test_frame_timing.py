#!/usr/bin/env python3
"""Test script to verify smart frame forwarding timing"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from by_your_command.msg import AudioDataUtterance
import time
import numpy as np

class FrameTimingTester(Node):
    def __init__(self):
        super().__init__('frame_timing_tester')
        
        # Publishers
        self.image_pub = self.create_publisher(
            CompressedImage,
            '/grunt1/arm1/cam_live/color/image_raw/compressed',
            10
        )
        self.voice_pub = self.create_publisher(
            AudioDataUtterance,
            'prompt_voice',
            10
        )
        self.text_pub = self.create_publisher(
            String,
            'prompt_text',
            10
        )
        
        # Stats tracking
        self.frame_counter = 0
        self.last_frame_time = 0
        
        self.get_logger().info("Frame timing tester started")
        
    def send_test_frame(self, frame_id: int):
        """Send a test image frame with ID"""
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        # Create fake image data with frame ID encoded
        msg.data = bytes(f"FRAME_{frame_id:04d}", 'utf-8')
        
        self.image_pub.publish(msg)
        self.frame_counter += 1
        current_time = time.time()
        
        if self.last_frame_time > 0:
            fps = 1.0 / (current_time - self.last_frame_time)
            self.get_logger().info(f"📸 Sent frame {frame_id} (fps: {fps:.1f})")
        else:
            self.get_logger().info(f"📸 Sent frame {frame_id}")
            
        self.last_frame_time = current_time
        
    def send_voice_chunk(self, chunk_id: int):
        """Send a voice chunk"""
        msg = AudioDataUtterance()
        msg.chunk_sequence = chunk_id
        msg.is_utterance_start = (chunk_id == 0)
        msg.is_utterance_end = False
        # Fake audio data
        msg.data = np.zeros(1600, dtype=np.int16).tobytes()  # 100ms at 16kHz
        
        self.voice_pub.publish(msg)
        self.get_logger().info(f"🎤 Sent voice chunk {chunk_id}")
        
    def send_text_input(self, text: str):
        """Send text input"""
        msg = String()
        msg.data = text
        self.text_pub.publish(msg)
        self.get_logger().info(f"📝 Sent text: {text}")
        
    def run_test_sequence(self):
        """Run a test sequence to verify frame forwarding"""
        self.get_logger().info("=" * 50)
        self.get_logger().info("Starting frame timing test sequence")
        self.get_logger().info("=" * 50)
        
        # Test 1: Send frames at 5 fps for 2 seconds (baseline)
        self.get_logger().info("\nTest 1: Sending frames at 5 fps (no triggers)")
        for i in range(10):
            self.send_test_frame(i)
            time.sleep(0.2)  # 5 fps
            
        time.sleep(1.0)
        
        # Test 2: Send voice trigger after frames
        self.get_logger().info("\nTest 2: Voice trigger - should forward latest frame")
        self.send_test_frame(100)
        time.sleep(0.1)
        self.send_test_frame(101)
        time.sleep(0.1)
        self.send_voice_chunk(0)  # First chunk - should trigger
        
        time.sleep(1.0)
        
        # Test 3: Continuous voice with frames
        self.get_logger().info("\nTest 3: Continuous voice - should forward every 5th chunk")
        for i in range(15):
            if i % 3 == 0:
                self.send_test_frame(200 + i)
            self.send_voice_chunk(i)
            time.sleep(0.1)
            
        time.sleep(1.0)
        
        # Test 4: Text trigger
        self.get_logger().info("\nTest 4: Text trigger - should forward latest frame")
        self.send_test_frame(300)
        time.sleep(0.1)
        self.send_text_input("What do you see?")
        
        time.sleep(1.0)
        
        # Test 5: Stale frame (> 1 second old)
        self.get_logger().info("\nTest 5: Stale frame test")
        self.send_test_frame(400)
        self.get_logger().info("Waiting 1.5 seconds for frame to become stale...")
        time.sleep(1.5)
        self.send_voice_chunk(0)  # Should not forward stale frame
        
        self.get_logger().info("\n" + "=" * 50)
        self.get_logger().info("Test sequence complete!")
        self.get_logger().info("Check bridge logs for frame forwarding events")
        self.get_logger().info("=" * 50)

def main():
    rclpy.init()
    
    tester = FrameTimingTester()
    
    # Give time for everything to initialize
    time.sleep(2.0)
    
    # Run test
    tester.run_test_sequence()
    
    # Cleanup
    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()