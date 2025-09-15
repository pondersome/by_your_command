#!/usr/bin/env python3
"""
Test script for VAD mute/unmute functionality
Tests the voice_active topic control on silero_vad_node
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from by_your_command.msg import AudioDataUtterance
import time

class VADMuteTest(Node):
    def __init__(self):
        super().__init__('vad_mute_test')
        
        # Publisher for voice_active control
        self.voice_active_pub = self.create_publisher(Bool, 'voice_active', 10)
        
        # Subscriber to monitor voice_chunks output
        self.voice_chunks_sub = self.create_subscription(
            AudioDataUtterance,
            'prompt_voice',
            self.voice_chunks_callback,
            10
        )
        
        self.chunks_received = 0
        self.test_phase = "initial"
        
        self.get_logger().info("VAD Mute Test Node Started")
        self.get_logger().info("This test will:")
        self.get_logger().info("1. Monitor voice_chunks for 5 seconds (should see chunks if audio is present)")
        self.get_logger().info("2. Send mute command and monitor for 5 seconds (should see no chunks)")
        self.get_logger().info("3. Send unmute command and monitor for 5 seconds (should see chunks again)")
        
        # Start test sequence
        self.timer = self.create_timer(1.0, self.test_timer_callback)
        self.test_start_time = time.time()
        
    def voice_chunks_callback(self, msg):
        """Monitor voice chunks to verify mute/unmute behavior"""
        self.chunks_received += 1
        if self.chunks_received % 10 == 0:  # Log every 10th chunk to avoid spam
            self.get_logger().info(f"[{self.test_phase}] Received voice chunk #{self.chunks_received}")
    
    def test_timer_callback(self):
        elapsed = time.time() - self.test_start_time
        
        if elapsed < 5.0:
            # Phase 1: Initial monitoring (should receive chunks if audio present)
            if self.test_phase != "initial":
                self.test_phase = "initial"
                self.get_logger().info("Phase 1: Monitoring initial voice chunks...")
        
        elif elapsed < 6.0:
            # Phase 2: Send mute command
            if self.test_phase == "initial":
                self.test_phase = "muting"
                self.get_logger().info("Phase 2: Sending MUTE command...")
                mute_msg = Bool()
                mute_msg.data = False
                self.voice_active_pub.publish(mute_msg)
                self.chunks_before_mute = self.chunks_received
        
        elif elapsed < 11.0:
            # Phase 3: Monitor during mute (should receive no chunks)
            if self.test_phase == "muting":
                self.test_phase = "muted"
                self.get_logger().info("Phase 3: Monitoring during mute (should see no chunks)...")
        
        elif elapsed < 12.0:
            # Phase 4: Send unmute command
            if self.test_phase == "muted":
                self.test_phase = "unmuting"
                self.get_logger().info("Phase 4: Sending UNMUTE command...")
                unmute_msg = Bool()
                unmute_msg.data = True
                self.voice_active_pub.publish(unmute_msg)
                self.chunks_during_mute = self.chunks_received - self.chunks_before_mute
        
        elif elapsed < 17.0:
            # Phase 5: Monitor after unmute (should receive chunks again)
            if self.test_phase == "unmuting":
                self.test_phase = "unmuted"
                self.get_logger().info("Phase 5: Monitoring after unmute (should see chunks again)...")
        
        else:
            # Test complete
            if self.test_phase == "unmuted":
                self.test_phase = "complete"
                self.chunks_after_unmute = self.chunks_received - self.chunks_before_mute - self.chunks_during_mute
                
                self.get_logger().info("=== TEST RESULTS ===")
                self.get_logger().info(f"Chunks before mute: {self.chunks_before_mute}")
                self.get_logger().info(f"Chunks during mute: {self.chunks_during_mute}")
                self.get_logger().info(f"Chunks after unmute: {self.chunks_after_unmute}")
                
                if self.chunks_during_mute == 0:
                    self.get_logger().info("✅ PASS: Mute functionality working correctly")
                else:
                    self.get_logger().error("❌ FAIL: Voice chunks still received during mute")
                
                self.get_logger().info("Test complete. Shutting down...")
                rclpy.shutdown()

def main():
    rclpy.init()
    
    test_node = VADMuteTest()
    
    try:
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        test_node.get_logger().info("Test interrupted by user")
    finally:
        test_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()