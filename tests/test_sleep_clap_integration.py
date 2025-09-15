#!/usr/bin/env python3
"""
Test script for sleep command and clap detection integration
Tests the complete flow: sleep command -> mute -> clap detection -> wake
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from by_your_command.msg import AudioDataUtterance
import time
import numpy as np

class SleepClapIntegrationTest(Node):
    def __init__(self):
        super().__init__('sleep_clap_integration_test')
        
        # Publishers
        self.command_pub = self.create_publisher(String, 'response_cmd', 10)
        self.voice_active_pub = self.create_publisher(Bool, 'voice_active', 10)
        
        # Subscribers to monitor system behavior
        self.voice_chunks_sub = self.create_subscription(
            AudioDataUtterance,
            'prompt_voice',
            self.voice_chunks_callback,
            10
        )
        
        self.voice_active_sub = self.create_subscription(
            Bool,
            'voice_active',
            self.voice_active_callback,
            10
        )
        
        # Test state
        self.voice_chunks_count = 0
        self.voice_active_state = True  # Assume starts active
        self.test_phase = "initial"
        self.test_start_time = time.time()
        
        self.get_logger().info("Sleep & Clap Integration Test Node Started")
        self.get_logger().info("This test will:")
        self.get_logger().info("1. Send 'sleep' command")
        self.get_logger().info("2. Verify voice processing stops")
        self.get_logger().info("3. Send manual unmute to simulate clap detection")
        self.get_logger().info("4. Verify voice processing resumes")
        
        # Start test sequence
        self.timer = self.create_timer(2.0, self.test_timer_callback)
        
    def voice_chunks_callback(self, msg):
        """Monitor voice chunks to verify mute/unmute behavior"""
        self.voice_chunks_count += 1
        self.get_logger().info(f"[{self.test_phase}] Voice chunk #{self.voice_chunks_count} received")
    
    def voice_active_callback(self, msg):
        """Monitor voice_active state changes"""
        self.voice_active_state = msg.data
        state_str = "ACTIVE" if self.voice_active_state else "MUTED"
        self.get_logger().info(f"[{self.test_phase}] Voice state: {state_str}")
    
    def test_timer_callback(self):
        elapsed = time.time() - self.test_start_time
        
        if elapsed < 5.0:
            # Phase 1: Initial monitoring
            if self.test_phase != "initial":
                self.test_phase = "initial"
                self.get_logger().info("Phase 1: Initial monitoring...")
        
        elif elapsed < 7.0:
            # Phase 2: Send sleep command
            if self.test_phase == "initial":
                self.test_phase = "sending_sleep"
                self.get_logger().info("Phase 2: Sending 'sleep' command...")
                
                sleep_msg = String()
                sleep_msg.data = "sleep"
                self.command_pub.publish(sleep_msg)
                
                self.chunks_before_sleep = self.voice_chunks_count
        
        elif elapsed < 12.0:
            # Phase 3: Monitor during sleep (should see no new chunks)
            if self.test_phase == "sending_sleep":
                self.test_phase = "monitoring_sleep"
                self.get_logger().info("Phase 3: Monitoring during sleep (should see no voice chunks)...")
        
        elif elapsed < 14.0:
            # Phase 4: Simulate clap detection wake-up
            if self.test_phase == "monitoring_sleep":
                self.test_phase = "simulating_clap"
                self.get_logger().info("Phase 4: Simulating clap detection wake-up...")
                
                # Send voice_active true to simulate clap detection
                wake_msg = Bool()
                wake_msg.data = True
                self.voice_active_pub.publish(wake_msg)
                
                self.chunks_during_sleep = self.voice_chunks_count - self.chunks_before_sleep
        
        elif elapsed < 19.0:
            # Phase 5: Monitor after wake (should see chunks again)
            if self.test_phase == "simulating_clap":
                self.test_phase = "monitoring_wake"
                self.get_logger().info("Phase 5: Monitoring after wake (should see voice chunks again)...")
        
        else:
            # Test complete
            if self.test_phase == "monitoring_wake":
                self.test_phase = "complete"
                self.chunks_after_wake = self.voice_chunks_count - self.chunks_before_sleep - self.chunks_during_sleep
                
                self.get_logger().info("=== TEST RESULTS ===")
                self.get_logger().info(f"Voice chunks before sleep: {self.chunks_before_sleep}")
                self.get_logger().info(f"Voice chunks during sleep: {self.chunks_during_sleep}")
                self.get_logger().info(f"Voice chunks after wake: {self.chunks_after_wake}")
                self.get_logger().info(f"Current voice state: {'ACTIVE' if self.voice_active_state else 'MUTED'}")
                
                # Evaluate test results
                success = True
                if self.chunks_during_sleep > 0:
                    self.get_logger().error("❌ FAIL: Voice chunks received during sleep")
                    success = False
                
                if not self.voice_active_state:
                    self.get_logger().error("❌ FAIL: Voice still muted after wake command")
                    success = False
                
                if success:
                    self.get_logger().info("✅ PASS: Sleep command and wake integration working correctly")
                
                self.get_logger().info("Test complete. Shutting down...")
                rclpy.shutdown()

def main():
    rclpy.init()
    
    test_node = SleepClapIntegrationTest()
    
    try:
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        test_node.get_logger().info("Test interrupted by user")
    finally:
        test_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()