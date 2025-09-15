#!/usr/bin/env python3
"""
Test script for repeated clap wake cycles
Tests: sleep -> clap wake -> sleep -> clap wake (multiple cycles)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from by_your_command.msg import AudioDataUtterance
import time

class RepeatedClapWakeTest(Node):
    def __init__(self):
        super().__init__('repeated_clap_wake_test')
        
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
        self.voice_active_state = True
        self.test_cycle = 0
        self.max_cycles = 3
        
        self.get_logger().info("Repeated Clap Wake Test Started")
        self.get_logger().info(f"Will test {self.max_cycles} sleep/wake cycles")
        self.get_logger().info("Each cycle: sleep command -> manual wake (simulating clap)")
        
        # Start test sequence
        self.timer = self.create_timer(3.0, self.test_cycle_callback)
        self.cycle_start_time = time.time()
        self.phase = "initial"
        
    def voice_chunks_callback(self, msg):
        """Monitor voice chunks"""
        self.voice_chunks_count += 1
        if self.voice_chunks_count % 5 == 0:
            self.get_logger().info(f"Voice chunks: {self.voice_chunks_count} total")
    
    def voice_active_callback(self, msg):
        """Monitor voice_active state"""
        if msg.data != self.voice_active_state:
            self.voice_active_state = msg.data
            state_str = "ACTIVE" if self.voice_active_state else "MUTED"
            self.get_logger().info(f"🔄 Voice state: {state_str}")
    
    def test_cycle_callback(self):
        elapsed = time.time() - self.cycle_start_time
        
        if self.test_cycle >= self.max_cycles:
            self.get_logger().info("All test cycles completed!")
            self.get_logger().info(f"Final voice chunks count: {self.voice_chunks_count}")
            rclpy.shutdown()
            return
        
        if self.phase == "initial" and elapsed > 2:
            # Start new cycle - send sleep command
            self.test_cycle += 1
            self.phase = "sleeping"
            self.cycle_start_time = time.time()
            
            self.get_logger().info(f"=== CYCLE {self.test_cycle}/{self.max_cycles} ===")
            self.get_logger().info("Sending sleep command...")
            
            sleep_msg = String()
            sleep_msg.data = "sleep"
            self.command_pub.publish(sleep_msg)
            
        elif self.phase == "sleeping" and elapsed > 3:
            # Send wake signal (simulating clap detection)
            self.phase = "waking"
            self.get_logger().info("Sending wake signal (simulating clap)...")
            
            wake_msg = Bool()
            wake_msg.data = True
            self.voice_active_pub.publish(wake_msg)
            
        elif self.phase == "waking" and elapsed > 5:
            # Wait and prepare for next cycle
            self.phase = "initial"
            self.cycle_start_time = time.time()
            self.get_logger().info(f"Cycle {self.test_cycle} complete. Waiting for next cycle...")

def main():
    rclpy.init()
    
    test_node = RepeatedClapWakeTest()
    
    try:
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        test_node.get_logger().info("Test interrupted by user")
    finally:
        test_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()