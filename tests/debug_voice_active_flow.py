#!/usr/bin/env python3
"""
Debug script to monitor voice_active topic flow
Helps identify feedback loops or unexpected publishers
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from by_your_command.msg import AudioDataUtterance
import time

class VoiceActiveDebugger(Node):
    def __init__(self):
        super().__init__('voice_active_debugger')
        
        # Monitor voice_active topic
        self.voice_active_sub = self.create_subscription(
            Bool,
            'voice_active',
            self.voice_active_callback,
            10
        )
        
        # Monitor voice_activity topic (VAD output)
        self.voice_activity_sub = self.create_subscription(
            Bool,
            'voice_activity',
            self.voice_activity_callback,
            10
        )
        
        # Monitor voice_chunks
        self.voice_chunks_sub = self.create_subscription(
            AudioDataUtterance,
            'prompt_voice',
            self.voice_chunks_callback,
            10
        )
        
        self.voice_active_count = 0
        self.voice_activity_count = 0
        self.voice_chunks_count = 0
        
        self.get_logger().info("Voice Active Flow Debugger Started")
        self.get_logger().info("Monitoring:")
        self.get_logger().info("  - voice_active (mute/unmute control)")
        self.get_logger().info("  - voice_activity (VAD detection)")
        self.get_logger().info("  - voice_chunks (processed audio)")
        
        # Timer for periodic status
        self.timer = self.create_timer(5.0, self.status_callback)
        self.start_time = time.time()
        
    def voice_active_callback(self, msg):
        """Monitor voice_active messages (mute/unmute control)"""
        self.voice_active_count += 1
        state_str = "ACTIVE" if msg.data else "MUTED"
        self.get_logger().info(f"🔧 voice_active #{self.voice_active_count}: {state_str}")
        
    def voice_activity_callback(self, msg):
        """Monitor voice_activity messages (VAD detection)"""
        self.voice_activity_count += 1
        detection_str = "SPEECH" if msg.data else "SILENCE"
        self.get_logger().info(f"👂 voice_activity #{self.voice_activity_count}: {detection_str}")
        
    def voice_chunks_callback(self, msg):
        """Monitor voice_chunks (processed audio)"""
        self.voice_chunks_count += 1
        if self.voice_chunks_count % 10 == 0:  # Log every 10th chunk
            self.get_logger().info(f"🎵 voice_chunks: {self.voice_chunks_count} total chunks received")
    
    def status_callback(self):
        """Periodic status update"""
        elapsed = time.time() - self.start_time
        self.get_logger().info(f"=== STATUS (t={elapsed:.1f}s) ===")
        self.get_logger().info(f"voice_active messages: {self.voice_active_count}")
        self.get_logger().info(f"voice_activity messages: {self.voice_activity_count}")
        self.get_logger().info(f"voice_chunks messages: {self.voice_chunks_count}")
        self.get_logger().info("========================")

def main():
    rclpy.init()
    
    debugger = VoiceActiveDebugger()
    
    try:
        rclpy.spin(debugger)
    except KeyboardInterrupt:
        debugger.get_logger().info("Debug session ended by user")
    finally:
        debugger.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()