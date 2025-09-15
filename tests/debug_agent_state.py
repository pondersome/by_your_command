#!/usr/bin/env python3
"""
Debug script for agent modal state issues
Monitors agent state and provides manual reset capability
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import time

class AgentStateDebugger(Node):
    def __init__(self):
        super().__init__('agent_state_debugger')
        
        # Publishers to reset agent state
        self.conversation_id_pub = self.create_publisher(String, 'conversation_id', 10)
        
        # Monitor transcripts to see if agents are responding
        self.llm_transcript_sub = self.create_subscription(
            String,
            'response_text',
            self.llm_transcript_callback,
            10
        )
        
        self.command_transcript_sub = self.create_subscription(
            String,
            'response_cmd', 
            self.command_transcript_callback,
            10
        )
        
        self.last_conv_response = time.time()
        self.last_cmd_response = time.time()
        
        self.get_logger().info("Agent State Debugger Started")
        self.get_logger().info("Monitoring agent responsiveness...")
        self.get_logger().info("Commands:")
        self.get_logger().info("  - Type 'reset' to send conversation reset signal")
        self.get_logger().info("  - Ctrl+C to exit")
        
        # Timer for periodic status
        self.timer = self.create_timer(10.0, self.status_callback)
        
        # Start input thread for commands
        import threading
        self.input_thread = threading.Thread(target=self.input_loop, daemon=True)
        self.input_thread.start()
        
    def llm_transcript_callback(self, msg):
        """Monitor conversational agent responses"""
        self.last_conv_response = time.time()
        self.get_logger().info(f"🗣️  Conv agent: {msg.data[:50]}...")
        
    def command_transcript_callback(self, msg):
        """Monitor command agent responses"""
        self.last_cmd_response = time.time()
        self.get_logger().info(f"⚡ Cmd agent: {msg.data}")
        
    def status_callback(self):
        """Check agent responsiveness"""
        now = time.time()
        conv_age = now - self.last_conv_response
        cmd_age = now - self.last_cmd_response
        
        self.get_logger().info(f"=== AGENT STATUS ===")
        self.get_logger().info(f"Conv agent: {conv_age:.1f}s since last response")
        self.get_logger().info(f"Cmd agent: {cmd_age:.1f}s since last response")
        
        if conv_age > 60 or cmd_age > 60:
            self.get_logger().warning("⚠️  Agent(s) may be stuck! Consider 'reset' command")
            
    def input_loop(self):
        """Handle user input for manual commands"""
        while rclpy.ok():
            try:
                cmd = input().strip().lower()
                if cmd == 'reset':
                    self.reset_conversation()
                elif cmd == 'quit' or cmd == 'exit':
                    rclpy.shutdown()
                    break
            except (EOFError, KeyboardInterrupt):
                break
                
    def reset_conversation(self):
        """Send conversation reset signal to agents"""
        reset_msg = String()
        reset_msg.data = f"reset_{int(time.time())}"
        self.conversation_id_pub.publish(reset_msg)
        self.get_logger().info("🔄 Sent conversation reset signal to agents")

def main():
    rclpy.init()
    
    debugger = AgentStateDebugger()
    
    try:
        rclpy.spin(debugger)
    except KeyboardInterrupt:
        debugger.get_logger().info("Agent state debugger terminated")
    finally:
        debugger.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()