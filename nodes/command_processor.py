#!/usr/bin/env python3
"""
Command Processor Node

Listens for response_cmd messages (was command_transcript) from the AI agents 
and processes them into actionable commands or behavior requests for various 
robot subsystems.

Currently supports:
- Arm preset commands (e.g., "lookup", "tenhut@rightish")
- Behavior commands (e.g., "stop", "move@forward", "turn@left", "follow@target")
- Compound commands with bearings

Author: Karim Virani
Version: 1.0
Date: January 2025
"""

import re
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from typing import Optional, Tuple


class CommandProcessor(Node):
    def __init__(self):
        super().__init__('command_processor')
        
        # Declare parameters for topic names - updated to new naming convention
        self.declare_parameter('command_transcript_topic', 'response_cmd')  # was command_transcript
        self.declare_parameter('arm_preset_topic', '/grunt1/arm_preset')  # Absolute path to arm_preset
        self.declare_parameter('behavior_command_topic', '/grunt1/behavior_command')  # Absolute path outside namespace
        
        # Get configured topic names
        command_topic = self.get_parameter('command_transcript_topic').value
        arm_topic = self.get_parameter('arm_preset_topic').value
        behavior_topic = self.get_parameter('behavior_command_topic').value
        
        # Create subscriber for command transcripts (relative - will be in node's namespace)
        self.command_sub = self.create_subscription(
            String,
            command_topic,
            self.command_callback,
            10
        )
        
        # Create publishers
        # arm_preset needs to be at /grunt1/arm_preset (absolute)
        self.arm_preset_pub = self.create_publisher(
            String,
            arm_topic,
            10
        )
        
        # behavior_command needs to be at /grunt1/behavior_command (absolute)
        self.behavior_command_pub = self.create_publisher(
            String,
            behavior_topic,
            10
        )
        
        # Publisher for wake_cmd control (sleep/wake commands)
        self.wake_cmd_pub = self.create_publisher(
            Bool,
            'wake_cmd',  # Relative topic name for namespacing
            10
        )
        
        # Define valid command categories
        self.arm_presets = {'bumper', 'tenhut', 'lookup', 'lookout', 'reach', 'pan'}  # pan controls arm joint
        self.behavior_commands = {'stop', 'follow', 'track', 'sleep', 'wake', 'move', 'turn'}
        self.bearing_presets = {
            'back-left', 'full-left', 'left', 'leftish', 'forward',
            'rightish', 'right', 'full-right', 'back-right', 'back'
        }
        
        self.get_logger().info(f"Command Processor initialized")
        self.get_logger().info(f"  Listening on: {command_topic}")
        self.get_logger().info(f"  Publishing arm presets to: {arm_topic}")
        self.get_logger().info(f"  Publishing behavior commands to: {behavior_topic}")
    
    def parse_command(self, command: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse a command string into its components.
        
        Returns:
            (command_type, base_command, modifier)
            - command_type: 'arm', 'behavior', or None
            - base_command: The primary command (e.g., 'lookup', 'move')
            - modifier: Optional modifier (e.g., 'rightish', 'forward', 'betty')
        """
        command = command.strip().lower()
        
        # Check for compound command with @ separator
        if '@' in command:
            base, modifier = command.split('@', 1)
            base = base.strip()
            modifier = modifier.strip()
        else:
            base = command
            modifier = None
        
        # Determine command type
        if base in self.arm_presets:
            # Pan command requires a modifier (bearing)
            if base == 'pan' and not modifier:
                self.get_logger().warn(f"Pan command requires a bearing modifier (e.g., pan@left)")
                return None, None, None
            return 'arm', base, modifier
        elif base in self.behavior_commands:
            # Some behavior commands require modifiers
            if base in {'follow', 'track', 'move', 'turn'} and not modifier:
                self.get_logger().warn(f"Behavior command '{base}' requires a modifier (e.g., {base}@forward)")
                return None, None, None
            return 'behavior', base, modifier
        elif base in self.bearing_presets:
            # Standalone bearing is interpreted as pan@bearing (arm command)
            return 'arm', 'pan', base
        else:
            self.get_logger().warn(f"Unrecognized command: '{command}'")
            return None, None, None
    
    def validate_modifier(self, command_type: str, base_command: str, modifier: str) -> bool:
        """Validate that a modifier is appropriate for the command."""
        if not modifier:
            return True
            
        if command_type == 'arm':
            # Arm commands can have bearing modifiers
            return modifier in self.bearing_presets or self._is_valid_angle(modifier)
        elif command_type == 'behavior':
            if base_command in {'move', 'turn'}:
                # These require bearing modifiers
                return modifier in self.bearing_presets or self._is_valid_angle(modifier)
            elif base_command in {'follow', 'track'}:
                # These can have any object label as modifier
                return True
        
        return False
    
    def _is_valid_angle(self, value: str) -> bool:
        """Check if a string represents a valid angle value."""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def command_callback(self, msg: String):
        """Process incoming command transcript messages."""
        command = msg.data.strip()
        
        if not command:
            return
            
        self.get_logger().info(f"Received command: '{command}'")
        
        # Parse the command
        command_type, base_command, modifier = self.parse_command(command)
        
        if not command_type:
            return
        
        # Validate modifier if present
        if modifier and not self.validate_modifier(command_type, base_command, modifier):
            self.get_logger().warn(f"Invalid modifier '{modifier}' for command '{base_command}'")
            return
        
        # Route to appropriate handler
        if command_type == 'arm':
            self.handle_arm_command(base_command, modifier)
        elif command_type == 'behavior':
            self.handle_behavior_command(base_command, modifier)
    
    def handle_arm_command(self, preset: str, bearing: Optional[str]):
        """Publish arm preset commands."""
        if bearing:
            command = f"{preset}@{bearing}"
        else:
            command = preset
            
        msg = String()
        msg.data = command
        self.arm_preset_pub.publish(msg)
        self.get_logger().info(f"Published arm preset: '{command}'")
    
    def handle_behavior_command(self, command: str, modifier: Optional[str]):
        """Publish behavior commands."""
        # Handle special sleep/wake commands for VAD control
        if command == 'sleep':
            # Send wake_cmd false to put VAD to sleep
            wake_msg = Bool()
            wake_msg.data = False
            self.wake_cmd_pub.publish(wake_msg)
            self.get_logger().info("Sleep command - putting voice detection to sleep")
        elif command == 'wake':
            # Send wake_cmd true to wake up VAD
            wake_msg = Bool()
            wake_msg.data = True
            self.wake_cmd_pub.publish(wake_msg)
            self.get_logger().info("Wake command - waking up voice detection")
        
        # Also publish the behavior command normally for other systems
        if modifier:
            full_command = f"{command}@{modifier}"
        else:
            full_command = command
            
        msg = String()
        msg.data = full_command
        self.behavior_command_pub.publish(msg)
        self.get_logger().info(f"Published behavior command: '{full_command}'")


def main(args=None):
    rclpy.init(args=args)
    node = CommandProcessor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()