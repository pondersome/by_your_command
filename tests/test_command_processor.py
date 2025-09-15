#!/usr/bin/env python3
"""
Test script for command processor node.

Publishes test commands to verify parsing and routing.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time


class CommandProcessorTester(Node):
    def __init__(self):
        super().__init__('command_processor_tester')
        
        # Publisher for command transcripts
        self.command_pub = self.create_publisher(String, 'response_cmd', 10)
        
        # Subscribers to monitor outputs
        self.arm_sub = self.create_subscription(
            String, '/grunt1/arm_preset', self.arm_callback, 10
        )
        self.behavior_sub = self.create_subscription(
            String, '/grunt1/behavior_command', self.behavior_callback, 10
        )
        
        # Give time for connections
        time.sleep(1.0)
        
        # Test commands
        self.test_commands = [
            # Arm presets
            ("lookup", "arm_preset", "lookup"),
            ("tenhut", "arm_preset", "tenhut"),
            ("tenhut@rightish", "arm_preset", "tenhut@rightish"),
            ("bumper@forward", "arm_preset", "bumper@forward"),
            
            # Behavior commands
            ("stop", "behavior_command", "stop"),
            ("move@forward", "behavior_command", "move@forward"),
            ("turn@left", "behavior_command", "turn@left"),
            ("follow@betty", "behavior_command", "follow@betty"),
            ("track@banana", "behavior_command", "track@banana"),
            
            # Pan commands (go to arm_preset)
            ("pan@full-right", "arm_preset", "pan@full-right"),
            
            # Standalone bearing (should become pan@bearing in arm_preset)
            ("back", "arm_preset", "pan@back"),
            ("leftish", "arm_preset", "pan@leftish"),
            
            # Invalid commands (should be ignored)
            ("invalid_command", None, None),
            ("move", None, None),  # Missing required modifier
            ("follow", None, None),  # Missing required modifier
            ("pan", None, None),  # Missing required modifier
        ]
        
        self.test_index = 0
        self.received_commands = {}
        
        # Start test timer
        self.timer = self.create_timer(1.0, self.send_next_command)
        
        self.get_logger().info("Command processor tester started")
    
    def arm_callback(self, msg: String):
        self.get_logger().info(f"Received arm preset: '{msg.data}'")
        self.received_commands['arm_preset'] = msg.data
    
    def behavior_callback(self, msg: String):
        self.get_logger().info(f"Received behavior command: '{msg.data}'")
        self.received_commands['behavior_command'] = msg.data
    
    def send_next_command(self):
        if self.test_index >= len(self.test_commands):
            self.get_logger().info("All tests completed!")
            self.timer.cancel()
            rclpy.shutdown()
            return
        
        command, expected_topic, expected_value = self.test_commands[self.test_index]
        
        # Clear received commands
        self.received_commands.clear()
        
        # Send command
        msg = String()
        msg.data = command
        self.command_pub.publish(msg)
        self.get_logger().info(f"\nTest {self.test_index + 1}: Sent command '{command}'")
        
        # Wait a bit for processing
        time.sleep(0.2)
        
        # Check results
        if expected_topic is None:
            # Should be ignored
            if len(self.received_commands) == 0:
                self.get_logger().info(f"✓ Command correctly ignored")
            else:
                self.get_logger().error(f"✗ Command should have been ignored but got: {self.received_commands}")
        else:
            # Should have received specific command
            if expected_topic in self.received_commands:
                received = self.received_commands[expected_topic]
                if received == expected_value:
                    self.get_logger().info(f"✓ Correct: {expected_topic} = '{expected_value}'")
                else:
                    self.get_logger().error(f"✗ Wrong value: expected '{expected_value}', got '{received}'")
            else:
                self.get_logger().error(f"✗ No command received on {expected_topic}")
        
        self.test_index += 1


def main(args=None):
    rclpy.init(args=args)
    
    # First ensure command processor is running
    print("\nMake sure command_processor is running:")
    print("  ros2 run by_your_command command_processor")
    print("\nStarting tests in 3 seconds...\n")
    time.sleep(3)
    
    node = CommandProcessorTester()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()