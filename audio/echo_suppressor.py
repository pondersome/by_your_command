#!/usr/bin/env python3
"""
Echo Suppressor Node

Prevents audio feedback by muting microphone input while assistant is speaking.
Subscribes to both raw audio and assistant speaking status.

Author: Karim Virani
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from audio_common_msgs.msg import AudioStamped
from std_msgs.msg import Bool


class EchoSuppressor(Node):
    def __init__(self):
        super().__init__('echo_suppressor')
        
        # State
        self.assistant_speaking = False
        self.mute_count = 0
        
        # QoS profile
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        
        # Subscribe to assistant speaking status
        self.speaking_sub = self.create_subscription(
            Bool,
            'assistant_speaking',  # Relative topic - will be in same namespace
            self.speaking_callback,
            10
        )
        
        # Subscribe to raw audio input
        self.audio_sub = self.create_subscription(
            AudioStamped,
            'audio',  # Relative topic - will be in same namespace
            self.audio_callback,
            qos
        )
        
        # Publisher for filtered audio
        self.audio_pub = self.create_publisher(
            AudioStamped,
            'audio_filtered',  # Relative topic - will be in same namespace
            qos
        )
        
        self.get_logger().info("Echo suppressor started - will mute mic when assistant speaks")
        
    def speaking_callback(self, msg: Bool):
        """Update assistant speaking status"""
        self.assistant_speaking = msg.data
        if self.assistant_speaking:
            self.get_logger().info("🔇 Assistant speaking - muting microphone")
        else:
            self.get_logger().info("🎤 Assistant finished - unmuting microphone")
            self.mute_count = 0
            
    def audio_callback(self, msg: AudioStamped):
        """Filter audio based on speaking status"""
        # If assistant is speaking, drop the audio
        if self.assistant_speaking:
            self.mute_count += 1
            if self.mute_count % 100 == 0:
                self.get_logger().debug(f"Muted {self.mute_count} audio chunks")
            return
            
        # Otherwise, pass it through
        self.audio_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = EchoSuppressor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()