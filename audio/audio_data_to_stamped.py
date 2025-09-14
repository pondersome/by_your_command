#!/usr/bin/env python3
"""
Audio Data to Stamped Converter Node

Converts AudioData messages to AudioStamped messages with proper header and audio info.
This bridges the gap between the OpenAI agent output and audio_player_node requirements.

Author: Karim Virani
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from audio_common_msgs.msg import AudioData, AudioStamped, Audio, AudioInfo
from std_msgs.msg import Header
from builtin_interfaces.msg import Time


class AudioDataToStampedNode(Node):
    def __init__(self):
        super().__init__('audio_data_to_stamped')
        
        # Parameters - updated to new naming convention
        self.declare_parameter('input_topic', '/response_voice')  # was /audio_out
        self.declare_parameter('output_topic', '/response_voice_stamped')  # was /audio_out_stamped
        self.declare_parameter('sample_rate', 16000)  # Standardized to 16kHz
        self.declare_parameter('format', 8)  # paInt16
        self.declare_parameter('channels', 1)
        
        # Get parameters
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.format = self.get_parameter('format').value
        self.channels = self.get_parameter('channels').value
        
        # QoS profile
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        
        # Create subscriber and publisher
        self.subscription = self.create_subscription(
            AudioData,
            self.input_topic,
            self.audio_callback,
            qos
        )
        
        self.publisher = self.create_publisher(
            AudioStamped,
            self.output_topic,
            qos
        )
        
        self.get_logger().info(
            f"Converting AudioData from {self.input_topic} to AudioStamped on {self.output_topic}"
        )
        self.get_logger().info(
            f"Audio format: {self.sample_rate}Hz, {self.channels} channel(s), format {self.format}"
        )
        
        self.msg_count = 0
        
    def audio_callback(self, msg: AudioData):
        """Convert AudioData to AudioStamped"""
        # Create AudioStamped message
        stamped_msg = AudioStamped()
        
        # Set header with current timestamp
        stamped_msg.header = Header()
        stamped_msg.header.stamp = self.get_clock().now().to_msg()
        stamped_msg.header.frame_id = "audio"
        
        # Create Audio message
        stamped_msg.audio = Audio()
        
        # Set AudioInfo
        stamped_msg.audio.info = AudioInfo()
        stamped_msg.audio.info.format = self.format
        stamped_msg.audio.info.channels = self.channels
        stamped_msg.audio.info.rate = self.sample_rate
        
        # Determine chunk size from data
        if msg.int16_data:
            stamped_msg.audio.info.chunk = len(msg.int16_data)
        elif msg.uint8_data:
            stamped_msg.audio.info.chunk = len(msg.uint8_data) // 2  # 2 bytes per sample
        else:
            self.get_logger().warning("No audio data found in message")
            return
        
        # Copy audio data
        stamped_msg.audio.audio_data = msg
        
        # Publish
        self.publisher.publish(stamped_msg)
        self.msg_count += 1
        
        # Log periodically
        if self.msg_count % 100 == 0:
            self.get_logger().debug(f"Converted {self.msg_count} messages")


def main(args=None):
    rclpy.init(args=args)
    node = AudioDataToStampedNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()