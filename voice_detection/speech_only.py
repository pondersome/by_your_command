#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from audio_common_msgs.msg import AudioData
import wave

# Default sample rate if not overridden via parameter
DEFAULT_SAMPLE_RATE = 16000
import os
import time

class TestNode(Node):
    def __init__(self):
        super().__init__('test_node')
        # Parameters
        self.declare_parameter('output_dir', '/tmp')
        self.declare_parameter('sample_rate', DEFAULT_SAMPLE_RATE)
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        self.writer = None
        # Subscribe to voice activity and chunks
        self.create_subscription(Bool, 'voice_activity', self.voice_callback, 10)
        self.create_subscription(AudioData, 'speech_chunks', self.chunk_callback, 10)

    def voice_callback(self, msg: Bool):
        if msg.data:
            # Utterance start: open new WAV file
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(self.output_dir, f"utterance_{ts}.wav")
            self.writer = wave.open(fname, 'wb')
            self.writer.setnchannels(1)
            self.writer.setsampwidth(2)
            self.writer.setframerate(self.sample_rate)
            self.get_logger().info(f"Started writing {fname}")
        else:
            # Utterance end: close file
            if self.writer:
                self.writer.close()
                self.get_logger().info("Closed WAV file")
                self.writer = None

    def chunk_callback(self, msg: AudioData):
        if self.writer:
            audio_bytes = bytes(msg.data)
            self.writer.writeframes(audio_bytes)


def main(args=None):
    rclpy.init(args=args)
    node = TestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.writer:
            node.writer.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
