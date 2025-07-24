#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from audio_common_msgs.msg import AudioData
import wave
from array import array
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

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
        self.declare_parameter('close_timeout_sec', 2.0)  # Timeout for waiting for final chunk
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.close_timeout = self.get_parameter('close_timeout_sec').get_parameter_value().double_value
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        self.writer = None
        # Flag to defer closing until after final chunk writes
        self.close_on_chunk = False
        self.close_timer = None
        self.chunks_received = 0
        # QoS profile matching silero_vad_node publishers - increased depth for reliability
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,  # Increased from 10 to handle intermittent drops
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        # Subscribe to voice activity and chunks with BEST_EFFORT

        self.create_subscription(Bool, 'voice_activity', self.voice_callback, qos_profile=qos)
        self.create_subscription(AudioData, 'speech_chunks', self.chunk_callback, qos_profile=qos)

    def voice_callback(self, msg: Bool):
        if msg.data:
            # Cancel any pending close timer
            if self.close_timer:
                self.close_timer.cancel()
                self.close_timer = None
            
            # Reset close flag at utterance start
            self.close_on_chunk = False
            self.chunks_received = 0
            
            # Utterance start: open new WAV file
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(self.output_dir, f"utterance_{ts}.wav")
            self.writer = wave.open(fname, 'wb')
            self.writer.setnchannels(1)
            self.writer.setsampwidth(2)
            self.writer.setframerate(self.sample_rate)
            self.get_logger().info(f"Started writing {fname}")
        else:
            # Utterance end: defer file close until after final chunk writes
            if self.writer:
                self.close_on_chunk = True
                self.get_logger().info("Deferring WAV close until final chunk is written")
                
                # Set timeout to force close if no final chunk arrives
                self.close_timer = self.create_timer(self.close_timeout, self._timeout_close_file)


    def chunk_callback(self, msg: AudioData):
        # Only write if WAV is open
        if not self.writer:
            return
            
        # Convert int16_data list back to bytes
        try:
            audio_array = array('h', msg.int16_data)
            audio_bytes = audio_array.tobytes()
        except Exception as e:
            self.get_logger().error(f'Failed to convert AudioData to bytes: {e}')
            return
            
        # Write audio frames to WAV
        self.writer.writeframes(audio_bytes)
        self.chunks_received += 1
        
        # If this is the last chunk, close now
        if self.close_on_chunk:
            self._force_close_file()
            
    def _force_close_file(self):
        """Force close the current WAV file"""
        if self.writer:
            self.writer.close()
            self.get_logger().info(f"Closed WAV file after receiving {self.chunks_received} chunks")
            self.writer = None
            self.close_on_chunk = False
            
        # Cancel timer if it exists
        if self.close_timer:
            self.close_timer.cancel()
            self.close_timer = None
            
    def _timeout_close_file(self):
        """Timeout callback to close file if no final chunk arrives"""
        if self.writer:
            self.get_logger().warning(f"Timeout waiting for final chunk, force closing file (received {self.chunks_received} chunks)")
            self._force_close_file()


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
