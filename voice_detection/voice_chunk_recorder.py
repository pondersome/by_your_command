#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from by_your_command.msg import AudioDataUtterance
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
        self.chunks_received = 0
        self.close_timer = None
        self.last_chunk_time = None
        # Utterance tracking
        self.current_utterance_id = None
        self.current_utterance_file = None
        self.utterance_start_time = None
        
        # QoS profile matching silero_vad_node publishers - increased depth for reliability
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,  # Increased from 10 to handle intermittent drops
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        
        # Subscribe to enhanced chunks with utterance metadata
        self.create_subscription(AudioDataUtterance, 'voice_chunks', self.chunk_callback, qos_profile=qos)
        
        # Optional: still subscribe to voice_activity for debugging/logging only
        self.create_subscription(Bool, 'voice_activity', self.voice_activity_debug, qos_profile=qos)

    def voice_activity_debug(self, msg: Bool):
        """Optional debug callback - voice_activity is no longer used for file management"""
        current_time = time.time()
        
        # Throttle logging to avoid spam
        if not hasattr(self, '_last_debug_log_time'):
            self._last_debug_log_time = 0
            
        if current_time - self._last_debug_log_time >= 2.0:  # Log every 2 seconds max
            self.get_logger().debug(f'Voice activity debug: {msg.data}')
            self._last_debug_log_time = current_time


    def chunk_callback(self, msg: AudioDataUtterance):
        """Enhanced chunk callback with utterance-aware file management"""
        current_time = time.time()
        
        # Check if this is a new utterance
        if self.current_utterance_id != msg.utterance_id:
            # Close previous file if open
            if self.writer:
                self._close_current_file("New utterance detected")
            
            # Start new utterance
            self.current_utterance_id = msg.utterance_id
            self.utterance_start_time = current_time
            
            # Create filename with utterance ID
            utterance_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(msg.utterance_id / 1e9))
            fname = os.path.join(self.output_dir, f"utterance_{msg.utterance_id}_{utterance_time}.wav")
            self.current_utterance_file = fname
            
            # Open new WAV file
            self.writer = wave.open(fname, 'wb')
            self.writer.setnchannels(1)
            self.writer.setsampwidth(2)
            self.writer.setframerate(self.sample_rate)
            self.chunks_received = 0
            self.get_logger().info(f"Started utterance {msg.utterance_id} -> {fname}")
            
        # Convert int16_data list back to bytes
        try:
            audio_array = array('h', msg.int16_data)
            audio_bytes = audio_array.tobytes()
        except Exception as e:
            self.get_logger().error(f'Failed to convert AudioDataUtterance to bytes: {e}')
            return
            
        # Write audio frames to WAV
        if self.writer:
            self.writer.writeframes(audio_bytes)
            self.chunks_received += 1
            self.last_chunk_time = current_time
            
            # Log chunk progress
            chunk_info = f"Chunk {msg.chunk_sequence}"
            if msg.is_utterance_end:
                chunk_info += " (FINAL)"
            self.get_logger().debug(f"Utterance {msg.utterance_id}: {chunk_info} ({len(msg.int16_data)} samples)")
            
            # Handle end of utterance
            if msg.is_utterance_end:
                self._close_current_file("End of utterance detected")
                return
        
        # Cancel existing close timer and start new one (fallback for missing end marker)
        if self.close_timer:
            self.close_timer.cancel()
            
        # Fallback close timer in case end-of-utterance marker is missed
        self.close_timer = self.create_timer(self.close_timeout, self._timeout_close_file)
            
    def _close_current_file(self, reason="Unknown"):
        """Close the current WAV file and log completion"""
        if self.writer:
            self.writer.close()
            elapsed_time = time.time() - self.utterance_start_time if self.utterance_start_time else 0
            self.get_logger().info(
                f"Closed utterance {self.current_utterance_id}: {self.chunks_received} chunks, "
                f"{elapsed_time:.1f}s duration. Reason: {reason}"
            )
            self.writer = None
            
        # Cancel timer if it exists  
        if self.close_timer:
            self.close_timer.cancel()
            self.close_timer = None
            
        # Reset utterance tracking
        self.current_utterance_id = None
        self.current_utterance_file = None
        self.utterance_start_time = None
            
    def _timeout_close_file(self):
        """Timeout callback to close file if no final chunk arrives"""
        if self.writer:
            self.get_logger().warning(
                f"Timeout waiting for end-of-utterance marker for utterance {self.current_utterance_id} "
                f"(received {self.chunks_received} chunks)"
            )
            self._close_current_file("Timeout waiting for end marker")


def main(args=None):
    rclpy.init(args=args)
    node = TestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.writer:
            node._close_current_file("Node shutdown")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
