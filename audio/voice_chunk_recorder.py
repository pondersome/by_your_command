#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from by_your_command.msg import AudioDataUtterance
from audio_common_msgs.msg import AudioData, AudioStamped
import wave
from array import array
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# Default sample rate if not overridden via parameter
DEFAULT_SAMPLE_RATE = 16000
import os
import time

class VoiceChunkRecorder(Node):
    def __init__(self):
        super().__init__('voice_chunk_recorder')
        # Parameters
        self.declare_parameter('output_dir', '/tmp')
        self.declare_parameter('sample_rate', DEFAULT_SAMPLE_RATE)
        self.declare_parameter('close_timeout_sec', 2.0)  # Timeout for waiting for final chunk
        self.declare_parameter('input_mode', 'utterance')  # 'utterance', 'audio_data', or 'audio_stamped'
        self.declare_parameter('input_topic', '/response_voice')  # Topic for audio_data mode (was /audio_out)
        self.declare_parameter('input_sample_rate', 24000)  # Sample rate of input audio
        self.declare_parameter('audio_timeout', 10.0)  # Timeout for audio_data mode
        
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.close_timeout = self.get_parameter('close_timeout_sec').get_parameter_value().double_value
        self.input_mode = self.get_parameter('input_mode').get_parameter_value().string_value
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.input_sample_rate = self.get_parameter('input_sample_rate').get_parameter_value().integer_value
        self.audio_timeout = self.get_parameter('audio_timeout').get_parameter_value().double_value
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
        
        # Audio data mode tracking
        self.audio_data_timer = None
        self.audio_data_start_time = None
        self.audio_data_file = None
        
        # QoS profile matching silero_vad_node publishers - increased depth for reliability
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,  # Increased from 10 to handle intermittent drops
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        
        # Subscribe based on input mode
        if self.input_mode == 'audio_data':
            self.get_logger().info(f"Starting in audio_data mode, subscribing to {self.input_topic}")
            self.get_logger().info(f"Input sample rate: {self.input_sample_rate} Hz, timeout: {self.audio_timeout} seconds")
            self.create_subscription(AudioData, self.input_topic, self.audio_data_callback, qos_profile=qos)
        elif self.input_mode == 'audio_stamped':
            self.get_logger().info(f"Starting in audio_stamped mode, subscribing to {self.input_topic}")
            self.get_logger().info(f"Input sample rate: {self.input_sample_rate} Hz, timeout: {self.audio_timeout} seconds")
            self.create_subscription(AudioStamped, self.input_topic, self.audio_stamped_callback, qos_profile=qos)
        else:
            self.get_logger().info("Starting in utterance mode")
            # Subscribe to enhanced chunks with utterance metadata
            self.create_subscription(AudioDataUtterance, 'prompt_voice', self.chunk_callback, qos_profile=qos)
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
    
    def audio_data_callback(self, msg: AudioData):
        """Handle continuous audio data from /audio_out"""
        current_time = time.time()
        
        # Check which field has data
        audio_bytes = None
        if msg.int16_data:
            # Convert int16 array to bytes
            try:
                audio_array = array('h', msg.int16_data)
                audio_bytes = audio_array.tobytes()
            except Exception as e:
                self.get_logger().error(f'Failed to convert int16_data to bytes: {e}')
                return
        elif msg.uint8_data:
            # Already in byte format
            audio_bytes = bytes(msg.uint8_data)
        else:
            self.get_logger().warning('No audio data found in message')
            return
        
        # Start new file if needed
        if not self.writer:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(self.output_dir, f"audio_out_{timestamp}.wav")
            self.audio_data_file = fname
            self.audio_data_start_time = current_time
            
            # Open WAV file with input sample rate
            self.writer = wave.open(fname, 'wb')
            self.writer.setnchannels(1)  # Mono
            self.writer.setsampwidth(2)  # 16-bit
            self.writer.setframerate(self.input_sample_rate)
            self.chunks_received = 0
            self.get_logger().info(f"Started recording audio_out -> {fname} at {self.input_sample_rate} Hz")
        
        # Write audio data
        self.writer.writeframes(audio_bytes)
        self.chunks_received += 1
        self.last_chunk_time = current_time
        
        # Log progress periodically
        if self.chunks_received % 100 == 0:
            elapsed = current_time - self.audio_data_start_time
            self.get_logger().debug(f"Recording: {self.chunks_received} chunks, {elapsed:.1f}s elapsed")
        
        # Reset timeout timer
        if self.audio_data_timer:
            self.audio_data_timer.cancel()
        
        # Start new timeout timer
        self.audio_data_timer = self.create_timer(self.audio_timeout, self._audio_timeout_callback)
    
    def _audio_timeout_callback(self):
        """Handle timeout for audio_data mode - close file after silence"""
        if self.writer:
            elapsed = time.time() - self.audio_data_start_time
            self.get_logger().info(
                f"Audio timeout reached - closing file after {elapsed:.1f}s, "
                f"{self.chunks_received} chunks received"
            )
            self._close_audio_file()
    
    def _close_audio_file(self):
        """Close audio data recording file"""
        if self.writer:
            self.writer.close()
            self.writer = None
            self.get_logger().info(f"Closed audio file: {self.audio_data_file}")
            
        # Cancel timer
        if self.audio_data_timer:
            self.audio_data_timer.cancel()
            self.audio_data_timer = None
            
        # Reset tracking
        self.audio_data_file = None
        self.audio_data_start_time = None
        self.chunks_received = 0
    
    def audio_stamped_callback(self, msg: AudioStamped):
        """Handle continuous AudioStamped data (wraps AudioData)"""
        # AudioStamped contains header + audio, which contains audio_data + info
        # Forward the audio_data part to the audio_data_callback
        self.audio_data_callback(msg.audio.audio_data)


def main(args=None):
    rclpy.init(args=args)
    node = VoiceChunkRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.writer:
            if node.input_mode in ['audio_data', 'audio_stamped']:
                node._close_audio_file()
            else:
                node._close_current_file("Node shutdown")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
