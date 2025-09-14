#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from audio_common_msgs.msg import AudioStamped, AudioData
from by_your_command.msg import AudioDataUtterance

import numpy as np
import collections
import time
from datetime import datetime
from collections import deque

# Silero VAD imports - direct import from PyPI package
from silero_vad import load_silero_vad, VADIterator, get_speech_timestamps

# CONFIGURABLE PARAMETERS
SAMPLE_RATE = 16000  # audio sampling rate
# Buffer and chunk definitions in frames
DEFAULT_MAX_BUFFER_FRAMES = 250
DEFAULT_PRE_ROLL_FRAMES = 15
DEFAULT_UTTERANCE_CHUNK_FRAMES = 100
# VAD tuning parameters
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_SILENCE_DURATION_MS = 200

def get_timestamp():
    """Get formatted timestamp HH:MM:SS.mmm"""
    now = datetime.now()
    return now.strftime("%H:%M:%S.%f")[:-3]

# Clap detector removed - now using dedicated clap_detector_node

class SileroVADNode(Node):
    def __init__(self):
        super().__init__('silero_vad_node')
        # Set DEBUG log level to see more info
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        
        # Declare frame‐based parameters
        self.declare_parameter('sample_rate', SAMPLE_RATE)
        self.declare_parameter('max_buffer_frames', DEFAULT_MAX_BUFFER_FRAMES)
        self.declare_parameter('pre_roll_frames', DEFAULT_PRE_ROLL_FRAMES)
        self.declare_parameter('utterance_chunk_frames', DEFAULT_UTTERANCE_CHUNK_FRAMES)
        self.declare_parameter('threshold', DEFAULT_THRESHOLD)
        self.declare_parameter('min_silence_duration_ms', DEFAULT_MIN_SILENCE_DURATION_MS)
        
        # Fetch parameter values
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.max_buffer_frames = self.get_parameter('max_buffer_frames').get_parameter_value().integer_value
        self.pre_roll_frames = self.get_parameter('pre_roll_frames').get_parameter_value().integer_value
        self.utterance_chunk_frames = self.get_parameter('utterance_chunk_frames').get_parameter_value().integer_value
        self.threshold = self.get_parameter('threshold').get_parameter_value().double_value
        self.min_silence_duration_ms = self.get_parameter('min_silence_duration_ms').get_parameter_value().integer_value
        # QoS and topics
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        self.create_subscription(AudioStamped, 'audio', self.audio_callback, qos_profile=qos)
        self.voice_pub = self.create_publisher(Bool, 'voice_activity', qos_profile=qos)
        # Publish to new topic name (was voice_chunks)
        self.chunk_pub = self.create_publisher(AudioDataUtterance, 'prompt_voice', qos_profile=qos)
        
        # Subscribe to wake_cmd topic for remote sleep/wake control
        self.create_subscription(Bool, 'wake_cmd', self.wake_cmd_callback, qos_profile=qos)
        
        # Subscribe to new topic name for text-based wake commands (was text_input)
        self.create_subscription(String, 'prompt_text', self.text_input_callback, qos_profile=qos)
        # Load and instantiate VADIterator with tuning
        self.model = load_silero_vad()
        self.vad_iterator = VADIterator(
            self.model,
            sampling_rate=self.sample_rate,
            threshold=self.threshold,
            min_silence_duration_ms=self.min_silence_duration_ms
        )
        self.vad_voice_state = False
        # Initialize buffer and state
        # Buffer for full-utterance mode (used when utterance_chunk_frames == 0)
        self.utterance_buffer = []
        # Circular buffer for VAD and pre-roll (always maintained)
        self.frame_buffer = collections.deque(maxlen=self.max_buffer_frames)
        # For chunking mode: accumulate frames directly, don't rely on circular buffer indices
        self.chunking_buffer = []  # Frames since last published chunk
        self.in_utterance = False
        self.chunk_count = 0
        # Utterance tracking
        self.current_utterance_id = None
        self.utterance_start_timestamp = None
        # End-of-utterance detection with one-frame delay
        self.pending_utterance_end = False
        self.last_chunk_data = None  # Store last chunk for end-of-utterance marking
        
        # Buffer for accumulating samples until we have exactly 512 for Silero
        self.vad_sample_buffer = np.array([], dtype=np.float32)
        self.log_info("Silero VAD requires exactly 512 samples at 16kHz")
        
        # Sleep/wake state for controlling whether VAD is active (default awake)
        self.is_awake = True
        
    def log_info(self, msg):
        """Log with custom format"""
        # Use ROS logger but with our format
        self.get_logger().info(f"[{get_timestamp()}] [vad] {msg}")
        
    def log_debug(self, msg):
        """Debug log with custom format"""
        self.get_logger().debug(f"[{get_timestamp()}] [vad] DEBUG: {msg}")
            
    def log_warning(self, msg):
        """Warning log with custom format"""
        self.get_logger().warning(f"[{get_timestamp()}] [vad] WARNING: {msg}")
        
    def log_error(self, msg):
        """Error log with custom format"""
        self.get_logger().error(f"[{get_timestamp()}] [vad] ERROR: {msg}")

    def wake_cmd_callback(self, msg: Bool):
        """Handle wake_cmd topic for remote sleep/wake control"""
        previous_state = self.is_awake
        self.is_awake = msg.data
        
        if previous_state != self.is_awake:
            state_str = "AWAKE" if self.is_awake else "SLEEPING"
            self.log_info(f"⚠️ Sleep/wake state changed to: {state_str} (external command)")
        else:
            # Log redundant state changes for debugging
            self.log_debug(f"Received wake_cmd={msg.data} but already in that state")

    def text_input_callback(self, msg: String):
        """Handle text_input messages for text-based sleep/wake commands (backup mechanism)"""
        text_content = msg.data.lower().strip()
        
        # Check for wake commands
        wake_commands = ['wake', 'awaken', 'wake up', 'wakeup']
        sleep_commands = ['sleep', 'go to sleep', 'sleeptime']
        
        if any(wake_cmd in text_content for wake_cmd in wake_commands):
            if not self.is_awake:
                self.log_info(f"📝 Text wake command received: '{msg.data}' - waking up")
                self.is_awake = True
            else:
                self.log_info(f"📝 Text wake command received but already awake: '{msg.data}'")
        elif any(sleep_cmd in text_content for sleep_cmd in sleep_commands):
            if self.is_awake:
                self.log_info(f"📝 Text sleep command received: '{msg.data}' - going to sleep")
                self.is_awake = False
            else:
                self.log_info(f"📝 Text sleep command received but already sleeping: '{msg.data}'")
        else:
            # Log other text input but don't act on it
            self.log_debug(f"📝 Text input received (no sleep/wake command): '{msg.data}'")

    def audio_callback(self, msg: AudioStamped):
        # Debug: Track audio callback frequency
        if not hasattr(self, '_audio_callback_count'):
            self._audio_callback_count = 0
        self._audio_callback_count += 1
        
        if self._audio_callback_count % 100 == 0:  # Every 100 callbacks
            print(f"[AUDIO DEBUG] Received {self._audio_callback_count} audio callbacks, is_awake={self.is_awake}")
        
        # Convert incoming AudioStamped to numpy int16
        audio_list = msg.audio.audio_data.int16_data
        
        # Skip empty chunks
        if len(audio_list) == 0:
            self.log_warning("Received empty audio chunk, skipping")
            return
        
        # Check if node is awake (sleep/wake state)
        if not self.is_awake:
            # When sleeping, do not process audio
            return
        
        # Log chunk size to debug Silero requirements
        if not hasattr(self, '_chunk_count'):
            self._chunk_count = 0
        self._chunk_count += 1
        if self._chunk_count <= 10 or self._chunk_count % 100 == 0:
            self.log_info(f"Audio chunk #{self._chunk_count}: {len(audio_list)} samples")
            
        audio_int16 = np.array(audio_list, dtype=np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # Convert to float and add to VAD buffer
        audio_float = audio_int16.astype(np.float32) / 32768.0
        self.vad_sample_buffer = np.concatenate([self.vad_sample_buffer, audio_float])
        
        # Process VAD in 512-sample chunks
        voice_activity = self.vad_voice_state  # Keep previous state by default
        
        while len(self.vad_sample_buffer) >= 512:
            # Extract exactly 512 samples for Silero
            vad_chunk = self.vad_sample_buffer[:512]
            self.vad_sample_buffer = self.vad_sample_buffer[512:]
            
            # Process through VAD
            events = self.vad_iterator(vad_chunk) or []
            for _ in events:
                self.vad_voice_state = not self.vad_voice_state
            voice_activity = self.vad_voice_state
            
            if self._chunk_count <= 10 or self._chunk_count % 100 == 0:
                self.log_debug(f"Processed 512-sample VAD chunk, {len(self.vad_sample_buffer)} samples remaining in buffer")
        
        # Log buffer accumulation status
        if len(self.vad_sample_buffer) > 0 and (self._chunk_count <= 10 or self._chunk_count % 100 == 0):
            self.log_debug(f"VAD buffer accumulating: {len(self.vad_sample_buffer)}/512 samples")
        
        # Log speech activity on state changes or periodic intervals
        current_time = time.time()
        should_log = False
        
        # Check if this is a state change
        if hasattr(self, '_prev_log_voice_activity'):
            if voice_activity != self._prev_log_voice_activity:
                # State changed - always log and reset timer
                should_log = True
                self._last_activity_log_time = current_time
        else:
            # First time - always log
            should_log = True
            self._last_activity_log_time = current_time
            
        # Check if 10 seconds have passed since last log
        if not should_log:
            if not hasattr(self, '_last_activity_log_time') or (current_time - self._last_activity_log_time) >= 10.0:
                should_log = True
                self._last_activity_log_time = current_time
                
        if should_log:
            self.log_info(f'Voice activity: {voice_activity}')
            
        self._prev_log_voice_activity = voice_activity
        # Append frame
        self.frame_buffer.append(audio_bytes)
        # Track VAD state transitions for utterance end detection
        vad_ended_voice = (not voice_activity and self.in_utterance and 
                          hasattr(self, '_prev_voice_activity') and self._prev_voice_activity)
        self._prev_voice_activity = voice_activity
        # Handle pending utterance end from previous frame
        if self.pending_utterance_end and self.last_chunk_data is not None:
            # Mark the last chunk as end-of-utterance and publish
            self.last_chunk_data.is_utterance_end = True
            self.chunk_pub.publish(self.last_chunk_data)
            self.log_info(f'Published end-of-utterance chunk for utterance {self.last_chunk_data.utterance_id}')
            self.pending_utterance_end = False
            self.last_chunk_data = None
        
        # Utterance start
        if voice_activity and not self.in_utterance:
            self.in_utterance = True
            self.chunk_count = 0
            # Create new utterance ID from current timestamp
            self.utterance_start_timestamp = msg.header.stamp
            self.current_utterance_id = int(msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec)
            self.log_info(f'Voice detected. Starting utterance {self.current_utterance_id}.')
            self.voice_pub.publish(Bool(data=True))
            
            if self.utterance_chunk_frames == 0:
                # Full utterance mode: initialize with pre-roll from circular buffer
                pre_roll_frames = list(self.frame_buffer)[-self.pre_roll_frames:]
                self.utterance_buffer = pre_roll_frames[:]
                self.log_info(f'Initialized utterance buffer with {len(self.utterance_buffer)} pre-roll frames')
            else:
                # Chunking mode: initialize chunking buffer with pre-roll
                pre_roll_frames = list(self.frame_buffer)[-self.pre_roll_frames:]
                self.chunking_buffer = pre_roll_frames[:]
                self.log_info(f'Initialized chunking buffer with {len(self.chunking_buffer)} pre-roll frames')
        
        # Accumulate current frame
        if self.in_utterance:
            if self.utterance_chunk_frames == 0:
                # Full utterance mode: add to utterance buffer
                self.utterance_buffer.append(audio_bytes)
            else:
                # Chunking mode: add to chunking buffer
                self.chunking_buffer.append(audio_bytes)
                
                # Check if we need to publish an interim chunk
                if len(self.chunking_buffer) >= self.utterance_chunk_frames:
                    self.log_info(f'Interim chunk reached. Publishing chunk with {len(self.chunking_buffer)} frames')
                    self._publish_chunking_buffer(is_end=False)
                    # Reset chunking buffer for next chunk (no pre-roll needed for interim chunks)
                    self.chunking_buffer = []
        # Utterance end by VAD state transition
        if vad_ended_voice:
            self.in_utterance = False
            self.log_info(f'Voice ended for utterance {self.current_utterance_id}. Preparing final chunk.')
            self.voice_pub.publish(Bool(data=False))
            # Set flag to mark end-of-utterance on next frame (one-frame delay)
            self.pending_utterance_end = True
            if self.utterance_chunk_frames > 0:
                # Chunking mode: publish any remaining frames in chunking buffer
                if len(self.chunking_buffer) > 0:
                    self.log_info(f'Publishing final chunk with {len(self.chunking_buffer)} remaining frames')
                    # Store for end-of-utterance marking with delay
                    self.last_chunk_data = self._create_chunk_message(is_end=False)
                else:
                    self.log_info('No remaining frames for final chunk')
            else:
                # Full utterance mode: publish entire utterance
                full_audio = b''.join(self.utterance_buffer)
                if len(full_audio) > 0:
                    chunk_msg = self._create_full_utterance_message(full_audio)
                    # Store for end-of-utterance marking with delay
                    self.last_chunk_data = chunk_msg
                else:
                    self.log_warning('No audio data in utterance buffer, not publishing chunk')
            self.vad_iterator.reset_states()
            
            # Reset buffers to prevent corruption on next utterance
            self.utterance_buffer = []
            self.chunking_buffer = []
            self.chunk_count = 0
            # Clean up utterance tracking
            self.current_utterance_id = None
            self.utterance_start_timestamp = None


    def publish_chunk(self):
        total = len(self.frame_buffer)
        # Determine start index
        if self.chunk_count == 0:
            start_idx = max(0, self.utterance_start_buffer_idx - self.pre_roll_frames)
        else:
            start_idx = self.last_chunk_buffer_idx
        # Slice frames
        frames = list(self.frame_buffer)[start_idx:total]
        audio_data = b''.join(frames)
        duration = len(audio_data) / 2 / self.sample_rate
        self.log_info(
            f'Publishing chunk {self.chunk_count}: frames {start_idx}-{total}, duration {duration:.2f}s'
        )
        # Publish
        chunk_msg = AudioData()
        chunk_msg.int16_data = np.frombuffer(audio_data, dtype=np.int16).tolist()
        self.chunk_pub.publish(chunk_msg)
        # Update counters
        self.last_chunk_buffer_idx = total
        self.chunk_count += 1

    def _publish_chunking_buffer(self, is_end=False):
        """Publish current chunking buffer contents"""
        chunk_msg = self._create_chunk_message(is_end)
        if not is_end:
            # Publish immediately for interim chunks
            self.chunk_pub.publish(chunk_msg)
            duration = len(b''.join(self.chunking_buffer)) / 2 / self.sample_rate
            self.log_info(
                f'Published chunk {self.chunk_count}: {len(self.chunking_buffer)} frames, duration {duration:.2f}s'
            )
        else:
            # Store for end-of-utterance marking with delay
            self.last_chunk_data = chunk_msg
            
        self.chunk_count += 1
    
    def _create_chunk_message(self, is_end=False):
        """Create AudioDataUtterance message from current chunking buffer"""
        audio_data = b''.join(self.chunking_buffer)
        
        chunk_msg = AudioDataUtterance()
        chunk_msg.int16_data = np.frombuffer(audio_data, dtype=np.int16).tolist()
        chunk_msg.utterance_id = self.current_utterance_id or 0
        chunk_msg.is_utterance_end = is_end
        chunk_msg.chunk_sequence = self.chunk_count
        
        return chunk_msg
    
    def _create_full_utterance_message(self, audio_data):
        """Create AudioDataUtterance message for full utterance mode"""
        chunk_msg = AudioDataUtterance()
        chunk_msg.int16_data = np.frombuffer(audio_data, dtype=np.int16).tolist()
        chunk_msg.utterance_id = self.current_utterance_id or 0
        chunk_msg.is_utterance_end = False  # Will be set to True with delay
        chunk_msg.chunk_sequence = 0  # Single chunk in full utterance mode
        
        return chunk_msg
    
    def __del__(self):
        """Handle cleanup when node is destroyed"""
        # If there's a pending end-of-utterance chunk when shutting down, publish it
        if hasattr(self, 'pending_utterance_end') and self.pending_utterance_end and hasattr(self, 'last_chunk_data') and self.last_chunk_data is not None:
            self.last_chunk_data.is_utterance_end = True
            if hasattr(self, 'chunk_pub'):
                self.chunk_pub.publish(self.last_chunk_data)
                self.log_info(f'Published final end-of-utterance chunk during shutdown for utterance {self.last_chunk_data.utterance_id}')


def main(args=None):
    rclpy.init(args=args)
    node = SileroVADNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
