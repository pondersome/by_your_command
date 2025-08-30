#!/usr/bin/env python3
"""
Audio recorder for capturing claps and speech samples
Records triggered audio segments to WAV files for analysis
"""

import rclpy
from rclpy.node import Node
from audio_common_msgs.msg import AudioStamped
from std_msgs.msg import String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import wave
import os
from datetime import datetime
from collections import deque
import time

def get_timestamp():
    """Get formatted timestamp HH:MM:SS.mmm"""
    now = datetime.now()
    return now.strftime("%H:%M:%S.%f")[:-3]

class AudioRecorder(Node):
    def __init__(self):
        super().__init__('audio_recorder')
        
        # Parameters
        self.declare_parameter('audio_topic', '/grunt1/agent/audio')
        self.declare_parameter('output_dir', '/tmp/audio_samples')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('auto_trigger_threshold', 0.1)  # Auto-trigger on loud sounds
        self.declare_parameter('pre_trigger_ms', 200)  # Record before trigger
        self.declare_parameter('post_trigger_ms', 800)  # Record after trigger
        
        # Get parameters
        self.audio_topic = self.get_parameter('audio_topic').value
        self.output_dir = self.get_parameter('output_dir').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.auto_trigger_threshold = self.get_parameter('auto_trigger_threshold').value
        self.pre_trigger_ms = self.get_parameter('pre_trigger_ms').value
        self.post_trigger_ms = self.get_parameter('post_trigger_ms').value
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Calculate buffer sizes
        self.pre_trigger_samples = int(self.sample_rate * self.pre_trigger_ms / 1000)
        self.post_trigger_samples = int(self.sample_rate * self.post_trigger_ms / 1000)
        self.total_samples = self.pre_trigger_samples + self.post_trigger_samples
        
        # QoS setup
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        
        # Subscribe to audio
        self.audio_sub = self.create_subscription(
            AudioStamped,
            self.audio_topic,
            self.audio_callback,
            qos_profile=qos
        )
        
        # Subscribe to commands
        self.command_sub = self.create_subscription(
            String,
            'recorder_command',
            self.command_callback,
            10
        )
        
        # Audio buffer (circular buffer for pre-trigger)
        self.audio_buffer = deque(maxlen=self.sample_rate * 2)  # 2 seconds
        
        # Recording state
        self.recording = False
        self.recording_buffer = []
        self.recording_start_time = 0
        self.recording_type = "auto"
        self.samples_to_record = 0
        
        # Counters
        self.clap_count = 0
        self.speech_count = 0
        self.other_count = 0
        self.auto_count = 0
        
        # Cooldown to prevent multiple triggers
        self.last_trigger_time = 0
        self.cooldown_time = 1.0  # 1 second between recordings
        
        self.get_logger().info(f"[{get_timestamp()}] Audio Recorder initialized")
        self.get_logger().info(f"  Output directory: {self.output_dir}")
        self.get_logger().info(f"  Auto-trigger threshold: {self.auto_trigger_threshold}")
        self.get_logger().info(f"  Recording window: -{self.pre_trigger_ms}ms to +{self.post_trigger_ms}ms")
        self.get_logger().info("")
        self.get_logger().info("Commands: (publish to /recorder_command)")
        self.get_logger().info("  'clap' - Record next sound as clap")
        self.get_logger().info("  'speech' - Record next sound as speech")
        self.get_logger().info("  'double' - Record next sound as double clap")
        self.get_logger().info("  'other' - Record next sound as other")
        self.get_logger().info("  'auto' - Enable auto-trigger mode")
        self.get_logger().info("  'stop' - Stop recording")
        
    def command_callback(self, msg: String):
        """Handle recording commands"""
        command = msg.data.lower().strip()
        
        if command in ['clap', 'speech', 'double', 'other']:
            self.recording_type = command
            self.get_logger().info(f"[{get_timestamp()}] Ready to record '{command}' - make the sound now!")
            # Will trigger on next loud sound
            
        elif command == 'auto':
            self.recording_type = 'auto'
            self.get_logger().info(f"[{get_timestamp()}] Auto-trigger mode enabled")
            
        elif command == 'stop':
            self.recording = False
            self.get_logger().info(f"[{get_timestamp()}] Recording stopped")
            
    def trigger_recording(self, trigger_idx=None):
        """Start recording with pre-trigger samples"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_trigger_time < self.cooldown_time:
            return
            
        self.last_trigger_time = current_time
        self.recording = True
        self.recording_start_time = current_time
        
        # Get pre-trigger samples from buffer
        if trigger_idx is None:
            trigger_idx = len(self.audio_buffer)
            
        start_idx = max(0, trigger_idx - self.pre_trigger_samples)
        self.recording_buffer = list(self.audio_buffer)[start_idx:trigger_idx]
        self.samples_to_record = self.post_trigger_samples
        
        self.get_logger().info(f"[{get_timestamp()}] 🔴 Recording triggered! Type: {self.recording_type}")
        
    def save_recording(self):
        """Save recorded audio to WAV file"""
        if len(self.recording_buffer) == 0:
            return
            
        # Generate filename
        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
        
        if self.recording_type == 'clap':
            self.clap_count += 1
            filename = f"clap_{self.clap_count:03d}_{timestamp}.wav"
        elif self.recording_type == 'speech':
            self.speech_count += 1
            filename = f"speech_{self.speech_count:03d}_{timestamp}.wav"
        elif self.recording_type == 'double':
            self.clap_count += 1
            filename = f"double_{self.clap_count:03d}_{timestamp}.wav"
        elif self.recording_type == 'other':
            self.other_count += 1
            filename = f"other_{self.other_count:03d}_{timestamp}.wav"
        else:  # auto
            self.auto_count += 1
            filename = f"auto_{self.auto_count:03d}_{timestamp}.wav"
            
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to int16
        audio_float = np.array(self.recording_buffer, dtype=np.float32)
        audio_int16 = (audio_float * 32767).astype(np.int16)
        
        # Write WAV file
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
            
        # Calculate peak and RMS for logging
        peak = np.max(np.abs(audio_float))
        rms = np.sqrt(np.mean(audio_float**2))
        
        self.get_logger().info(f"[{get_timestamp()}] 💾 Saved: {filename} "
                             f"(peak={peak:.3f}, rms={rms:.4f}, samples={len(audio_int16)})")
        
        # Reset for next recording
        self.recording_type = 'auto'
        
    def audio_callback(self, msg: AudioStamped):
        """Process incoming audio"""
        # Convert ROS audio to numpy array
        audio_list = msg.audio.audio_data.int16_data
        if len(audio_list) == 0:
            return
            
        audio_int16 = np.array(audio_list, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        
        # Add to circular buffer
        self.audio_buffer.extend(audio_float)
        
        # If recording, collect samples
        if self.recording:
            samples_needed = min(len(audio_float), self.samples_to_record)
            self.recording_buffer.extend(audio_float[:samples_needed])
            self.samples_to_record -= samples_needed
            
            # Check if recording complete
            if self.samples_to_record <= 0:
                self.recording = False
                self.save_recording()
                return
        
        # Check for auto-trigger or manual trigger
        peak = np.max(np.abs(audio_float))
        
        if not self.recording and peak > self.auto_trigger_threshold:
            # Find the peak location in buffer for better alignment
            buffer_array = np.array(list(self.audio_buffer))
            recent_samples = min(len(buffer_array), len(audio_float) * 2)
            recent_peak_idx = len(buffer_array) - recent_samples + np.argmax(np.abs(buffer_array[-recent_samples:]))
            
            # Trigger recording
            if self.recording_type != 'auto' or self.recording_type == 'auto':
                self.trigger_recording(recent_peak_idx)

def main(args=None):
    rclpy.init(args=args)
    
    node = AudioRecorder()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down audio recorder")
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()