#!/usr/bin/env python3
"""
Sliding window clap detector
Accumulates audio in a continuous buffer for better clap detection
"""

import rclpy
from rclpy.node import Node
from audio_common_msgs.msg import AudioStamped
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import time
from datetime import datetime
from collections import deque

def get_timestamp():
    """Get formatted timestamp HH:MM:SS.mmm"""
    now = datetime.now()
    return now.strftime("%H:%M:%S.%f")[:-3]

class SlidingWindowClapDetector(Node):
    def __init__(self):
        super().__init__('sliding_window_clap_detector')
        
        # Parameters
        self.declare_parameter('audio_topic', '/grunt1/agent/audio')
        self.declare_parameter('wake_cmd_topic', 'wake_cmd')
        self.declare_parameter('sample_rate', 16000)
        
        # Detection parameters  
        self.declare_parameter('energy_threshold_ratio', 15.0)  # Higher ratio for better rejection
        self.declare_parameter('peak_threshold', 0.05)  # Minimum peak amplitude
        self.declare_parameter('window_size_ms', 100)  # Analysis window
        self.declare_parameter('min_clap_energy', 0.0001)  # Minimum total energy for valid clap
        self.declare_parameter('double_clap_min_gap_ms', 150)
        self.declare_parameter('double_clap_max_gap_ms', 600)
        
        # Get parameters
        self.audio_topic = self.get_parameter('audio_topic').value
        self.wake_cmd_topic = self.get_parameter('wake_cmd_topic').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.energy_threshold_ratio = self.get_parameter('energy_threshold_ratio').value
        self.peak_threshold = self.get_parameter('peak_threshold').value
        self.window_size_ms = self.get_parameter('window_size_ms').value
        self.min_clap_energy = self.get_parameter('min_clap_energy').value
        self.double_clap_min_gap_ms = self.get_parameter('double_clap_min_gap_ms').value
        self.double_clap_max_gap_ms = self.get_parameter('double_clap_max_gap_ms').value
        
        # Calculate window size in samples
        self.window_size = int(self.window_size_ms * self.sample_rate / 1000)
        
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
        
        # Publisher for wake signal
        self.wake_pub = self.create_publisher(Bool, self.wake_cmd_topic, qos_profile=qos)
        
        # Continuous audio buffer (1 second)
        self.audio_buffer = deque(maxlen=self.sample_rate)
        
        # Background noise estimation
        self.background_rms = 0.001  # Start with reasonable value
        self.noise_floor = 0.001  # Minimum noise floor
        
        # Clap detection state
        self.last_clap_time = 0
        self.clap_times = []
        self.cooldown_time = 0.1  # 100ms cooldown after clap
        
        # Stats
        self.callback_count = 0
        self.single_claps = 0
        self.double_claps = 0
        
        self.get_logger().info(f"[{get_timestamp()}] Sliding window clap detector initialized")
        self.get_logger().info(f"  Listening on: {self.audio_topic}")
        self.get_logger().info(f"  Window size: {self.window_size_ms}ms")
        self.get_logger().info(f"  Energy ratio: {self.energy_threshold_ratio}x")
        self.get_logger().info(f"  Peak threshold: {self.peak_threshold}")
        self.get_logger().info(f"  Min clap energy: {self.min_clap_energy}")
        
    def update_background_noise(self, audio_chunk):
        """Update background noise estimate using percentile method"""
        if len(self.audio_buffer) < self.window_size:
            return  # Not enough data yet
            
        # Get recent audio (last 500ms)
        recent_samples = min(self.sample_rate // 2, len(self.audio_buffer))
        recent_audio = np.array(list(self.audio_buffer))[-recent_samples:]
        
        # Calculate RMS in small chunks
        chunk_size = 160  # 10ms chunks
        rms_values = []
        for i in range(0, len(recent_audio) - chunk_size, chunk_size):
            chunk = recent_audio[i:i+chunk_size]
            rms = np.sqrt(np.mean(chunk**2))
            rms_values.append(rms)
        
        if rms_values:
            # Use 20th percentile as background estimate (robust to outliers)
            self.background_rms = np.percentile(rms_values, 20)
            # Enforce minimum noise floor
            self.background_rms = max(self.background_rms, self.noise_floor)
            
    def detect_clap_in_window(self):
        """Analyze the current window for clap characteristics"""
        if len(self.audio_buffer) < self.window_size:
            return False, 0, 0
            
        # Get analysis window
        window = np.array(list(self.audio_buffer))[-self.window_size:]
        
        # Calculate features
        rms = np.sqrt(np.mean(window**2))
        peak = np.max(np.abs(window))
        
        # Calculate energy ratio
        energy_ratio = rms / self.background_rms if self.background_rms > 0 else 0
        
        # Clap detection criteria
        is_clap = (
            peak > self.peak_threshold and  # Strong peak
            energy_ratio > self.energy_threshold_ratio and  # Much louder than background
            rms > self.min_clap_energy  # Minimum energy threshold
        )
        
        return is_clap, peak, energy_ratio
        
    def check_double_clap(self):
        """Check if recent claps form a double-clap pattern"""
        current_time = time.time()
        
        # Clean old claps
        max_age = self.double_clap_max_gap_ms / 1000 * 2
        self.clap_times = [t for t in self.clap_times if current_time - t < max_age]
        
        # Check for double clap
        if len(self.clap_times) >= 2:
            gap_ms = (self.clap_times[-1] - self.clap_times[-2]) * 1000
            
            if self.double_clap_min_gap_ms <= gap_ms <= self.double_clap_max_gap_ms:
                self.get_logger().info(f"[{get_timestamp()}] 👏👏 DOUBLE CLAP! Gap: {gap_ms:.0f}ms")
                self.clap_times = []  # Clear after detection
                return True
                
        return False
        
    def audio_callback(self, msg: AudioStamped):
        """Process incoming audio"""
        self.callback_count += 1
        
        # Convert ROS audio to numpy array
        audio_list = msg.audio.audio_data.int16_data
        if len(audio_list) == 0:
            return
            
        audio_int16 = np.array(audio_list, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        
        # Add to continuous buffer
        self.audio_buffer.extend(audio_float)
        
        # Update background noise estimate
        if self.callback_count % 10 == 0:  # Every 10 callbacks
            self.update_background_noise(audio_float)
        
        # Check for clap in current window
        current_time = time.time()
        
        # Skip if in cooldown period
        if current_time - self.last_clap_time < self.cooldown_time:
            return
            
        is_clap, peak, energy_ratio = self.detect_clap_in_window()
        
        if is_clap:
            self.single_claps += 1
            self.last_clap_time = current_time
            self.clap_times.append(current_time)
            
            # Calculate window RMS for logging
            window = np.array(list(self.audio_buffer))[-self.window_size:]
            window_rms = np.sqrt(np.mean(window**2))
            
            self.get_logger().info(f"[{get_timestamp()}] 👏 CLAP #{self.single_claps}! "
                                 f"Peak: {peak:.3f}, RMS: {window_rms:.6f}, "
                                 f"Ratio: {energy_ratio:.1f}x")
            
            # Check for double clap
            if self.check_double_clap():
                self.double_claps += 1
                
                # Send wake command
                wake_msg = Bool()
                wake_msg.data = True
                self.wake_pub.publish(wake_msg)
                self.get_logger().info(f"[{get_timestamp()}] ✅ Wake command sent!")
        
        # Periodic stats
        if self.callback_count % 100 == 0:
            # Calculate current window stats
            if len(self.audio_buffer) >= self.window_size:
                window = np.array(list(self.audio_buffer))[-self.window_size:]
                current_rms = np.sqrt(np.mean(window**2))
                current_peak = np.max(np.abs(window))
            else:
                current_rms = 0
                current_peak = 0
                
            self.get_logger().info(f"[{get_timestamp()}] Stats: callbacks={self.callback_count}, "
                                 f"bg_rms={self.background_rms:.6f}, "
                                 f"current_rms={current_rms:.6f}, "
                                 f"current_peak={current_peak:.3f}, "
                                 f"single={self.single_claps}, double={self.double_claps}")

def main(args=None):
    rclpy.init(args=args)
    
    node = SlidingWindowClapDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down sliding window clap detector")
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()