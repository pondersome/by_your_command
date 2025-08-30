#!/usr/bin/env python3
"""
Transient-based clap detector
Uses sharp onset and decay characteristics to detect claps
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

class TransientClapDetector(Node):
    def __init__(self):
        super().__init__('transient_clap_detector')
        
        # Parameters
        self.declare_parameter('audio_topic', '/grunt1/agent/audio')
        self.declare_parameter('wake_cmd_topic', 'wake_cmd')
        self.declare_parameter('sample_rate', 16000)
        
        # Detection parameters - focus on transient characteristics
        self.declare_parameter('peak_threshold', 0.15)  # Minimum peak amplitude
        self.declare_parameter('rise_time_ms', 10)  # Maximum time to reach peak
        self.declare_parameter('decay_ratio', 0.3)  # How much it should decay after peak
        self.declare_parameter('duration_ms', 150)  # Maximum total duration
        self.declare_parameter('double_clap_min_gap_ms', 150)
        self.declare_parameter('double_clap_max_gap_ms', 600)
        
        # Get parameters
        self.audio_topic = self.get_parameter('audio_topic').value
        self.wake_cmd_topic = self.get_parameter('wake_cmd_topic').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.peak_threshold = self.get_parameter('peak_threshold').value
        self.rise_time_ms = self.get_parameter('rise_time_ms').value
        self.decay_ratio = self.get_parameter('decay_ratio').value
        self.duration_ms = self.get_parameter('duration_ms').value
        self.double_clap_min_gap_ms = self.get_parameter('double_clap_min_gap_ms').value
        self.double_clap_max_gap_ms = self.get_parameter('double_clap_max_gap_ms').value
        
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
        
        # Audio buffer (250ms rolling window)
        self.buffer_size = int(self.sample_rate * 0.25)
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        # Clap detection state
        self.last_clap_time = 0
        self.clap_times = []
        self.cooldown_time = 0.1
        
        # Stats
        self.callback_count = 0
        self.single_claps = 0
        self.double_claps = 0
        self.rejections = 0
        
        self.get_logger().info(f"[{get_timestamp()}] Transient clap detector initialized")
        self.get_logger().info(f"  Listening on: {self.audio_topic}")
        self.get_logger().info(f"  Peak threshold: {self.peak_threshold}")
        self.get_logger().info(f"  Rise time: {self.rise_time_ms}ms")
        self.get_logger().info(f"  Decay ratio: {self.decay_ratio}")
        
    def detect_transient(self):
        """Detect sharp transient (clap-like) in buffer"""
        if len(self.audio_buffer) < self.sample_rate // 10:  # Need at least 100ms
            return False, {}
            
        # Get recent audio
        audio = np.array(list(self.audio_buffer))
        
        # Find peak
        peak_idx = np.argmax(np.abs(audio))
        peak_value = np.abs(audio[peak_idx])
        
        # Check peak threshold
        if peak_value < self.peak_threshold:
            return False, {'peak': peak_value, 'reason': 'low_peak'}
        
        # Calculate envelope for timing analysis
        envelope = np.abs(audio)
        
        # Smooth slightly to remove micro-variations
        window = 5
        if len(envelope) > window:
            envelope = np.convolve(envelope, np.ones(window)/window, mode='same')
        
        # Find onset (10% of peak)
        onset_threshold = peak_value * 0.1
        onset_idx = 0
        for i in range(max(0, peak_idx - 200), peak_idx):  # Look back up to 200 samples
            if envelope[i] < onset_threshold and i < peak_idx - 1:
                onset_idx = i
                
        # Calculate rise time
        rise_samples = peak_idx - onset_idx
        rise_ms = (rise_samples / self.sample_rate) * 1000
        
        # Check rise time (should be sharp)
        if rise_ms > self.rise_time_ms:
            return False, {'peak': peak_value, 'rise_ms': rise_ms, 'reason': 'slow_rise'}
        
        # Find decay point (where it drops to decay_ratio of peak)
        decay_threshold = peak_value * self.decay_ratio
        decay_idx = peak_idx
        for i in range(peak_idx + 1, min(len(envelope), peak_idx + 800)):  # Look ahead up to 50ms
            if envelope[i] < decay_threshold:
                decay_idx = i
                break
        
        # Calculate total duration
        duration_samples = decay_idx - onset_idx
        duration_ms = (duration_samples / self.sample_rate) * 1000
        
        # Check duration (should be brief)
        if duration_ms > self.duration_ms:
            return False, {'peak': peak_value, 'duration_ms': duration_ms, 'reason': 'too_long'}
        
        # Calculate sharpness metric (peak relative to surrounding average)
        surround_start = max(0, onset_idx - 100)
        surround_end = min(len(audio), decay_idx + 100)
        surrounding = np.concatenate([audio[surround_start:onset_idx], audio[decay_idx:surround_end]])
        if len(surrounding) > 0:
            surround_avg = np.mean(np.abs(surrounding))
            sharpness = peak_value / (surround_avg + 0.001)
        else:
            sharpness = peak_value / 0.001
            
        # Clap should be much sharper than surroundings
        if sharpness < 10:
            return False, {'peak': peak_value, 'sharpness': sharpness, 'reason': 'not_sharp'}
        
        # All checks passed
        return True, {
            'peak': peak_value,
            'rise_ms': rise_ms,
            'duration_ms': duration_ms,
            'sharpness': sharpness
        }
    
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
                self.clap_times = []
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
        
        # Add to buffer
        self.audio_buffer.extend(audio_float)
        
        # Check for cooldown
        current_time = time.time()
        if current_time - self.last_clap_time < self.cooldown_time:
            return
        
        # Detect transient
        is_clap, info = self.detect_transient()
        
        # Log rejections of loud sounds
        if not is_clap and info.get('peak', 0) > 0.1:
            self.rejections += 1
            reason = info.get('reason', 'unknown')
            self.get_logger().info(f"[{get_timestamp()}] 🚫 Rejected ({reason}): "
                                 f"peak={info.get('peak', 0):.3f}, "
                                 f"rise={info.get('rise_ms', 0):.0f}ms, "
                                 f"duration={info.get('duration_ms', 0):.0f}ms, "
                                 f"sharpness={info.get('sharpness', 0):.1f}")
        
        if is_clap:
            self.single_claps += 1
            self.last_clap_time = current_time
            self.clap_times.append(current_time)
            
            self.get_logger().info(f"[{get_timestamp()}] 👏 CLAP #{self.single_claps}! "
                                 f"Peak: {info['peak']:.3f}, "
                                 f"Rise: {info['rise_ms']:.0f}ms, "
                                 f"Sharp: {info['sharpness']:.1f}x")
            
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
            self.get_logger().info(f"[{get_timestamp()}] Stats: callbacks={self.callback_count}, "
                                 f"single={self.single_claps}, double={self.double_claps}, "
                                 f"rejected={self.rejections}")

def main(args=None):
    rclpy.init(args=args)
    
    node = TransientClapDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down transient clap detector")
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()