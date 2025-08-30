#!/usr/bin/env python3
"""
Data-driven clap detector based on actual measurements
Uses ZCR as primary discriminator with supporting features
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

class DataDrivenClapDetector(Node):
    def __init__(self):
        super().__init__('data_driven_clap_detector')
        
        # Parameters
        self.declare_parameter('audio_topic', '/grunt1/agent/audio')
        self.declare_parameter('wake_cmd_topic', 'wake_cmd')
        self.declare_parameter('sample_rate', 16000)
        
        # Detection thresholds based on measured data
        # ZCR is the best discriminator: claps ~0.33, speech ~0.16
        self.declare_parameter('zcr_threshold', 0.20)  # Lowered more to catch quiet claps
        self.declare_parameter('peak_threshold', 0.01)  # Lower threshold for quiet claps
        
        # Supporting features (more lenient due to variability)
        self.declare_parameter('max_rise_time_ms', 60)  # More lenient
        self.declare_parameter('min_spectral_centroid', 1500)  # Much more lenient
        
        # Double clap detection (avoid reverb in 300-400ms range)
        self.declare_parameter('double_clap_min_gap_ms', 160)  # Avoid reverb
        self.declare_parameter('double_clap_max_gap_ms', 1200)  # Tighter upper bound
        
        # Get parameters
        self.audio_topic = self.get_parameter('audio_topic').value
        self.wake_cmd_topic = self.get_parameter('wake_cmd_topic').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.zcr_threshold = self.get_parameter('zcr_threshold').value
        self.peak_threshold = self.get_parameter('peak_threshold').value
        self.max_rise_time_ms = self.get_parameter('max_rise_time_ms').value
        self.min_spectral_centroid = self.get_parameter('min_spectral_centroid').value
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
        
        # Audio buffer (200ms window)
        self.buffer_size = int(self.sample_rate * 0.2)
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        # Detection state
        self.last_clap_time = 0
        self.clap_times = []
        self.cooldown_time = 0.1
        
        # Stats
        self.callback_count = 0
        self.single_claps = 0
        self.double_claps = 0
        self.rejections = 0
        
        self.get_logger().info(f"[{get_timestamp()}] Data-driven clap detector initialized")
        self.get_logger().info(f"  Based on analysis of recorded samples:")
        self.get_logger().info(f"    ZCR threshold: {self.zcr_threshold} (claps: 0.33, speech: 0.16)")
        self.get_logger().info(f"    Peak threshold: {self.peak_threshold}")
        self.get_logger().info(f"    Centroid threshold: {self.min_spectral_centroid} Hz")
        
    def calculate_zcr(self, audio):
        """Calculate zero crossing rate - primary discriminator"""
        zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
        zcr = zero_crossings / len(audio)
        return zcr
    
    def calculate_spectral_centroid(self, audio):
        """Calculate spectral centroid - supporting feature"""
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        magnitude = np.abs(fft)
        
        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            centroid = 0
        return centroid
    
    def calculate_rise_time(self, audio):
        """Calculate rise time - supporting feature"""
        envelope = np.abs(audio)
        
        # Smooth slightly
        window = 5
        if len(envelope) > window:
            envelope = np.convolve(envelope, np.ones(window)/window, mode='same')
        
        # Find peak
        peak_idx = np.argmax(envelope)
        peak_value = envelope[peak_idx]
        
        # Find onset (20% of peak for better stability)
        onset_threshold = peak_value * 0.2
        onset_idx = 0
        for i in range(peak_idx):
            if envelope[i] < onset_threshold:
                onset_idx = i
        
        rise_samples = peak_idx - onset_idx
        rise_ms = (rise_samples / self.sample_rate) * 1000
        return rise_ms, peak_value
    
    def detect_clap(self, audio):
        """Detect clap based on measured characteristics"""
        # Calculate features
        zcr = self.calculate_zcr(audio)
        centroid = self.calculate_spectral_centroid(audio)
        rise_ms, peak = self.calculate_rise_time(audio)
        
        # Primary check: ZCR (best discriminator)
        if zcr < self.zcr_threshold:
            return False, f"low_zcr({zcr:.3f})", {'zcr': zcr, 'peak': peak}
        
        # Secondary check: Peak amplitude
        if peak < self.peak_threshold:
            return False, f"low_peak({peak:.3f})", {'zcr': zcr, 'peak': peak}
        
        # Tertiary checks (only reject if ZCR is also borderline)
        if centroid < self.min_spectral_centroid:
            # Only reject if ZCR is also low
            if zcr < 0.25:  # Lowered from 0.30
                return False, f"low_freq({centroid:.0f}Hz)", {'zcr': zcr, 'peak': peak, 'centroid': centroid}
        
        if rise_ms > self.max_rise_time_ms:
            # Only reject if ZCR is also low
            if zcr < 0.25:  # Lowered from 0.30
                return False, f"slow_rise({rise_ms:.0f}ms)", {'zcr': zcr, 'peak': peak, 'rise_ms': rise_ms}
        
        # All checks passed or overridden by high ZCR
        return True, "clap", {
            'zcr': zcr,
            'peak': peak,
            'centroid': centroid,
            'rise_ms': rise_ms
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
        
        # Need enough data
        if len(self.audio_buffer) < self.sample_rate // 10:  # 100ms
            return
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_clap_time < self.cooldown_time:
            return
        
        # Always analyze the most recent window (don't pre-filter)
        buffer_array = np.array(list(self.audio_buffer))
        window = buffer_array[-self.sample_rate // 10:]
        
        # Quick check if there's any significant sound in the window
        window_peak = np.max(np.abs(window))
        if window_peak < self.peak_threshold * 0.5:  # Lower threshold for pre-check
            return  # Too quiet, skip
        
        # Analyze the window
        is_clap, reason, features = self.detect_clap(window)
        
        # Log ALL rejections of loud sounds for debugging
        if not is_clap and features.get('peak', 0) > 0.05:
            self.rejections += 1
            # Always log so we can see why claps are missed
            self.get_logger().info(f"[{get_timestamp()}] 🚫 Rejected: {reason}, "
                                 f"ZCR={features.get('zcr', 0):.3f}, "
                                 f"Peak={features.get('peak', 0):.3f}")
        
        if is_clap:
            self.single_claps += 1
            self.last_clap_time = current_time
            self.clap_times.append(current_time)
            
            self.get_logger().info(f"[{get_timestamp()}] 👏 CLAP #{self.single_claps}! "
                                 f"ZCR: {features['zcr']:.3f}, "
                                 f"Peak: {features['peak']:.3f}, "
                                 f"Freq: {features.get('centroid', 0):.0f}Hz")
            
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
    
    node = DataDrivenClapDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down data-driven clap detector")
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()