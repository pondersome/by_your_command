#!/usr/bin/env python3
"""
Spectral-based clap detector
Uses frequency characteristics to distinguish claps from speech
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

class SpectralClapDetector(Node):
    def __init__(self):
        super().__init__('spectral_clap_detector')
        
        # Parameters
        self.declare_parameter('audio_topic', '/grunt1/agent/audio')
        self.declare_parameter('wake_cmd_topic', 'wake_cmd')
        self.declare_parameter('sample_rate', 16000)
        
        # Detection parameters
        self.declare_parameter('energy_threshold', 0.005)  # Lower threshold
        self.declare_parameter('spectral_flatness_threshold', 0.3)  # More permissive
        self.declare_parameter('zero_crossing_threshold', 0.10)  # Lower threshold  
        self.declare_parameter('attack_time_ms', 50)  # More permissive
        self.declare_parameter('double_clap_min_gap_ms', 150)
        self.declare_parameter('double_clap_max_gap_ms', 600)
        
        # Get parameters
        self.audio_topic = self.get_parameter('audio_topic').value
        self.wake_cmd_topic = self.get_parameter('wake_cmd_topic').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.energy_threshold = self.get_parameter('energy_threshold').value
        self.spectral_flatness_threshold = self.get_parameter('spectral_flatness_threshold').value
        self.zero_crossing_threshold = self.get_parameter('zero_crossing_threshold').value
        self.attack_time_ms = self.get_parameter('attack_time_ms').value
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
        
        # Audio buffer (500ms)
        self.audio_buffer = deque(maxlen=self.sample_rate // 2)
        
        # Detection state
        self.last_clap_time = 0
        self.clap_times = []
        self.cooldown_time = 0.1
        
        # Stats
        self.callback_count = 0
        self.single_claps = 0
        self.double_claps = 0
        self.speech_rejections = 0
        
        self.get_logger().info(f"[{get_timestamp()}] Spectral clap detector initialized")
        self.get_logger().info(f"  Listening on: {self.audio_topic}")
        self.get_logger().info(f"  Spectral flatness threshold: {self.spectral_flatness_threshold}")
        self.get_logger().info(f"  Zero crossing threshold: {self.zero_crossing_threshold}")
        self.get_logger().info(f"  Attack time: {self.attack_time_ms}ms")
        
    def calculate_spectral_flatness(self, audio_chunk):
        """Calculate spectral flatness (indicates broadband vs tonal)"""
        # Compute power spectrum
        fft = np.fft.rfft(audio_chunk)
        power = np.abs(fft)**2
        
        # Avoid log(0)
        power = power[power > 1e-10]
        
        if len(power) == 0:
            return 0
        
        # Spectral flatness = geometric mean / arithmetic mean
        geometric_mean = np.exp(np.mean(np.log(power)))
        arithmetic_mean = np.mean(power)
        
        if arithmetic_mean > 0:
            flatness = geometric_mean / arithmetic_mean
        else:
            flatness = 0
            
        return flatness
    
    def calculate_zero_crossing_rate(self, audio_chunk):
        """Calculate zero crossing rate (indicates frequency content)"""
        zero_crossings = np.sum(np.diff(np.sign(audio_chunk)) != 0)
        zcr = zero_crossings / len(audio_chunk)
        return zcr
    
    def calculate_attack_time(self, audio_chunk):
        """Calculate attack time (time to reach peak)"""
        envelope = np.abs(audio_chunk)
        
        # Smooth envelope
        window_size = 10
        if len(envelope) > window_size:
            envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
        
        peak_idx = np.argmax(envelope)
        peak_value = envelope[peak_idx]
        
        # Find where envelope reaches 10% of peak
        threshold = peak_value * 0.1
        start_idx = 0
        for i in range(peak_idx):
            if envelope[i] > threshold:
                start_idx = i
                break
        
        # Calculate attack time in ms
        attack_samples = peak_idx - start_idx
        attack_ms = (attack_samples / self.sample_rate) * 1000
        
        return attack_ms
    
    def is_clap_not_speech(self, audio_chunk):
        """Determine if audio chunk is a clap rather than speech"""
        # Calculate energy
        rms = np.sqrt(np.mean(audio_chunk**2))
        peak = np.max(np.abs(audio_chunk))
        
        # Energy gate
        if rms < self.energy_threshold or peak < 0.05:
            return False, "low_energy", {'rms': rms, 'peak': peak}
        
        # Calculate spectral flatness
        flatness = self.calculate_spectral_flatness(audio_chunk)
        
        # Calculate zero crossing rate
        zcr = self.calculate_zero_crossing_rate(audio_chunk)
        
        # Calculate attack time
        attack_ms = self.calculate_attack_time(audio_chunk)
        
        # Debug info
        debug_info = {
            'rms': rms,
            'peak': peak,
            'flatness': flatness,
            'zcr': zcr,
            'attack_ms': attack_ms
        }
        
        # Clap detection criteria
        is_clap = (
            flatness > self.spectral_flatness_threshold and  # Broadband (not tonal like speech)
            zcr > self.zero_crossing_threshold and  # High frequency content
            attack_ms < self.attack_time_ms  # Sharp attack
        )
        
        # Rejection reason
        if not is_clap:
            if flatness <= self.spectral_flatness_threshold:
                reason = f"tonal (flatness={flatness:.2f})"
            elif zcr <= self.zero_crossing_threshold:
                reason = f"low_freq (zcr={zcr:.2f})"
            else:
                reason = f"slow_attack ({attack_ms:.0f}ms)"
        else:
            reason = "clap"
        
        return is_clap, reason, debug_info
    
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
        
        # Need enough data for analysis
        if len(self.audio_buffer) < self.sample_rate // 10:  # 100ms minimum
            return
        
        # Check for cooldown
        current_time = time.time()
        if current_time - self.last_clap_time < self.cooldown_time:
            return
        
        # Analyze recent audio (100ms window)
        window_size = self.sample_rate // 10
        analysis_window = np.array(list(self.audio_buffer))[-window_size:]
        
        # Check if it's a clap
        is_clap, reason, debug_info = self.is_clap_not_speech(analysis_window)
        
        # Log rejections of loud sounds
        if not is_clap and debug_info['peak'] > 0.05:  # Lower threshold to see more
            self.speech_rejections += 1
            self.get_logger().info(f"[{get_timestamp()}] 🚫 Rejected: {reason} "
                                  f"(peak={debug_info['peak']:.3f}, "
                                  f"flatness={debug_info.get('flatness', 0):.3f}, "
                                  f"zcr={debug_info.get('zcr', 0):.3f}, "
                                  f"attack={debug_info.get('attack_ms', 0):.0f}ms)")
        
        if is_clap:
            self.single_claps += 1
            self.last_clap_time = current_time
            self.clap_times.append(current_time)
            
            self.get_logger().info(f"[{get_timestamp()}] 👏 CLAP #{self.single_claps}! "
                                 f"Flatness: {debug_info['flatness']:.2f}, "
                                 f"ZCR: {debug_info['zcr']:.2f}, "
                                 f"Attack: {debug_info['attack_ms']:.0f}ms")
            
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
                                 f"speech_rejected={self.speech_rejections}")

def main(args=None):
    rclpy.init(args=args)
    
    node = SpectralClapDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down spectral clap detector")
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()