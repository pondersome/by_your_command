#!/usr/bin/env python3
"""
Simple energy-based clap detector
Detects claps based on energy spikes without frequency filtering
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

class SimpleEnergyClapDetector(Node):
    def __init__(self):
        super().__init__('simple_energy_clap_detector')
        
        # Parameters
        self.declare_parameter('audio_topic', '/grunt1/agent/audio')
        self.declare_parameter('wake_cmd_topic', 'wake_cmd')
        self.declare_parameter('sample_rate', 16000)
        
        # Detection parameters
        self.declare_parameter('energy_threshold_ratio', 8.0)  # Spike must be 8x background
        self.declare_parameter('min_clap_duration_ms', 10)  # Minimum duration of spike
        self.declare_parameter('max_clap_duration_ms', 100)  # Maximum duration of spike
        self.declare_parameter('double_clap_min_gap_ms', 100)  # Min gap between claps
        self.declare_parameter('double_clap_max_gap_ms', 500)  # Max gap between claps
        self.declare_parameter('background_adaptation_rate', 0.995)  # Slow adaptation
        
        # Get parameters
        self.audio_topic = self.get_parameter('audio_topic').value
        self.wake_cmd_topic = self.get_parameter('wake_cmd_topic').value
        self.sample_rate = self.get_parameter('sample_rate').value
        
        self.energy_threshold_ratio = self.get_parameter('energy_threshold_ratio').value
        self.min_clap_duration_ms = self.get_parameter('min_clap_duration_ms').value
        self.max_clap_duration_ms = self.get_parameter('max_clap_duration_ms').value
        self.double_clap_min_gap_ms = self.get_parameter('double_clap_min_gap_ms').value
        self.double_clap_max_gap_ms = self.get_parameter('double_clap_max_gap_ms').value
        self.background_adaptation_rate = self.get_parameter('background_adaptation_rate').value
        
        # Convert ms to samples
        self.min_clap_samples = int(self.min_clap_duration_ms * self.sample_rate / 1000)
        self.max_clap_samples = int(self.max_clap_duration_ms * self.sample_rate / 1000)
        
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
        
        # Background noise tracking
        self.background_energy = 0.0001  # Initialize with small value
        self.energy_history = deque(maxlen=50)  # Keep last 50 energy measurements
        
        # Clap detection state
        self.in_spike = False
        self.spike_start_time = None
        self.spike_samples = 0
        self.last_clap_time = None
        self.clap_times = []  # For double-clap detection
        
        # Stats
        self.callback_count = 0
        self.single_claps = 0
        self.double_claps = 0
        
        self.get_logger().info(f"[{get_timestamp()}] Simple energy clap detector initialized")
        self.get_logger().info(f"  Listening on: {self.audio_topic}")
        self.get_logger().info(f"  Energy threshold: {self.energy_threshold_ratio}x background")
        self.get_logger().info(f"  Clap duration: {self.min_clap_duration_ms}-{self.max_clap_duration_ms}ms")
        self.get_logger().info(f"  Double-clap gap: {self.double_clap_min_gap_ms}-{self.double_clap_max_gap_ms}ms")
        
    def detect_clap(self, audio_chunk):
        """Detect claps based on energy spikes"""
        # Calculate energy (RMS squared for more sensitivity to peaks)
        energy = np.mean(audio_chunk**2)
        peak = np.max(np.abs(audio_chunk))
        
        # Update energy history
        self.energy_history.append(energy)
        
        # Update background energy (only from quiet periods)
        if energy < self.background_energy * 2:  # Only adapt when relatively quiet
            self.background_energy = (self.background_adaptation_rate * self.background_energy + 
                                     (1 - self.background_adaptation_rate) * energy)
            # Prevent background from going too low
            self.background_energy = max(self.background_energy, 0.000001)
        
        # Calculate dynamic threshold
        threshold = self.background_energy * self.energy_threshold_ratio
        
        current_time = time.time()
        clap_detected = False
        
        # State machine for spike detection
        if not self.in_spike:
            # Check for spike start
            if energy > threshold and peak > 0.01:  # Also require minimum peak amplitude
                self.in_spike = True
                self.spike_start_time = current_time
                self.spike_samples = len(audio_chunk)
                
                # Log spike start
                ratio = energy / self.background_energy if self.background_energy > 0 else 0
                self.get_logger().debug(f"[{get_timestamp()}] Spike start: energy={energy:.6f}, "
                                      f"bg={self.background_energy:.6f}, ratio={ratio:.1f}x")
        else:
            # In spike - check if it continues or ends
            if energy > threshold * 0.5:  # Use lower threshold to track spike continuation
                self.spike_samples += len(audio_chunk)
            else:
                # Spike ended - check if it was a valid clap
                spike_duration_ms = self.spike_samples * 1000 / self.sample_rate
                
                if self.min_clap_duration_ms <= spike_duration_ms <= self.max_clap_duration_ms:
                    # Valid clap detected!
                    clap_detected = True
                    self.last_clap_time = current_time
                    
                    self.get_logger().info(f"[{get_timestamp()}] ⚡ CLAP! Duration: {spike_duration_ms:.0f}ms, "
                                         f"Peak: {peak:.3f}")
                else:
                    self.get_logger().debug(f"[{get_timestamp()}] Spike rejected: {spike_duration_ms:.0f}ms "
                                          f"(not in {self.min_clap_duration_ms}-{self.max_clap_duration_ms}ms range)")
                
                # Reset spike state
                self.in_spike = False
                self.spike_start_time = None
                self.spike_samples = 0
        
        return clap_detected, energy, peak
        
    def check_double_clap(self):
        """Check if recent claps form a double-clap pattern"""
        current_time = time.time()
        
        # Clean old claps (older than max gap)
        max_age = self.double_clap_max_gap_ms / 1000 * 2
        self.clap_times = [t for t in self.clap_times if current_time - t < max_age]
        
        # Check for double clap
        if len(self.clap_times) >= 2:
            # Check last two claps
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
        
        # Detect claps
        clap_detected, energy, peak = self.detect_clap(audio_float)
        
        if clap_detected:
            self.single_claps += 1
            self.clap_times.append(time.time())
            
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
            # Calculate recent energy stats
            if self.energy_history:
                recent_max = max(self.energy_history)
                recent_avg = np.mean(list(self.energy_history))
            else:
                recent_max = 0
                recent_avg = 0
                
            self.get_logger().info(f"[{get_timestamp()}] Stats: callbacks={self.callback_count}, "
                                 f"bg_energy={self.background_energy:.6f}, "
                                 f"threshold={self.background_energy * self.energy_threshold_ratio:.6f}, "
                                 f"recent_max={recent_max:.6f}, "
                                 f"single={self.single_claps}, double={self.double_claps}")
        
        # Log very loud sounds for debugging
        if peak > 0.1:
            ratio = energy / self.background_energy if self.background_energy > 0 else 0
            self.get_logger().info(f"[{get_timestamp()}] 🔊 Loud sound: peak={peak:.3f}, "
                                 f"energy={energy:.6f}, ratio={ratio:.1f}x")

def main(args=None):
    rclpy.init(args=args)
    
    node = SimpleEnergyClapDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down simple energy clap detector")
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()