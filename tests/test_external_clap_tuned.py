#!/usr/bin/env python3
"""
Test node for external clap-detector library with tunable parameters
Tests the tzursoffer/clapDetection library with ROS audio input
"""

import rclpy
from rclpy.node import Node
from audio_common_msgs.msg import AudioStamped
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import time
from datetime import datetime

# Try to import the clap detector library
try:
    from clapDetector import ClapDetector
    CLAP_DETECTOR_AVAILABLE = True
except ImportError:
    CLAP_DETECTOR_AVAILABLE = False
    print("WARNING: clap-detector library not installed. Install with: pip install clap-detector")

def get_timestamp():
    """Get formatted timestamp HH:MM:SS.mmm"""
    now = datetime.now()
    return now.strftime("%H:%M:%S.%f")[:-3]

class TunedClapTestNode(Node):
    def __init__(self):
        super().__init__('tuned_clap_test_node')
        
        if not CLAP_DETECTOR_AVAILABLE:
            self.get_logger().error("clap-detector library not available. Please install it.")
            return
            
        # Parameters for clap detection
        self.declare_parameter('threshold_bias', 3000)  # Lower threshold for easier detection
        self.declare_parameter('lowcut', 100)  # Wider frequency range
        self.declare_parameter('highcut', 4000)  # Wider frequency range
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('audio_topic', '/grunt1/agent/audio')
        self.declare_parameter('wake_cmd_topic', 'wake_cmd')
        
        # Timing parameters (more forgiving)
        self.declare_parameter('reset_time', 1.0)  # Longer reset time (was 0.35)
        self.declare_parameter('clap_interval', 0.5)  # Much more forgiving interval (was 0.08)
        self.declare_parameter('debounce_time', 0.1)  # Shorter debounce (was 0.15)
        
        # Get parameters
        self.threshold_bias = self.get_parameter('threshold_bias').value
        self.lowcut = self.get_parameter('lowcut').value
        self.highcut = self.get_parameter('highcut').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.audio_topic = self.get_parameter('audio_topic').value
        self.wake_cmd_topic = self.get_parameter('wake_cmd_topic').value
        
        # Timing parameters
        self.reset_time = self.get_parameter('reset_time').value
        self.clap_interval = self.get_parameter('clap_interval').value
        self.debounce_time = self.get_parameter('debounce_time').value
        
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
        
        # Initialize clap detector with custom timing
        self.clap_detector = ClapDetector(
            inputDevice=None,  # External audio source
            rate=self.sample_rate,
            logLevel=10,  # DEBUG level
            bufferLength=512,
            resetTime=self.reset_time,  # Custom reset time
            clapInterval=self.clap_interval,  # Custom clap interval
            debounceTimeFactor=self.debounce_time,  # Custom debounce
            initialVolumeThreshold=1000,  # Lower initial threshold
            volumeAverageFactor=0.95  # Slower adaptation
        )
        
        # Manually initialize for external audio mode
        from collections import deque
        self.clap_detector.resetTimeSamples = int(self.clap_detector.resetTime * self.sample_rate)
        self.clap_detector.clapIntervalSamples = int(self.clap_detector.clapInterval * self.sample_rate)
        self.clap_detector.samplesPerTimePeriod = self.clap_detector.secondsPerTimePeriod * self.sample_rate
        self.clap_detector.audioBuffer = deque(maxlen=int((self.sample_rate * self.clap_detector.audioBufferLength) / self.clap_detector.bufferLength))
        self.clap_detector.currentSampleTime = 0 + int(self.clap_detector.debounceTimeFactor * self.sample_rate)
        
        # Stats tracking
        self.audio_callback_count = 0
        self.single_clap_count = 0
        self.double_clap_count = 0
        self.multi_clap_count = 0
        self.last_detection_time = 0
        
        # Manual double-clap detection backup
        self.manual_clap_times = []
        self.manual_double_threshold = 0.8  # 800ms max between claps
        
        self.get_logger().info(f"[{get_timestamp()}] Tuned clap test node initialized")
        self.get_logger().info(f"  Listening on: {self.audio_topic}")
        self.get_logger().info(f"  Threshold bias: {self.threshold_bias}")
        self.get_logger().info(f"  Bandpass: {self.lowcut}-{self.highcut} Hz")
        self.get_logger().info(f"  Reset time: {self.reset_time}s")
        self.get_logger().info(f"  Clap interval: {self.clap_interval}s")
        self.get_logger().info(f"  Debounce: {self.debounce_time}s")
        
    def check_manual_double_clap(self):
        """Manual double-clap detection as backup"""
        current_time = time.time()
        
        # Clean old claps
        self.manual_clap_times = [t for t in self.manual_clap_times 
                                  if current_time - t < self.manual_double_threshold * 2]
        
        # Check for double clap
        if len(self.manual_clap_times) >= 2:
            for i in range(len(self.manual_clap_times) - 1):
                gap = self.manual_clap_times[i+1] - self.manual_clap_times[i]
                if 0.15 < gap < self.manual_double_threshold:  # Between 150ms and 800ms
                    self.get_logger().info(f"[{get_timestamp()}] 🎯 MANUAL DOUBLE CLAP! Gap: {gap*1000:.0f}ms")
                    self.manual_clap_times = []  # Reset after detection
                    return True
        return False
        
    def audio_callback(self, msg: AudioStamped):
        """Process incoming audio and detect claps"""
        if not CLAP_DETECTOR_AVAILABLE:
            return
            
        self.audio_callback_count += 1
        
        # Convert ROS audio to numpy array
        audio_list = msg.audio.audio_data.int16_data
        if len(audio_list) == 0:
            return
            
        audio_int16 = np.array(audio_list, dtype=np.int16)
        
        # Calculate current audio level
        audio_float = audio_int16.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float**2))
        peak = np.max(np.abs(audio_float))
        
        # Debug every 50 callbacks
        if self.audio_callback_count % 50 == 0:
            self.get_logger().info(f"[{get_timestamp()}] Stats - callbacks: {self.audio_callback_count}, "
                                 f"RMS: {rms:.6f}, peak: {peak:.6f}, "
                                 f"single: {self.single_clap_count}, double: {self.double_clap_count}, "
                                 f"threshold: {self.clap_detector.volumeThreshold:.0f}")
        
        # Log loud sounds for debugging
        if peak > 0.1:  # Significant sound
            self.get_logger().debug(f"[{get_timestamp()}] Loud sound - RMS: {rms:.6f}, peak: {peak:.6f}")
        
        # Run clap detection
        try:
            result = self.clap_detector.run(
                thresholdBias=self.threshold_bias,
                lowcut=self.lowcut,
                highcut=self.highcut,
                audioData=audio_int16
            )
            
            # Check result
            if result:
                current_time = time.time()
                
                # Log all detections (no debouncing for debugging)
                self.get_logger().info(f"[{get_timestamp()}] 🔊 Detection result: {result}")
                
                if len(result) == 1:
                    self.single_clap_count += 1
                    self.manual_clap_times.append(current_time)
                    self.get_logger().info(f"[{get_timestamp()}] 👏 SINGLE CLAP #{self.single_clap_count}")
                    
                    # Check manual double-clap detection
                    if self.check_manual_double_clap():
                        self.double_clap_count += 1
                        # Send wake command
                        wake_msg = Bool()
                        wake_msg.data = True
                        self.wake_pub.publish(wake_msg)
                        self.get_logger().info(f"[{get_timestamp()}] Sent wake command (manual detection)")
                    
                elif len(result) == 2:
                    self.double_clap_count += 1
                    self.get_logger().info(f"[{get_timestamp()}] 👏👏 DOUBLE CLAP #{self.double_clap_count} (library detection)")
                    
                    # Send wake command
                    wake_msg = Bool()
                    wake_msg.data = True
                    self.wake_pub.publish(wake_msg)
                    self.get_logger().info(f"[{get_timestamp()}] Sent wake command")
                    
                else:
                    # Multi-clap pattern
                    self.multi_clap_count += 1
                    self.get_logger().info(f"[{get_timestamp()}] 👏x{sum(result)} Pattern: {result}")
                    
        except Exception as e:
            self.get_logger().error(f"Clap detection error: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    node = TunedClapTestNode()
    
    if CLAP_DETECTOR_AVAILABLE:
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Shutting down tuned clap test node")
    else:
        node.get_logger().error("Cannot run without clap-detector library")
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()