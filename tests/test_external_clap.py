#!/usr/bin/env python3
"""
Test node for external clap-detector library
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

class ExternalClapTestNode(Node):
    def __init__(self):
        super().__init__('external_clap_test_node')
        
        if not CLAP_DETECTOR_AVAILABLE:
            self.get_logger().error("clap-detector library not available. Please install it.")
            return
            
        # Parameters for clap detection
        self.declare_parameter('threshold_bias', 6000)
        self.declare_parameter('lowcut', 200)
        self.declare_parameter('highcut', 3200)
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('audio_topic', '/grunt1/agent/audio')  # Default to your current topic
        self.declare_parameter('wake_cmd_topic', 'wake_cmd')  # Relative topic by default
        
        # Get parameters
        self.threshold_bias = self.get_parameter('threshold_bias').value
        self.lowcut = self.get_parameter('lowcut').value
        self.highcut = self.get_parameter('highcut').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.audio_topic = self.get_parameter('audio_topic').value
        self.wake_cmd_topic = self.get_parameter('wake_cmd_topic').value
        
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
        
        # Publisher for wake signal (if we want to test waking up VAD)
        self.wake_pub = self.create_publisher(Bool, self.wake_cmd_topic, qos_profile=qos)
        
        # Initialize clap detector without internal audio (we'll feed it externally)
        self.clap_detector = ClapDetector(
            inputDevice=None,  # None means external audio source
            rate=self.sample_rate,
            logLevel=10,  # DEBUG level
            bufferLength=512  # Match typical audio chunk size
        )
        
        # Manually initialize attributes that are normally set in initAudio()
        # These are needed for external audio mode but not initialized when inputDevice=None
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
        self.last_detection_time = 0
        
        self.get_logger().info(f"[{get_timestamp()}] External clap test node initialized")
        self.get_logger().info(f"  Listening on: {self.audio_topic}")
        self.get_logger().info(f"  Publishing wake to: {self.wake_cmd_topic}")
        self.get_logger().info(f"  Sample rate: {self.sample_rate}")
        self.get_logger().info(f"  Threshold bias: {self.threshold_bias}")
        self.get_logger().info(f"  Bandpass filter: {self.lowcut}-{self.highcut} Hz")
        
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
        
        # Debug every 100 callbacks (~3.2 seconds at 16kHz with 512 samples)
        if self.audio_callback_count % 100 == 0:
            audio_float = audio_int16.astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(audio_float**2))
            self.get_logger().info(f"[{get_timestamp()}] Audio stats - callbacks: {self.audio_callback_count}, "
                                 f"RMS: {rms:.6f}, single claps: {self.single_clap_count}, "
                                 f"double claps: {self.double_clap_count}")
        
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
                # Debounce - ignore if too close to last detection
                if current_time - self.last_detection_time < 0.5:
                    return
                    
                self.last_detection_time = current_time
                
                if len(result) == 1:
                    self.single_clap_count += 1
                    self.get_logger().info(f"[{get_timestamp()}] 👏 SINGLE CLAP DETECTED! (#{self.single_clap_count})")
                    
                elif len(result) == 2:
                    self.double_clap_count += 1
                    self.get_logger().info(f"[{get_timestamp()}] 👏👏 DOUBLE CLAP DETECTED! (#{self.double_clap_count})")
                    
                    # Send wake command on double clap
                    wake_msg = Bool()
                    wake_msg.data = True
                    self.wake_pub.publish(wake_msg)
                    self.get_logger().info(f"[{get_timestamp()}] Sent wake command")
                    
                else:
                    # Multi-clap pattern
                    self.get_logger().info(f"[{get_timestamp()}] 👏x{len(result)} Multi-clap pattern detected: {result}")
                    
        except Exception as e:
            self.get_logger().error(f"Clap detection error: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    node = ExternalClapTestNode()
    
    if CLAP_DETECTOR_AVAILABLE:
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Shutting down external clap test node")
    else:
        node.get_logger().error("Cannot run without clap-detector library")
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()