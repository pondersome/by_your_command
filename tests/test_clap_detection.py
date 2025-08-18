#!/usr/bin/env python3
"""
Test script for clap detection functionality
Simulates clap sounds and tests the adaptive clap detector
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from audio_common_msgs.msg import AudioStamped, AudioData
import time
import numpy as np

class ClapDetectionTest(Node):
    def __init__(self):
        super().__init__('clap_detection_test')
        
        # Publisher for audio data (simulate microphone)
        self.audio_pub = self.create_publisher(AudioStamped, 'audio', 10)
        
        # Publisher to control voice_active (start muted for clap test)
        self.voice_active_pub = self.create_publisher(Bool, 'voice_active', 10)
        
        # Subscriber to monitor voice_active changes (from clap detection)
        self.voice_active_sub = self.create_subscription(
            Bool,
            'voice_active',
            self.voice_active_callback,
            10
        )
        
        # Test parameters
        self.sample_rate = 16000
        self.chunk_size = 1024  # ~64ms at 16kHz
        self.test_phase = "initial"
        self.voice_active_state = True
        self.clap_detected_count = 0
        
        self.get_logger().info("Clap Detection Test Node Started")
        self.get_logger().info("This test will:")
        self.get_logger().info("1. Mute the VAD")
        self.get_logger().info("2. Send background noise")
        self.get_logger().info("3. Send single clap (should not wake)")
        self.get_logger().info("4. Send double clap (should wake)")
        self.get_logger().info("5. Send triple clap (should not wake after first double)")
        
        # Start test sequence
        self.timer = self.create_timer(0.1, self.test_timer_callback)
        self.test_start_time = time.time()
        
    def voice_active_callback(self, msg):
        """Monitor voice_active state changes from clap detection"""
        if msg.data != self.voice_active_state:
            self.voice_active_state = msg.data
            state_str = "ACTIVE" if self.voice_active_state else "MUTED"
            if msg.data:  # If waking up
                self.clap_detected_count += 1
                self.get_logger().info(f"🎉 CLAP DETECTION #{self.clap_detected_count}! Voice state: {state_str}")
            else:
                self.get_logger().info(f"Voice state changed to: {state_str}")
    
    def generate_background_noise(self, duration_samples, amplitude=0.01):
        """Generate background noise"""
        return np.random.normal(0, amplitude, duration_samples).astype(np.float32)
    
    def generate_clap(self, amplitude=0.5):
        """Generate a simulated clap sound (sharp transient)"""
        # Create a sharp spike with rapid decay
        clap_duration = int(0.1 * self.sample_rate)  # 100ms
        t = np.linspace(0, 0.1, clap_duration)
        
        # Sharp attack, exponential decay
        envelope = np.exp(-t * 50)
        # Add some high-frequency content typical of claps
        clap = amplitude * envelope * (np.sin(2 * np.pi * 1000 * t) + 
                                      0.5 * np.sin(2 * np.pi * 3000 * t) +
                                      0.3 * np.random.normal(0, 1, len(t)))
        
        return clap.astype(np.float32)
    
    def send_audio_chunk(self, audio_data):
        """Send audio data to the VAD node"""
        # Convert float to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Create AudioStamped message
        msg = AudioStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.audio.audio_data.int16_data = audio_int16.tolist()
        
        self.audio_pub.publish(msg)
    
    def test_timer_callback(self):
        elapsed = time.time() - self.test_start_time
        
        if elapsed < 1.0:
            # Phase 1: Mute the system
            if self.test_phase != "muting":
                self.test_phase = "muting"
                self.get_logger().info("Phase 1: Muting VAD...")
                mute_msg = Bool()
                mute_msg.data = False
                self.voice_active_pub.publish(mute_msg)
        
        elif elapsed < 3.0:
            # Phase 2: Send background noise
            if self.test_phase != "background":
                self.test_phase = "background"
                self.get_logger().info("Phase 2: Sending background noise...")
            
            # Send continuous background noise
            noise = self.generate_background_noise(self.chunk_size, amplitude=0.02)
            self.send_audio_chunk(noise)
        
        elif elapsed < 4.0:
            # Phase 3: Single clap (should not wake)
            if self.test_phase != "single_clap":
                self.test_phase = "single_clap"
                self.get_logger().info("Phase 3: Sending single clap (should NOT wake)...")
                
                clap = self.generate_clap(amplitude=0.3)
                self.send_audio_chunk(clap)
        
        elif elapsed < 6.0:
            # Wait and send more background noise
            noise = self.generate_background_noise(self.chunk_size, amplitude=0.02)
            self.send_audio_chunk(noise)
        
        elif elapsed < 7.0:
            # Phase 4: Double clap (should wake)
            if self.test_phase != "double_clap":
                self.test_phase = "double_clap"
                self.get_logger().info("Phase 4: Sending double clap (SHOULD wake)...")
                
                # First clap
                clap1 = self.generate_clap(amplitude=0.4)
                self.send_audio_chunk(clap1)
                
        elif elapsed < 7.5:
            # Send gap (silence/background)
            if self.test_phase == "double_clap":
                gap = self.generate_background_noise(self.chunk_size, amplitude=0.01)
                self.send_audio_chunk(gap)
        
        elif elapsed < 8.0:
            # Second clap
            if self.test_phase == "double_clap":
                self.test_phase = "double_clap_sent"
                clap2 = self.generate_clap(amplitude=0.4)
                self.send_audio_chunk(clap2)
        
        elif elapsed < 10.0:
            # Phase 5: Verify wake-up
            if self.test_phase != "verify_wake":
                self.test_phase = "verify_wake"
                self.get_logger().info("Phase 5: Verifying wake-up...")
        
        elif elapsed < 11.0:
            # Phase 6: Re-mute for triple clap test
            if self.test_phase != "remute":
                self.test_phase = "remute"
                self.get_logger().info("Phase 6: Re-muting for triple clap test...")
                mute_msg = Bool()
                mute_msg.data = False
                self.voice_active_pub.publish(mute_msg)
        
        elif elapsed < 13.0:
            # Phase 7: Triple clap (should only trigger once)
            if self.test_phase != "triple_clap":
                self.test_phase = "triple_clap"
                self.get_logger().info("Phase 7: Sending triple clap (should only wake ONCE)...")
                
                # Three claps with proper spacing
                claps = [
                    self.generate_clap(amplitude=0.3),
                    self.generate_background_noise(int(0.4 * self.sample_rate), amplitude=0.01),
                    self.generate_clap(amplitude=0.3),
                    self.generate_background_noise(int(0.4 * self.sample_rate), amplitude=0.01),
                    self.generate_clap(amplitude=0.3)
                ]
                
                combined = np.concatenate(claps)
                # Send in chunks
                for i in range(0, len(combined), self.chunk_size):
                    chunk = combined[i:i+self.chunk_size]
                    if len(chunk) > 0:
                        self.send_audio_chunk(chunk)
                        time.sleep(0.05)  # Small delay between chunks
        
        else:
            # Test complete
            if self.test_phase != "complete":
                self.test_phase = "complete"
                
                self.get_logger().info("=== TEST RESULTS ===")
                self.get_logger().info(f"Total clap detections: {self.clap_detected_count}")
                self.get_logger().info(f"Final voice state: {'ACTIVE' if self.voice_active_state else 'MUTED'}")
                
                # Expected: 2 detections (double clap + first pair of triple clap)
                if self.clap_detected_count == 2:
                    self.get_logger().info("✅ PASS: Clap detection working correctly")
                elif self.clap_detected_count == 1:
                    self.get_logger().warning("⚠️  PARTIAL: Only one detection (double clap test)")
                else:
                    self.get_logger().error(f"❌ FAIL: Expected 2 detections, got {self.clap_detected_count}")
                
                self.get_logger().info("Test complete. Shutting down...")
                rclpy.shutdown()

def main():
    rclpy.init()
    
    test_node = ClapDetectionTest()
    
    try:
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        test_node.get_logger().info("Test interrupted by user")
    finally:
        test_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()