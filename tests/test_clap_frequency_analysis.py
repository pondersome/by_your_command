#!/usr/bin/env python3
"""
Frequency analysis test for clap detection
Analyzes the frequency content of audio to understand why claps are missed
"""

import rclpy
from rclpy.node import Node
from audio_common_msgs.msg import AudioStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
from scipy import signal
from datetime import datetime

def get_timestamp():
    """Get formatted timestamp HH:MM:SS.mmm"""
    now = datetime.now()
    return now.strftime("%H:%M:%S.%f")[:-3]

class ClapFrequencyAnalyzer(Node):
    def __init__(self):
        super().__init__('clap_frequency_analyzer')
        
        # Parameters
        self.declare_parameter('audio_topic', '/grunt1/agent/audio')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('energy_threshold', 0.01)  # Minimum energy to analyze
        
        # Get parameters
        self.audio_topic = self.get_parameter('audio_topic').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.energy_threshold = self.get_parameter('energy_threshold').value
        
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
        
        # Buffer for accumulating audio
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_max_size = self.sample_rate * 2  # 2 seconds max
        
        # Stats
        self.callback_count = 0
        self.loud_event_count = 0
        
        self.get_logger().info(f"[{get_timestamp()}] Frequency analyzer initialized")
        self.get_logger().info(f"  Listening on: {self.audio_topic}")
        self.get_logger().info(f"  Sample rate: {self.sample_rate} Hz")
        self.get_logger().info(f"  Energy threshold: {self.energy_threshold}")
        self.get_logger().info("  Make different types of claps to see frequency content!")
        
    def analyze_frequency_content(self, audio_chunk):
        """Analyze frequency content of audio chunk"""
        # Compute FFT
        fft = np.fft.rfft(audio_chunk)
        freqs = np.fft.rfftfreq(len(audio_chunk), 1/self.sample_rate)
        magnitude = np.abs(fft)
        
        # Find dominant frequencies
        # Normalize magnitude
        if np.max(magnitude) > 0:
            magnitude_norm = magnitude / np.max(magnitude)
        else:
            return None
            
        # Find peaks in spectrum
        peaks, properties = signal.find_peaks(magnitude_norm, height=0.1, distance=50)
        
        if len(peaks) == 0:
            return None
            
        # Get top frequencies
        top_indices = peaks[np.argsort(properties['peak_heights'])[-5:]]  # Top 5 peaks
        top_freqs = freqs[top_indices]
        top_mags = magnitude_norm[top_indices]
        
        # Calculate energy in different bands
        bands = {
            'sub-bass': (20, 60),
            'bass': (60, 250),
            'low-mid': (250, 500),
            'mid': (500, 2000),
            'high-mid': (2000, 4000),
            'high': (4000, 8000)
        }
        
        band_energy = {}
        total_energy = np.sum(magnitude**2)
        
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs < high)
            band_energy[band_name] = np.sum(magnitude[band_mask]**2) / total_energy if total_energy > 0 else 0
            
        # Find peak frequency
        peak_freq = freqs[np.argmax(magnitude)]
        
        # Calculate spectral centroid (brightness indicator)
        if np.sum(magnitude) > 0:
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            spectral_centroid = 0
            
        return {
            'peak_freq': peak_freq,
            'spectral_centroid': spectral_centroid,
            'top_freqs': top_freqs,
            'top_mags': top_mags,
            'band_energy': band_energy
        }
        
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
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_float])
        if len(self.audio_buffer) > self.buffer_max_size:
            self.audio_buffer = self.audio_buffer[-self.buffer_max_size:]
        
        # Calculate energy
        rms = np.sqrt(np.mean(audio_float**2))
        peak = np.max(np.abs(audio_float))
        
        # Analyze loud sounds
        if peak > self.energy_threshold:
            self.loud_event_count += 1
            
            # Get a window around the peak for analysis
            if len(self.audio_buffer) > self.sample_rate // 10:  # At least 100ms
                # Get last 100ms for analysis
                analysis_window = self.audio_buffer[-(self.sample_rate // 10):]
                
                # Analyze frequency content
                freq_info = self.analyze_frequency_content(analysis_window)
                
                if freq_info:
                    self.get_logger().info(f"\n[{get_timestamp()}] 🔊 LOUD SOUND DETECTED #{self.loud_event_count}")
                    self.get_logger().info(f"  RMS: {rms:.4f}, Peak: {peak:.4f}")
                    self.get_logger().info(f"  Peak frequency: {freq_info['peak_freq']:.1f} Hz")
                    self.get_logger().info(f"  Spectral centroid: {freq_info['spectral_centroid']:.1f} Hz")
                    
                    # Show band energy distribution
                    self.get_logger().info("  Energy distribution:")
                    for band, energy in freq_info['band_energy'].items():
                        bar = '█' * int(energy * 20)
                        self.get_logger().info(f"    {band:10s}: {bar:20s} {energy*100:.1f}%")
                    
                    # Show top frequencies
                    if len(freq_info['top_freqs']) > 0:
                        self.get_logger().info(f"  Top frequencies: {freq_info['top_freqs'].astype(int)} Hz")
                    
                    # Determine if this would pass different filters
                    peak_f = freq_info['peak_freq']
                    centroid = freq_info['spectral_centroid']
                    
                    # Check against common filter ranges
                    filters_passed = []
                    if 200 <= peak_f <= 3200:  # Original filter
                        filters_passed.append("PASS: Original (200-3200Hz)")
                    if 100 <= peak_f <= 4000:  # Wider filter
                        filters_passed.append("PASS: Wide (100-4000Hz)")
                    if 50 <= peak_f <= 8000:  # Very wide filter
                        filters_passed.append("PASS: Very wide (50-8000Hz)")
                        
                    if not filters_passed:
                        self.get_logger().warning(f"  ⚠️ Would NOT pass any standard bandpass filters!")
                        self.get_logger().warning(f"     Peak at {peak_f:.0f}Hz is outside all ranges")
                    else:
                        for f in filters_passed:
                            self.get_logger().info(f"  ✓ {f}")
                    
                    # Classify the sound
                    if centroid < 500:
                        sound_type = "LOW/MUFFLED (cupped hands?)"
                    elif centroid < 1500:
                        sound_type = "MID-RANGE (normal clap?)"
                    elif centroid < 3000:
                        sound_type = "BRIGHT (tight palm clap?)"
                    else:
                        sound_type = "VERY BRIGHT (sharp/metallic)"
                        
                    self.get_logger().info(f"  Sound type: {sound_type}")
                    self.get_logger().info("-" * 60)
        
        # Periodic stats
        if self.callback_count % 100 == 0:
            self.get_logger().info(f"[{get_timestamp()}] Processed {self.callback_count} callbacks, "
                                 f"detected {self.loud_event_count} loud sounds")

def main(args=None):
    rclpy.init(args=args)
    
    node = ClapFrequencyAnalyzer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down frequency analyzer")
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()