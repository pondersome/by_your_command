#!/usr/bin/env python3
"""
Simple Audio Player for AudioData Messages

A minimal audio player that plays AudioData messages directly without
needing AudioStamped format. Designed for OpenAI Realtime API output.

Author: Karim Virani
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from audio_common_msgs.msg import AudioData
from std_msgs.msg import Bool
import pyaudio
import numpy as np
import threading
import queue


class SimpleAudioPlayer(Node):
    def __init__(self):
        super().__init__('simple_audio_player')
        
        # Parameters
        self.declare_parameter('topic', '/audio_out')
        self.declare_parameter('sample_rate', 24000)
        self.declare_parameter('channels', 1)
        self.declare_parameter('device', -1)
        
        # Get parameters
        self.topic = self.get_parameter('topic').value
        self.sample_rate = self.get_parameter('sample_rate').value
        self.channels = self.get_parameter('channels').value
        self.device = self.get_parameter('device').value
        
        # Audio setup
        self.p = pyaudio.PyAudio()
        
        # Use default device if -1
        if self.device == -1:
            self.device = self.p.get_default_output_device_info()['index']
            
        # Audio queue for smooth playback
        self.audio_queue = queue.Queue(maxsize=500)  # Increased for better buffering
        self.playing = False
        
        # QoS profile
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        
        # Subscribe to audio topic
        self.subscription = self.create_subscription(
            AudioData,
            self.topic,
            self.audio_callback,
            qos
        )
        
        # Publisher to signal when assistant is speaking (for echo suppression)
        self.speaking_pub = self.create_publisher(
            Bool,
            '/assistant_speaking',
            10
        )
        
        # Start playback thread
        self.playback_thread = threading.Thread(target=self.playback_loop, daemon=True)
        self.playback_thread.start()
        
        self.get_logger().info(
            f"Simple audio player started on topic {self.topic} "
            f"({self.sample_rate}Hz, {self.channels} channel(s), device {self.device})"
        )
        
        self.msg_count = 0
        
    def audio_callback(self, msg: AudioData):
        """Handle incoming audio data"""
        try:
            # Get audio data
            if msg.int16_data:
                # Convert int16 list to numpy array
                audio_array = np.array(msg.int16_data, dtype=np.int16)
                
                # Add to queue (non-blocking)
                try:
                    self.audio_queue.put_nowait(audio_array)
                    
                    # Start playback if not already playing
                    if not self.playing:
                        self.start_playback()
                        
                except queue.Full:
                    self.get_logger().warning("Audio queue full, dropping chunk")
                    
                self.msg_count += 1
                if self.msg_count == 1:
                    self.get_logger().info(f"First audio chunk received! Size: {len(audio_array)} samples")
                elif self.msg_count % 10 == 0:
                    self.get_logger().info(f"Received {self.msg_count} audio chunks, queue size: {self.audio_queue.qsize()}")
                    
        except Exception as e:
            self.get_logger().error(f"Error in audio callback: {e}")
            
    def start_playback(self):
        """Start audio playback stream"""
        if self.playing:
            return
            
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.device,
                frames_per_buffer=1024
            )
            self.playing = True
            # Signal that assistant is speaking
            self.speaking_pub.publish(Bool(data=True))
            self.get_logger().info("Started audio playback - Assistant speaking")
        except Exception as e:
            self.get_logger().error(f"Failed to start playback: {e}")
            
    def stop_playback(self):
        """Stop audio playback stream"""
        if not self.playing:
            return
            
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.playing = False
            # Signal that assistant stopped speaking
            self.speaking_pub.publish(Bool(data=False))
            self.get_logger().info("Stopped audio playback - Assistant finished")
        except Exception as e:
            self.get_logger().error(f"Error stopping playback: {e}")
            
    def playback_loop(self):
        """Background thread for audio playback"""
        silence_count = 0
        max_silence = 200  # ~2 seconds at 100Hz - increased to avoid cutting off
        
        while True:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=0.01)
                
                # Play audio if stream is active
                if self.playing and hasattr(self, 'stream'):
                    self.stream.write(audio_chunk.tobytes())
                    if self.msg_count <= 10 or self.msg_count % 100 == 0:
                        self.get_logger().info(f"Playing audio chunk {self.msg_count}, size: {len(audio_chunk)} samples")
                    
                silence_count = 0
                
            except queue.Empty:
                # No audio data
                if self.playing:  # Only count silence when playing
                    silence_count += 1
                    
                    # Stop playback after extended silence
                    if silence_count > max_silence:
                        self.stop_playback()
                        silence_count = 0
                    
            except Exception as e:
                self.get_logger().error(f"Playback error: {e}")
                
    def destroy_node(self):
        """Cleanup on shutdown"""
        self.stop_playback()
        self.p.terminate()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SimpleAudioPlayer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()