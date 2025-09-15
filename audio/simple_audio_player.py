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
from collections import deque


class SimpleAudioPlayer(Node):
    def __init__(self):
        super().__init__('simple_audio_player')
        
        # Parameters - updated to new naming convention
        self.declare_parameter('topic', '/response_voice')  # was /audio_out
        self.declare_parameter('sample_rate', 16000)  # Default to 16kHz
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
        
        # Remove artificial delay - let AEC work with natural acoustic delay
        # self.delay_buffer = deque()
        # self.playback_delay_ms = 0
        # self.playback_delay_samples = 0
        
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
            'assistant_speaking',  # Relative topic name - will be in node's namespace
            10
        )
        
        # Subscribe to interruption signal to clear audio queue
        self.interruption_sub = self.create_subscription(
            Bool,
            'interruption_signal',  # Relative topic name - will be in node's namespace
            self.interruption_callback,
            10
        )
        
        # Start playback thread
        self.playback_thread = threading.Thread(target=self.playback_loop, daemon=True)
        self.playback_thread.start()
        
        self.get_logger().info(
            f"Simple audio player started on topic {self.topic} "
            f"({self.sample_rate}Hz, {self.channels} channel(s), device {self.device})"
        )
        if self.sample_rate != 16000:
            self.get_logger().warning(f"WARNING: Sample rate is {self.sample_rate}Hz but test expects 16000Hz!")
        
        # Test audio device
        try:
            test_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.device,
                frames_per_buffer=1024
            )
            # Play a short silent test
            test_data = np.zeros(1024, dtype=np.int16)
            test_stream.write(test_data.tobytes())
            test_stream.close()
            self.get_logger().info("Audio device test successful")
        except Exception as e:
            self.get_logger().error(f"Audio device test failed: {e}")
        
        self.msg_count = 0
        
    def audio_callback(self, msg: AudioData):
        """Handle incoming audio data"""
        try:
            # Get audio data
            if msg.int16_data:
                # Convert int16 list to numpy array
                audio_array = np.array(msg.int16_data, dtype=np.int16)
                
                # Add to queue directly - no artificial delay
                try:
                    self.audio_queue.put_nowait(audio_array)
                    
                    # Start playback if not already playing
                    if not self.playing:
                        self.get_logger().info(f"Starting playback, queue size: {self.audio_queue.qsize()}")
                        self.start_playback()
                        
                except queue.Full:
                    self.get_logger().warning("Audio queue full, dropping chunk")
                    
                self.msg_count += 1
                if self.msg_count == 1:
                    self.get_logger().info(f"First audio chunk received! Size: {len(audio_array)} samples ({len(audio_array)/self.sample_rate:.3f}s)")
                    self.get_logger().info(f"Queue empty: {self.audio_queue.empty()}, Playing: {self.playing}")
                    # Don't assume chunk size - just log what we get
                    if len(audio_array) > self.sample_rate * 2:  # More than 2 seconds
                        self.get_logger().warning(f"Very large chunk: {len(audio_array)} samples ({len(audio_array)/self.sample_rate:.2f}s @ {self.sample_rate}Hz)")
                elif self.msg_count <= 10:  # Log first 10 chunks
                    self.get_logger().info(f"Audio chunk {self.msg_count}: {len(audio_array)} samples")
                elif self.msg_count % 10 == 0:
                    self.get_logger().info(f"Received {self.msg_count} audio chunks, queue size: {self.audio_queue.qsize()}")
                    
        except Exception as e:
            self.get_logger().error(f"Error in audio callback: {e}")
            
    def interruption_callback(self, msg: Bool):
        """Handle interruption signal to clear audio queue"""
        if msg.data:
            self.clear_audio_queue()
            self.get_logger().info("⚡ Received interruption signal - cleared audio queue")
    
    def clear_audio_queue(self):
        """Clear all pending audio from the queue and force stop playback"""
        queue_size = self.audio_queue.qsize()
        if queue_size > 0:
            # Clear the queue
            with self.audio_queue.mutex:
                self.audio_queue.queue.clear()
                
            self.get_logger().info(f"🗑️ Cleared {queue_size} audio chunks from queue")
        
        # Force stop current playback immediately, even if queue was empty
        if self.playing:
            self.force_stop_playback()
    
    def force_stop_playback(self):
        """Immediately stop audio playback and clear PyAudio buffers"""
        if not self.playing:
            return
            
        try:
            # More aggressive stopping - abort instead of stop
            if hasattr(self, 'stream') and self.stream:
                self.stream.abort()  # Immediately stop, don't wait for buffer to drain
                self.stream.close()
                
            self.playing = False
            # Signal that assistant stopped speaking
            self.speaking_pub.publish(Bool(data=False))
            self.get_logger().info("🛑 FORCE stopped audio playback - cleared PyAudio buffers")
        except Exception as e:
            self.get_logger().error(f"Error force stopping playback: {e}")
            # Set playing to False anyway
            self.playing = False
            
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
        max_silence = 500  # ~5 seconds at 100Hz - longer for test mode
        
        while True:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=0.01)
                
                # Play audio if stream is active
                if self.playing and hasattr(self, 'stream'):
                    try:
                        self.stream.write(audio_chunk.tobytes())
                        self.get_logger().info(f"Played audio chunk, size: {len(audio_chunk)} samples")
                    except Exception as e:
                        self.get_logger().error(f"Error writing to stream: {e}")
                elif self.playing:
                    self.get_logger().warning("Playing but no stream!")
                else:
                    self.get_logger().warning(f"Not playing, chunk dropped")
                    
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