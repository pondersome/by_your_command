#!/usr/bin/env python3
"""
File Audio Publisher

Reads a WAV file and publishes it as AudioData messages,
simulating real-time playback for AEC testing.

Author: Assistant
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from audio_common_msgs.msg import AudioData
import wave
import numpy as np
import time


class FileAudioPublisher(Node):
    def __init__(self):
        super().__init__('file_audio_publisher')
        
        # Parameters
        self.declare_parameter('file_path', '')
        self.declare_parameter('chunk_size', 320)  # 20ms at 16kHz
        self.declare_parameter('topic', '/voice_playback')
        self.declare_parameter('loop', False)
        
        # Get parameters
        self.file_path = self.get_parameter('file_path').value
        self.chunk_size = self.get_parameter('chunk_size').value
        self.topic = self.get_parameter('topic').value
        self.loop = self.get_parameter('loop').value
        
        if not self.file_path:
            self.get_logger().error("No file_path specified!")
            return
            
        # QoS profile
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        
        # Publisher
        self.publisher = self.create_publisher(
            AudioData,
            self.topic,
            qos
        )
        
        # Open WAV file
        try:
            self.wav = wave.open(self.file_path, 'rb')
            self.sample_rate = self.wav.getframerate()
            self.channels = self.wav.getnchannels()
            self.sample_width = self.wav.getsampwidth()
            self.n_frames = self.wav.getnframes()
            
            self.get_logger().info(
                f"Opened {self.file_path}: {self.sample_rate}Hz, "
                f"{self.channels} channel(s), {self.n_frames} frames"
            )
            
            # Calculate chunk interval
            self.chunk_interval = self.chunk_size / self.sample_rate
            
            # Start publishing timer
            self.timer = self.create_timer(self.chunk_interval, self.publish_chunk)
            self.start_time = time.time()
            
        except Exception as e:
            self.get_logger().error(f"Failed to open file: {e}")
            
    def publish_chunk(self):
        """Publish next chunk of audio"""
        try:
            # Read chunk
            frames = self.wav.readframes(self.chunk_size)
            
            if len(frames) == 0:
                if self.loop:
                    # Restart from beginning
                    self.wav.rewind()
                    frames = self.wav.readframes(self.chunk_size)
                else:
                    # End of file
                    self.get_logger().info("Reached end of file")
                    self.timer.cancel()
                    return
                    
            # Convert to int16 array
            if self.sample_width == 2:
                audio_array = np.frombuffer(frames, dtype=np.int16)
            else:
                self.get_logger().error(f"Unsupported sample width: {self.sample_width}")
                return
                
            # Handle stereo to mono conversion if needed
            if self.channels == 2:
                # Mix stereo to mono
                audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
                
            # Create and publish message
            msg = AudioData()
            msg.int16_data = audio_array.tolist()
            
            self.publisher.publish(msg)
            
            # Log timing info periodically
            elapsed = time.time() - self.start_time
            if int(elapsed) % 1 == 0 and int(elapsed * 10) % 10 == 0:
                file_position = self.wav.tell() / self.sample_rate
                self.get_logger().info(
                    f"Publishing: elapsed={elapsed:.1f}s, file_pos={file_position:.1f}s, "
                    f"chunk_size={len(audio_array)}"
                )
                
        except Exception as e:
            self.get_logger().error(f"Error publishing chunk: {e}")
            
    def destroy_node(self):
        """Cleanup on shutdown"""
        if hasattr(self, 'wav'):
            self.wav.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FileAudioPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()