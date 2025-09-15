#!/usr/bin/env python3
"""
Integration test for speech_chunk_recorder with AudioDataUtterance messages
"""
import rclpy
from rclpy.node import Node
from by_your_command.msg import AudioDataUtterance
import numpy as np
import time
import os
import tempfile

class RecorderTester(Node):
    def __init__(self):
        super().__init__('recorder_tester')
        
        # Publisher for test chunks
        self.chunk_pub = self.create_publisher(AudioDataUtterance, 'prompt_voice', 10)
        
        # Timer to send test chunks
        self.timer = self.create_timer(0.1, self.send_test_chunk)  # 10Hz
        
        # Test state
        self.chunk_count = 0
        self.utterance_count = 0
        self.max_chunks_per_utterance = 5
        
        self.get_logger().info('Recorder integration tester started')

    def send_test_chunk(self):
        """Send test audio chunks with utterance metadata"""
        
        # Create test audio data (sine wave)
        sample_rate = 16000
        duration = 0.1  # 100ms chunks
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples)
        frequency = 440 + self.utterance_count * 100  # Different frequency per utterance
        audio_data = (np.sin(2 * np.pi * frequency * t) * 16383).astype(np.int16)
        
        # Create message
        msg = AudioDataUtterance()
        msg.int16_data = audio_data.tolist()
        
        # Set utterance metadata
        if self.chunk_count == 0:
            # New utterance - use current timestamp
            self.current_utterance_id = int(time.time() * 1e9)
            self.get_logger().info(f'Starting test utterance {self.utterance_count + 1} (ID: {self.current_utterance_id})')
        
        msg.utterance_id = self.current_utterance_id
        msg.chunk_sequence = self.chunk_count
        msg.is_utterance_end = (self.chunk_count == self.max_chunks_per_utterance - 1)
        
        # Publish chunk
        self.chunk_pub.publish(msg)
        
        if msg.is_utterance_end:
            self.get_logger().info(f'Completed test utterance {self.utterance_count + 1} with {self.chunk_count + 1} chunks')
        
        # Update counters
        self.chunk_count += 1
        
        if msg.is_utterance_end:
            self.utterance_count += 1
            self.chunk_count = 0
            
            # Stop after 3 test utterances
            if self.utterance_count >= 3:
                self.get_logger().info('Integration test completed - 3 utterances sent')
                self.timer.cancel()
                rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    
    # Create temporary directory for test output
    test_dir = tempfile.mkdtemp(prefix='recorder_test_')
    print(f"\nTest files will be saved to: {test_dir}")
    print("Run the recorder in another terminal with:")
    print(f"ros2 run by_your_command speech_chunk_recorder --ros-args -p output_dir:={test_dir}")
    print("\nThen run this test script to send test utterances.\n")
    
    node = RecorderTester()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        
        # List generated files
        if os.path.exists(test_dir):
            files = os.listdir(test_dir)
            if files:
                print(f"\nGenerated test files in {test_dir}:")
                for f in sorted(files):
                    file_path = os.path.join(test_dir, f)
                    size = os.path.getsize(file_path)
                    print(f"  {f} ({size} bytes)")
            else:
                print(f"\nNo files generated in {test_dir}")
        
        rclpy.shutdown()

if __name__ == '__main__':
    main()