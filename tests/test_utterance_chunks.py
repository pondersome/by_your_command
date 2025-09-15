#!/usr/bin/env python3
"""
Test script to demonstrate utterance ID and end-of-utterance detection
"""
import rclpy
from rclpy.node import Node
from by_your_command.msg import AudioDataUtterance
from std_msgs.msg import Bool

class UtteranceChunkTester(Node):
    def __init__(self):
        super().__init__('utterance_chunk_tester')
        
        # Subscribe to both voice activity and enhanced voice chunks
        self.create_subscription(Bool, 'voice_activity', self.voice_activity_callback, 10)
        self.create_subscription(AudioDataUtterance, 'prompt_voice', self.chunk_callback, 10)
        
        # Track utterance state
        self.current_utterance_id = None
        self.chunk_count = 0
        
        self.get_logger().info('Utterance chunk tester started. Listening for voice chunks...')

    def voice_activity_callback(self, msg):
        """Handle voice activity state changes"""
        if msg.data:
            self.get_logger().info('🎤 Voice activity detected')
        else:
            self.get_logger().info('🔇 Voice activity ended')

    def chunk_callback(self, msg):
        """Handle enhanced audio chunks with utterance metadata"""
        # Check if this is a new utterance
        if self.current_utterance_id != msg.utterance_id:
            if self.current_utterance_id is not None:
                self.get_logger().info(f'📝 Utterance {self.current_utterance_id} finished with {self.chunk_count} chunks')
            
            self.current_utterance_id = msg.utterance_id
            self.chunk_count = 0
            self.get_logger().info(f'🆕 New utterance started: ID={msg.utterance_id}')
        
        self.chunk_count += 1
        
        # Log chunk details
        audio_length = len(msg.int16_data)
        duration = audio_length / 16000.0  # Assuming 16kHz sample rate
        
        status = "END" if msg.is_utterance_end else "CHUNK"
        self.get_logger().info(
            f'📊 {status} | Utterance: {msg.utterance_id} | '
            f'Chunk: {msg.chunk_sequence} | '
            f'Samples: {audio_length} | '
            f'Duration: {duration:.2f}s'
        )
        
        if msg.is_utterance_end:
            self.get_logger().info(f'✅ Utterance {msg.utterance_id} completed')

def main(args=None):
    rclpy.init(args=args)
    node = UtteranceChunkTester()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()