#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import whisper
import openai

class InteractionNode(Node):
    def __init__(self):
        super().__init__('interaction_node')
        self.declare_parameter('openai_api_key', '')
        key = self.get_parameter('openai_api_key').get_parameter_value().string_value
        openai.api_key = key
        self.get_logger().info('OpenAI API key set')
        self.model = whisper.load_model('base')
        self.get_logger().info('Whisper model loaded')
        # TODO: subscribe to VAD topics and process audio chunks

    def process_audio(self, audio_chunk):
        # TODO: transcribe and call LLM
        text = self.model.transcribe(audio_chunk)['text']
        self.get_logger().info(f'Transcribed: {text}')
        # TODO: send to LLM via openai.ChatCompletion


def main(args=None):
    rclpy.init(args=args)
    node = InteractionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
