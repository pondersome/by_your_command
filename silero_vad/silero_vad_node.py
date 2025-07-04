#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class SileroVADNode(Node):
    def __init__(self):
        super().__init__('silero_vad_node')
        self.declare_parameter('silero_model', 'silero_vad.jit')
        model = self.get_parameter('silero_model').get_parameter_value().string_value
        self.get_logger().info(f'Using Silero VAD model: {model}')
        # TODO: subscribe to audio stream and publish VAD events

def main(args=None):
    rclpy.init(args=args)
    node = SileroVADNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
