#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from audio_common_msgs.msg import AudioStamped

import numpy as np
import collections
import time

# Silero VAD imports
from silero_vad import load_silero_vad, VADIterator, get_speech_timestamps

# CONFIGURABLE PARAMETERS
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 20  # duration per frame
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
MAX_BUFFER_SECONDS = 5  # seconds of audio to keep for pre-roll
PRE_ROLL_MS = 300       # how much to go back before VAD triggers
UTTERANCE_TIMEOUT_SEC = 1.0  # silence duration to consider end of utterance
UTTERANCE_CHUNK_SEC = 2.0    # max duration for sub-chunking; 0 to publish only on utterance end

class SileroVADNode(Node):
    def __init__(self):
        super().__init__('silero_vad_node')
        # Declare configurable parameters
        self.declare_parameter('sample_rate', SAMPLE_RATE)
        self.declare_parameter('frame_duration_ms', FRAME_DURATION_MS)
        self.declare_parameter('max_buffer_seconds', MAX_BUFFER_SECONDS)
        self.declare_parameter('pre_roll_ms', PRE_ROLL_MS)
        self.declare_parameter('utterance_timeout_sec', UTTERANCE_TIMEOUT_SEC)
        self.declare_parameter('utterance_chunk_sec', UTTERANCE_CHUNK_SEC)
        # Load parameter values
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().integer_value
        self.frame_duration_ms = self.get_parameter('frame_duration_ms').get_parameter_value().integer_value
        self.max_buffer_seconds = self.get_parameter('max_buffer_seconds').get_parameter_value().integer_value
        self.pre_roll_ms = self.get_parameter('pre_roll_ms').get_parameter_value().integer_value
        self.utterance_timeout_sec = self.get_parameter('utterance_timeout_sec').get_parameter_value().double_value
        self.utterance_chunk_sec = self.get_parameter('utterance_chunk_sec').get_parameter_value().double_value
        # Subscriptions and publishers
        # QoS profile for audio streaming
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        self.create_subscription(AudioStamped, 'audio', self.audio_callback, qos_profile=qos)
        self.voice_pub = self.create_publisher(Bool, 'voice_activity', qos_profile=qos)
        self.chunk_pub = self.create_publisher(AudioData, 'speech_chunks', qos_profile=qos)

        # Initialize Silero VAD model and iterator
        self.model = load_silero_vad()
        self.vad_iterator = VADIterator(self.model, sampling_rate=self.sample_rate)

        # Buffers for audio and timestamps
        max_frames = int(MAX_BUFFER_SECONDS * 1000 / FRAME_DURATION_MS)
        self.frame_buffer = collections.deque(maxlen=max_frames)
        self.time_buffer = collections.deque(maxlen=max_frames)

        # Utterance tracking state
        self.in_utterance = False
        self.last_voice_time = None
        self.utterance_start_time = None

    def audio_callback(self, msg: AudioStamped):
        # Convert incoming AudioStamped to float32 samples
        # Extract int16 samples from AudioStamped
        audio_list = msg.audio.audio_data.int16_data
        audio_int16 = np.array(audio_list, dtype=np.int16)
        if audio_int16.size < FRAME_SIZE:
            return  # ignore partial frames
        audio_bytes = audio_int16.tobytes()
        audio_float = audio_int16.astype(np.float32) / 32768.0
        if audio_float.size < FRAME_SIZE:
            return  # ignore partial frames

        # Take one frame for VAD
        frame = audio_float[:FRAME_SIZE]

        # Run VAD iterator on numpy array
        speech_activity = bool(self.vad_iterator(frame))
        now = time.time()

        # Buffer frame and timestamp for pre-roll
        self.frame_buffer.append(audio_bytes)
        self.time_buffer.append(now)

        # Voice detection logic
        if speech_activity:
            self.last_voice_time = now
            if not self.in_utterance:
                # Speech started
                self.in_utterance = True
                self.utterance_start_time = now
                self.get_logger().info('Voice detected. Starting utterance.')
                self.voice_pub.publish(Bool(data=True))
                # If not chunking, publish full utterance immediately
                if self.utterance_chunk_sec == 0:
                    self.publish_chunk()
            elif self.utterance_chunk_sec > 0 and now - self.utterance_start_time >= self.utterance_chunk_sec:
                # Interim chunk reached
                self.get_logger().info('Max chunk duration reached. Publishing interim chunk.')
                self.utterance_start_time = now
                self.publish_chunk()
        else:
            # No speech in this frame
            if self.in_utterance and (now - self.last_voice_time > self.utterance_timeout_sec):
                # End of utterance
                self.in_utterance = False
                self.get_logger().info('Voice ended. Publishing final chunk.')
                self.voice_pub.publish(Bool(data=False))
                self.publish_chunk()
                # Reset VAD internal state for next utterance
                self.vad_iterator.reset_states()
                # Clear buffers after utterance
                self.frame_buffer.clear()
                self.time_buffer.clear()

    def publish_chunk(self):
        # Determine start index using PRE_ROLL_MS
        now = time.time()
        pre_roll_sec = self.pre_roll_ms / 1000.0
        start_index = 0
        for i, t in enumerate(self.time_buffer):
            if now - t < pre_roll_sec:
                start_index = max(0, i)
                break

        # Concatenate buffered frames into one audio bytes blob
        full_audio = b''.join(list(self.frame_buffer)[start_index:])
        duration_sec = len(full_audio) / 2 / self.sample_rate
        self.get_logger().info(f'Chunk extracted: {duration_sec:.2f} sec')

        # Publish as AudioData message
        chunk_msg = AudioData()
        chunk_msg.data = list(full_audio)
        self.chunk_pub.publish(chunk_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SileroVADNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
