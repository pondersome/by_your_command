"""
OpenAI Realtime API Serializers

Handles conversion between ROS messages and OpenAI Realtime API formats.
Zero-copy ROS message access with efficient serialization.

Author: Karim Virani
Version: 1.0
Date: July 2025
"""

import base64
import logging
from typing import Optional, Dict
import numpy as np

from ros_ai_bridge import MessageEnvelope


class SerializationError(Exception):
    """Raised when ROS message cannot be serialized for API"""
    pass


class OpenAIRealtimeSerializer:
    """Handle ROS → OpenAI Realtime API serialization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.serialization_errors = 0
        
    def serialize_audio_data(self, envelope: MessageEnvelope) -> Optional[Dict]:
        """Convert ROS AudioData to OpenAI format"""
        try:
            if envelope.ros_msg_type == "audio_common_msgs/AudioData":
                # Direct access to ROS message fields
                audio_msg = envelope.raw_data
                pcm_bytes = np.array(audio_msg.int16_data, dtype=np.int16).tobytes()
                base64_audio = base64.b64encode(pcm_bytes).decode()
                
                return {
                    "type": "input_audio_buffer.append",
                    "audio": base64_audio
                }
            elif envelope.ros_msg_type == "audio_common_msgs/AudioStamped":
                # Extract audio from stamped message
                audio_msg = envelope.raw_data.audio
                pcm_bytes = np.array(audio_msg.int16_data, dtype=np.int16).tobytes()
                base64_audio = base64.b64encode(pcm_bytes).decode()
                
                return {
                    "type": "input_audio_buffer.append", 
                    "audio": base64_audio
                }
            else:
                self.logger.warn(f"Unsupported audio message type: {envelope.ros_msg_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Audio serialization failed: {e}")
            self.serialization_errors += 1
            raise SerializationError(f"Audio serialization failed: {e}") from e
            
    def serialize_text_data(self, envelope: MessageEnvelope) -> Optional[Dict]:
        """Convert ROS String to OpenAI text input"""
        try:
            if envelope.ros_msg_type == "std_msgs/String":
                return {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user", 
                        "content": [
                            {
                                "type": "input_text",
                                "text": envelope.raw_data.data
                            }
                        ]
                    }
                }
            else:
                self.logger.warn(f"Unsupported text message type: {envelope.ros_msg_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Text serialization failed: {e}")
            self.serialization_errors += 1
            raise SerializationError(f"Text serialization failed: {e}") from e
    
    async def safe_serialize(self, envelope: MessageEnvelope) -> Optional[Dict]:
        """Safe serialization with error handling"""
        try:
            if envelope.ros_msg_type.startswith("audio_common_msgs/"):
                return self.serialize_audio_data(envelope)
            elif envelope.ros_msg_type == "std_msgs/String":
                return self.serialize_text_data(envelope)
            else:
                self.logger.warn(f"Unsupported message type: {envelope.ros_msg_type}")
                return None
        except SerializationError:
            # Already logged, return None to drop message
            return None
        except Exception as e:
            self.logger.error(f"Unexpected serialization error: {e}")
            self.serialization_errors += 1
            return None
    
    def get_metrics(self) -> Dict:
        """Get serialization metrics"""
        return {
            'serialization_errors': self.serialization_errors
        }