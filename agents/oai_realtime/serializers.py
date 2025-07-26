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
import time
from typing import Optional, Dict, Any
import numpy as np

from ros_ai_bridge import MessageEnvelope
from .websocket_bridge import WebSocketMessageEnvelope


class SerializationError(Exception):
    """Raised when ROS message cannot be serialized for API"""
    pass


class OpenAIRealtimeSerializer:
    """Handle ROS → OpenAI Realtime API serialization with metadata support"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.serialization_errors = 0
        self.current_utterance_metadata = {}
        self.utterance_contexts = []
        
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
    
    def serialize_audio_utterance(self, envelope) -> Optional[Dict]:
        """Convert AudioDataUtterance to OpenAI format with metadata preservation"""
        try:
            if envelope.ros_msg_type == "by_your_command/AudioDataUtterance":
                # AudioDataUtterance uses int16_data field, not audio_data
                audio_data = envelope.raw_data.int16_data
                
                # Convert audio data to base64 PCM
                if isinstance(audio_data, list):
                    pcm_bytes = np.array(audio_data, dtype=np.int16).tobytes()
                else:
                    pcm_bytes = audio_data  # Already bytes
                    
                # Check if audio data is empty
                if len(pcm_bytes) == 0:
                    self.logger.warning("⚠️ Empty audio data received - skipping")
                    return None
                    
                base64_audio = base64.b64encode(pcm_bytes).decode()
                
                # OpenAI API message (audio only)
                api_msg = {
                    "type": "input_audio_buffer.append",
                    "audio": base64_audio
                }
                
                # Store metadata separately for context injection
                self.current_utterance_metadata = {
                    "utterance_id": envelope.raw_data.utterance_id,
                    "chunk_sequence": envelope.raw_data.chunk_sequence,
                    "is_utterance_end": envelope.raw_data.is_utterance_end,
                    "timestamp": envelope.timestamp
                }
                
                self.logger.debug(f"Serialized AudioDataUtterance: {envelope.raw_data.utterance_id}")
                return api_msg
            else:
                self.logger.warn(f"Not an AudioDataUtterance: {envelope.ros_msg_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"AudioDataUtterance serialization failed: {e}")
            self.serialization_errors += 1
            raise SerializationError(f"AudioDataUtterance serialization failed: {e}") from e
            
    async def safe_serialize(self, envelope) -> Optional[Dict]:
        """Safe serialization with error handling"""
        try:
            if envelope.ros_msg_type == "by_your_command/AudioDataUtterance":
                return self.serialize_audio_utterance(envelope)
            elif envelope.ros_msg_type.startswith("audio_common_msgs/"):
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
    
    def get_utterance_metadata(self) -> Dict:
        """Get metadata from last processed utterance"""
        return self.current_utterance_metadata.copy()
        
    def add_utterance_context(self, metadata: Dict):
        """Store utterance metadata for session context"""
        self.utterance_contexts.append({
            "utterance_id": metadata.get("utterance_id", ""),
            "confidence": metadata.get("confidence", 0.0), 
            "start_time": metadata.get("start_time", 0.0),
            "chunk_sequence": metadata.get("chunk_sequence", 0),
            "is_utterance_end": metadata.get("is_utterance_end", False),
            "processed_at": time.time()
        })
        
        # Keep only recent utterances (last 10)
        if len(self.utterance_contexts) > 10:
            self.utterance_contexts = self.utterance_contexts[-10:]
            
    def build_context_prompt(self, base_prompt: str) -> str:
        """Inject utterance context into system prompt"""
        if not self.utterance_contexts:
            return base_prompt
            
        recent_contexts = self.utterance_contexts[-3:]  # Last 3 utterances
        context_info = []
        
        for ctx in recent_contexts:
            context_info.append(
                f"Utterance {ctx['utterance_id']}: confidence={ctx['confidence']:.2f}"
            )
            
        context_section = "\n".join([
            "\nRecent speech context:",
            *context_info,
            "Use this context to better understand speech quality and user intent.\n"
        ])
        
        return base_prompt + context_section
    
    def get_metrics(self) -> Dict:
        """Get serialization metrics"""
        return {
            'serialization_errors': self.serialization_errors,
            'utterance_contexts_count': len(self.utterance_contexts),
            'current_utterance_id': self.current_utterance_metadata.get('utterance_id', 'none')
        }