"""
Base Serializer for Multi-Provider LLM Support

Provider-agnostic message serialization with common functionality for
converting ROS messages to LLM API formats.

Author: Karim Virani  
Version: 1.0
Date: August 2025
"""

import base64
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import numpy as np

from ros_ai_bridge import MessageEnvelope
from .websocket_bridge import WebSocketMessageEnvelope


class SerializationError(Exception):
    """Raised when ROS message cannot be serialized for API"""
    pass


class BaseSerializer(ABC):
    """Base serializer for ROS to LLM API conversion"""
    
    def __init__(self):
        """Initialize base serializer"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.serialization_errors = 0
        self.current_utterance_metadata = {}
        self.utterance_contexts = []
    
    # ============================================================
    # Common utility methods (used by all providers)
    # ============================================================
    
    def encode_audio_base64(self, audio_data: Union[list, np.ndarray, bytes]) -> str:
        """
        Convert audio data to base64 string
        
        Args:
            audio_data: Audio as list of ints, numpy array, or bytes
            
        Returns:
            str: Base64 encoded audio string
        """
        try:
            if isinstance(audio_data, list):
                # Convert list to numpy array then to bytes
                pcm_bytes = np.array(audio_data, dtype=np.int16).tobytes()
            elif isinstance(audio_data, np.ndarray):
                # Convert numpy array to bytes
                pcm_bytes = audio_data.astype(np.int16).tobytes()
            elif isinstance(audio_data, bytes):
                # Already bytes
                pcm_bytes = audio_data
            else:
                raise SerializationError(f"Unsupported audio data type: {type(audio_data)}")
                
            return base64.b64encode(pcm_bytes).decode()
            
        except Exception as e:
            self.logger.error(f"Audio encoding failed: {e}")
            self.serialization_errors += 1
            raise SerializationError(f"Audio encoding failed: {e}") from e
    
    def extract_audio_from_ros(self, envelope: Union[MessageEnvelope, WebSocketMessageEnvelope]) -> Optional[np.ndarray]:
        """
        Extract audio data from various ROS message types
        
        Args:
            envelope: Message envelope containing ROS data
            
        Returns:
            Optional[np.ndarray]: Audio data as numpy array, or None if unsupported
        """
        try:
            msg_type = envelope.ros_msg_type
            
            if msg_type == "audio_common_msgs/AudioData":
                # Direct AudioData message
                audio_msg = envelope.raw_data
                if hasattr(audio_msg, 'int16_data'):
                    return np.array(audio_msg.int16_data, dtype=np.int16)
                elif hasattr(audio_msg, 'data'):
                    # Byte data
                    return np.frombuffer(audio_msg.data, dtype=np.int16)
                    
            elif msg_type == "audio_common_msgs/AudioStamped":
                # AudioStamped contains AudioData
                audio_msg = envelope.raw_data.audio
                if hasattr(audio_msg, 'int16_data'):
                    return np.array(audio_msg.int16_data, dtype=np.int16)
                elif hasattr(audio_msg, 'data'):
                    return np.frombuffer(audio_msg.data, dtype=np.int16)
                    
            elif msg_type == "by_your_command/AudioDataUtterance":
                # Custom utterance message
                if hasattr(envelope.raw_data, 'int16_data'):
                    return np.array(envelope.raw_data.int16_data, dtype=np.int16)
                elif hasattr(envelope.raw_data, 'audio_data'):
                    # Sometimes stored as audio_data
                    return np.array(envelope.raw_data.audio_data, dtype=np.int16)
                    
            else:
                self.logger.warning(f"Unsupported audio message type: {msg_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Audio extraction failed: {e}")
            return None
    
    def extract_text_from_ros(self, envelope: Union[MessageEnvelope, WebSocketMessageEnvelope]) -> Optional[str]:
        """
        Extract text data from ROS String messages
        
        Args:
            envelope: Message envelope containing ROS data
            
        Returns:
            Optional[str]: Text string, or None if not a string message
        """
        try:
            if envelope.ros_msg_type == "std_msgs/String":
                if hasattr(envelope.raw_data, 'data'):
                    return envelope.raw_data.data
                elif isinstance(envelope.raw_data, dict) and 'data' in envelope.raw_data:
                    return envelope.raw_data['data']
            return None
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            return None
    
    def extract_utterance_metadata(self, envelope: Union[MessageEnvelope, WebSocketMessageEnvelope]) -> Dict[str, Any]:
        """
        Extract metadata from AudioDataUtterance messages
        
        Args:
            envelope: Message envelope containing utterance data
            
        Returns:
            Dict: Metadata dictionary
        """
        metadata = {}
        
        try:
            if envelope.ros_msg_type == "by_your_command/AudioDataUtterance":
                raw_data = envelope.raw_data
                
                # Extract available metadata fields
                if hasattr(raw_data, 'utterance_id'):
                    metadata['utterance_id'] = raw_data.utterance_id
                if hasattr(raw_data, 'chunk_sequence'):
                    metadata['chunk_sequence'] = raw_data.chunk_sequence
                if hasattr(raw_data, 'is_utterance_end'):
                    metadata['is_utterance_end'] = raw_data.is_utterance_end
                if hasattr(raw_data, 'confidence'):
                    metadata['confidence'] = raw_data.confidence
                if hasattr(raw_data, 'start_time'):
                    metadata['start_time'] = raw_data.start_time
                    
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            
        return metadata
    
    # ============================================================
    # Common serialization interface (wraps provider-specific methods)
    # ============================================================
    
    async def safe_serialize(self, envelope: Union[MessageEnvelope, WebSocketMessageEnvelope]) -> Optional[Dict]:
        """
        Safely serialize any supported message type
        
        Args:
            envelope: Message envelope to serialize
            
        Returns:
            Optional[Dict]: Serialized message or None if unsupported
        """
        try:
            # Route to appropriate serializer based on message type
            if "Audio" in envelope.ros_msg_type:
                return self.serialize_audio(envelope)
            elif envelope.ros_msg_type == "std_msgs/String":
                text = self.extract_text_from_ros(envelope)
                if text and envelope.topic_name and "text" in envelope.topic_name.lower():
                    return self.serialize_text(envelope)
            else:
                self.logger.debug(f"No serializer for message type: {envelope.ros_msg_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Serialization failed: {e}")
            self.serialization_errors += 1
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get serialization statistics"""
        return {
            'serialization_errors': self.serialization_errors,
            'utterance_contexts': len(self.utterance_contexts)
        }
    
    # ============================================================
    # Abstract methods - must be implemented by provider subclasses
    # ============================================================
    
    @abstractmethod
    def serialize_audio(self, envelope: Union[MessageEnvelope, WebSocketMessageEnvelope]) -> Optional[Dict]:
        """
        Convert audio message to provider-specific format
        
        Args:
            envelope: Message envelope containing audio data
            
        Returns:
            Optional[Dict]: Provider-specific message format
        """
        pass
    
    @abstractmethod
    def serialize_text(self, envelope: Union[MessageEnvelope, WebSocketMessageEnvelope]) -> Optional[Dict]:
        """
        Convert text message to provider-specific format
        
        Args:
            envelope: Message envelope containing text data
            
        Returns:
            Optional[Dict]: Provider-specific message format
        """
        pass