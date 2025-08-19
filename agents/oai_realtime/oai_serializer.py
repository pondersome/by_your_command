"""
OpenAI Realtime API Serializer

OpenAI-specific implementation of the base serializer for converting
ROS messages to OpenAI Realtime API format.

Author: Karim Virani
Version: 1.0
Date: August 2025
"""

import logging
from typing import Optional, Dict, Any, Union

from ros_ai_bridge import MessageEnvelope
from ..common.websocket_bridge import WebSocketMessageEnvelope
from ..common.base_serializer import BaseSerializer, SerializationError


class OpenAISerializer(BaseSerializer):
    """OpenAI-specific message serialization"""
    
    def __init__(self):
        """Initialize OpenAI serializer"""
        super().__init__()
        
    def serialize_audio(self, envelope: Union[MessageEnvelope, WebSocketMessageEnvelope]) -> Optional[Dict]:
        """
        Convert audio message to OpenAI format
        
        Args:
            envelope: Message envelope containing audio data
            
        Returns:
            Optional[Dict]: OpenAI input_audio_buffer.append message
        """
        try:
            # Extract audio data using base class method
            audio_data = self.extract_audio_from_ros(envelope)
            
            if audio_data is None:
                self.logger.warning(f"Could not extract audio from {envelope.ros_msg_type}")
                return None
            
            # Convert to base64 using base class method
            base64_audio = self.encode_audio_base64(audio_data)
            
            # Format for OpenAI Realtime API
            openai_msg = {
                "type": "input_audio_buffer.append",
                "audio": base64_audio
            }
            
            # If this is an utterance message, store metadata for later use
            if envelope.ros_msg_type == "by_your_command/AudioDataUtterance":
                metadata = self.extract_utterance_metadata(envelope)
                if metadata:
                    self.current_utterance_metadata = metadata
                    self.logger.debug(f"Stored utterance metadata: {metadata}")
            
            return openai_msg
            
        except Exception as e:
            self.logger.error(f"OpenAI audio serialization failed: {e}")
            self.serialization_errors += 1
            raise SerializationError(f"OpenAI audio serialization failed: {e}") from e
    
    def serialize_text(self, envelope: Union[MessageEnvelope, WebSocketMessageEnvelope]) -> Optional[Dict]:
        """
        Convert text message to OpenAI conversation item format
        
        Args:
            envelope: Message envelope containing text data
            
        Returns:
            Optional[Dict]: OpenAI conversation.item.create message
        """
        try:
            # Extract text using base class method
            text = self.extract_text_from_ros(envelope)
            
            if not text:
                self.logger.warning("No text found in message")
                return None
            
            # Format for OpenAI Realtime API
            openai_msg = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": text
                        }
                    ]
                }
            }
            
            return openai_msg
            
        except Exception as e:
            self.logger.error(f"OpenAI text serialization failed: {e}")
            self.serialization_errors += 1
            raise SerializationError(f"OpenAI text serialization failed: {e}") from e
    
    def create_response_trigger(self) -> Dict[str, str]:
        """
        Create OpenAI response.create message
        
        Returns:
            Dict: Response trigger message
        """
        return {"type": "response.create"}
    
    def create_response_cancel(self) -> Dict[str, str]:
        """
        Create OpenAI response.cancel message
        
        Returns:
            Dict: Response cancel message
        """
        return {"type": "response.cancel"}
    
    def create_audio_buffer_clear(self) -> Dict[str, str]:
        """
        Create OpenAI output_audio_buffer.clear message
        
        Returns:
            Dict: Audio buffer clear message
        """
        return {"type": "output_audio_buffer.clear"}
    
    def create_conversation_truncate(self, item_id: str, audio_end_ms: int = 0) -> Dict[str, Any]:
        """
        Create OpenAI conversation.item.truncate message
        
        Args:
            item_id: ID of conversation item to truncate
            audio_end_ms: End position in milliseconds
            
        Returns:
            Dict: Conversation truncate message
        """
        return {
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": 0,
            "audio_end_ms": audio_end_ms
        }
    
    def create_audio_buffer_commit(self) -> Dict[str, str]:
        """
        Create OpenAI input_audio_buffer.commit message
        
        Returns:
            Dict: Audio buffer commit message
        """
        return {"type": "input_audio_buffer.commit"}
    
    def get_utterance_metadata(self) -> Dict:
        """Get metadata from last processed utterance"""
        return self.current_utterance_metadata.copy() if hasattr(self, 'current_utterance_metadata') else {}
    
    def add_utterance_context(self, metadata: Dict):
        """Store utterance metadata for session context"""
        if not hasattr(self, 'utterance_contexts'):
            self.utterance_contexts = []
        self.utterance_contexts.append({
            "utterance_id": metadata.get("utterance_id", ""),
            "confidence": metadata.get("confidence", 0.0), 
            "start_time": metadata.get("start_time", 0.0),
            "chunk_sequence": metadata.get("chunk_sequence", 0),
            "duration": metadata.get("duration", 0.0)
        })