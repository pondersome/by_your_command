"""
Gemini Live API Serializer

Gemini-specific implementation of the base serializer for converting
ROS messages to Gemini Live API format.

Author: Karim Virani
Version: 1.0
Date: August 2025
"""

import logging
from typing import Optional, Dict, Any, Union
import numpy as np

from ros_ai_bridge import MessageEnvelope
from ..common.websocket_bridge import WebSocketMessageEnvelope
from ..common.base_serializer import BaseSerializer, SerializationError


class GeminiSerializer(BaseSerializer):
    """Gemini Live-specific message serialization"""
    
    def __init__(self):
        """Initialize Gemini serializer"""
        super().__init__()
        
    def serialize_audio(self, envelope: Union[MessageEnvelope, WebSocketMessageEnvelope]) -> Optional[bytes]:
        """
        Convert audio message to Gemini format
        
        Gemini Live expects raw PCM16 audio bytes, not base64 or JSON messages.
        This is simpler than OpenAI which requires buffer management.
        
        Args:
            envelope: Message envelope containing audio data
            
        Returns:
            Optional[bytes]: Raw PCM16 audio bytes for Gemini
        """
        try:
            # Extract audio data using base class method
            audio_data = self.extract_audio_from_ros(envelope)
            
            if audio_data is None:
                self.logger.warning(f"Could not extract audio from {envelope.ros_msg_type}")
                return None
            
            # Gemini expects raw PCM16 bytes (16-bit signed integers)
            # Convert numpy array to bytes if needed
            if isinstance(audio_data, (list, tuple)):
                audio_data = np.array(audio_data, dtype=np.int16)
            
            if isinstance(audio_data, np.ndarray):
                # Ensure it's int16
                if audio_data.dtype != np.int16:
                    audio_data = audio_data.astype(np.int16)
                # Convert to bytes
                audio_bytes = audio_data.tobytes()
            else:
                # Already bytes
                audio_bytes = audio_data
            
            # If this is an utterance message, store metadata for context
            if envelope.ros_msg_type == "by_your_command/AudioDataUtterance":
                metadata = self.extract_utterance_metadata(envelope)
                if metadata:
                    self.current_utterance_metadata = metadata
                    self.logger.debug(f"Stored utterance metadata: {metadata}")
            
            return audio_bytes
            
        except Exception as e:
            self.logger.error(f"Gemini audio serialization failed: {e}")
            self.serialization_errors += 1
            raise SerializationError(f"Gemini audio serialization failed: {e}") from e
    
    def serialize_text(self, envelope: Union[MessageEnvelope, WebSocketMessageEnvelope]) -> Optional[str]:
        """
        Convert text message to Gemini format
        
        Gemini Live accepts plain text directly, no JSON wrapping needed.
        
        Args:
            envelope: Message envelope containing text data
            
        Returns:
            Optional[str]: Plain text for Gemini
        """
        try:
            # Extract text using base class method
            text = self.extract_text_from_ros(envelope)
            
            if not text:
                self.logger.warning("No text found in message")
                return None
            
            # Gemini accepts plain text directly
            # No need to wrap in conversation.item.create like OpenAI
            return text
            
        except Exception as e:
            self.logger.error(f"Gemini text serialization failed: {e}")
            self.serialization_errors += 1
            raise SerializationError(f"Gemini text serialization failed: {e}") from e
    
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
    
    # Gemini-specific helper methods (no OpenAI equivalents needed)
    
    def prepare_audio_for_send(self, audio_data: Union[list, np.ndarray, bytes]) -> bytes:
        """
        Prepare audio data for sending to Gemini Live
        
        Args:
            audio_data: Audio data in various formats
            
        Returns:
            bytes: PCM16 audio bytes ready to send
        """
        if isinstance(audio_data, (list, tuple)):
            audio_data = np.array(audio_data, dtype=np.int16)
        
        if isinstance(audio_data, np.ndarray):
            if audio_data.dtype != np.int16:
                audio_data = audio_data.astype(np.int16)
            return audio_data.tobytes()
        
        # Assume already bytes
        return audio_data
    
    # Methods not needed for Gemini (kept for interface compatibility)
    
    def create_response_trigger(self) -> None:
        """
        Gemini doesn't need manual response triggering.
        Responses are automatic based on conversation flow.
        """
        self.logger.debug("Response trigger not needed for Gemini (automatic responses)")
        return None
    
    def create_response_cancel(self) -> None:
        """
        Gemini uses session.interrupt() instead.
        This is handled at the session level, not serialization.
        """
        self.logger.debug("Response cancel not needed at serialization level for Gemini")
        return None
    
    def create_audio_buffer_clear(self) -> None:
        """
        Gemini doesn't have audio buffer management.
        Interruption handles everything.
        """
        self.logger.debug("Audio buffer clear not needed for Gemini")
        return None
    
    def create_conversation_truncate(self, item_id: str, audio_end_ms: int = 0) -> None:
        """
        Gemini doesn't support conversation truncation.
        Would need to handle differently if needed.
        """
        self.logger.debug("Conversation truncate not available in Gemini")
        return None
    
    def create_audio_buffer_commit(self) -> None:
        """
        Gemini doesn't need buffer commits.
        Audio is processed continuously as it's sent.
        """
        self.logger.debug("Audio buffer commit not needed for Gemini (continuous processing)")
        return None