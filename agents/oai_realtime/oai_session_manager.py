"""
OpenAI Realtime API Session Manager

OpenAI-specific implementation of the base session manager with
configuration for the OpenAI Realtime API.

Author: Karim Virani
Version: 1.0
Date: December 2024
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional, Dict, Any

try:
    import websockets
except ImportError:
    raise ImportError("websockets library required: pip install websockets")

from ..common.base_session_manager import BaseSessionManager, SessionState
from ..common.context import ConversationContext


class OpenAISessionManager(BaseSessionManager):
    """OpenAI-specific session manager implementation"""
    
    def __init__(self, config: Dict):
        """Initialize OpenAI session manager"""
        super().__init__(config)
        
        # OpenAI-specific attributes
        self.session_id: Optional[str] = None
        self.conversation_id: Optional[str] = None
        
    def _get_websocket_url(self) -> str:
        """Get OpenAI WebSocket URL with authentication"""
        api_key = self.config.get('openai_api_key', os.getenv('OPENAI_API_KEY'))
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")
            
        model = self.config.get('model', 'gpt-4o-realtime-preview')
        
        # Construct URL with API key as bearer token
        url = f"wss://api.openai.com/v1/realtime?model={model}"
        
        # Note: OpenAI expects the API key in the Authorization header
        # This is handled by the websockets library with custom headers
        self._websocket_headers = {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        return url
    
    async def _configure_session(self, context: Optional[ConversationContext] = None):
        """Send OpenAI-specific session configuration"""
        
        # Load and build system prompt
        prompt_id = self.config.get('prompt_id')
        selected_prompt = self.prompt_loader.get_prompt(prompt_id) if prompt_id else None
        
        if not selected_prompt:
            selected_prompt = "You are a helpful assistant."
            self.logger.warning("No prompt found, using default")
            
        # Build prompt with context if available
        system_prompt = self.context_manager.build_system_prompt(
            selected_prompt, 
            context
        )
        
        # Log voice configuration for debugging
        voice_setting = self.config.get('voice', 'alloy')
        self.logger.info(f"Using voice: {voice_setting} (from config)")
        
        # OpenAI session configuration message
        config_msg = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": system_prompt,
                "voice": voice_setting,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": self.config.get('vad_threshold', 0.5),
                    "prefix_padding_ms": self.config.get('vad_prefix_padding', 300),
                    "silence_duration_ms": self.config.get('vad_silence_duration', 200),
                    "create_response": self.config.get('vad_create_response', False)
                }
            }
        }
        
        # Need to handle custom headers for OpenAI auth
        if hasattr(self, '_websocket_headers'):
            # Reconnect with headers if not already connected with them
            if not self.websocket:
                url = self._get_websocket_url()
                self.websocket = await asyncio.wait_for(
                    websockets.connect(
                        url,
                        extra_headers=self._websocket_headers,
                        ping_interval=30,
                        ping_timeout=10,
                        close_timeout=10,
                        max_size=10 * 1024 * 1024
                    ),
                    timeout=10.0
                )
        
        await self.websocket.send(json.dumps(config_msg))
        self.logger.info("📤 OpenAI session configuration sent")
        self.logger.debug(f"Configuration: {json.dumps(config_msg, indent=2)}")
    
    async def _wait_for_session_ready(self, timeout: float = 5.0) -> bool:
        """Wait for OpenAI session.created event"""
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # Check for messages from OpenAI
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=0.5
                    )
                    
                    data = json.loads(message)
                    event_type = data.get("type", "")
                    
                    if event_type == "session.created":
                        self.session_id = data.get("session", {}).get("id")
                        self.logger.info(f"✅ OpenAI session created: {self.session_id}")
                        return True
                    elif event_type == "error":
                        error_msg = data.get("error", {}).get("message", "Unknown error")
                        self.logger.error(f"❌ OpenAI error: {error_msg}")
                        return False
                        
                except asyncio.TimeoutError:
                    continue
                    
            self.logger.error(f"⏰ Timeout waiting for session.created after {timeout}s")
            return False
            
        except Exception as e:
            self.logger.error(f"Error waiting for session ready: {e}")
            return False
    
    async def _update_session_prompt(self, prompt: str) -> bool:
        """Update OpenAI session with new prompt"""
        try:
            # Build update message with new prompt
            config_msg = {
                "type": "session.update",
                "session": {
                    "instructions": prompt,
                    # Keep other settings the same
                    "modalities": ["text", "audio"],
                    "voice": self.config.get('voice', 'alloy'),
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": self.config.get('vad_threshold', 0.5),
                        "prefix_padding_ms": self.config.get('vad_prefix_padding', 300),
                        "silence_duration_ms": self.config.get('vad_silence_duration', 200),
                        "create_response": self.config.get('vad_create_response', False)
                    }
                }
            }
            
            await self.websocket.send(json.dumps(config_msg))
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update OpenAI prompt: {e}")
            return False
    
    def _check_provider_limits(self) -> bool:
        """
        Check OpenAI-specific limits
        
        OpenAI doesn't have documented hard limits for Realtime API,
        but we can add soft limits based on usage patterns.
        """
        # Currently no OpenAI-specific limits beyond base class limits
        # Could add token counting or cost estimation here
        return False
    
    async def send_response_create(self):
        """OpenAI-specific: Trigger response generation"""
        if not self.is_ready_for_audio():
            self.logger.warning("Cannot trigger response - session not ready")
            return False
            
        try:
            response_msg = {"type": "response.create"}
            await self.websocket.send(json.dumps(response_msg))
            self.logger.debug("Triggered OpenAI response generation")
            return True
        except Exception as e:
            self.logger.error(f"Failed to trigger response: {e}")
            return False
    
    async def cancel_response(self):
        """OpenAI-specific: Cancel ongoing response"""
        if not self.is_connected():
            return False
            
        try:
            cancel_msg = {"type": "response.cancel"}
            await self.websocket.send(json.dumps(cancel_msg))
            self.logger.debug("Sent response.cancel to OpenAI")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel response: {e}")
            return False
    
    async def clear_audio_buffer(self):
        """OpenAI-specific: Clear output audio buffer"""
        if not self.is_connected():
            return False
            
        try:
            clear_msg = {"type": "output_audio_buffer.clear"}
            await self.websocket.send(json.dumps(clear_msg))
            self.logger.debug("Cleared OpenAI audio buffer")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear audio buffer: {e}")
            return False
    
    async def truncate_conversation_item(self, item_id: str, audio_end_ms: int = 0):
        """OpenAI-specific: Truncate conversation item"""
        if not self.is_connected():
            return False
            
        try:
            truncate_msg = {
                "type": "conversation.item.truncate",
                "item_id": item_id,
                "content_index": 0,
                "audio_end_ms": audio_end_ms
            }
            await self.websocket.send(json.dumps(truncate_msg))
            self.logger.debug(f"Truncated conversation item: {item_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to truncate item: {e}")
            return False