"""
OpenAI Realtime API Session Manager

OpenAI-specific implementation of the base session manager with
configuration for the OpenAI Realtime API.

Author: Karim Virani
Version: 1.0
Date: August 2025
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
        """Get OpenAI WebSocket URL"""
        from urllib.parse import quote
        model = self.config.get('model', 'gpt-4o-realtime-preview')
        # URL encode the model parameter in case of special characters
        encoded_model = quote(model, safe='')
        url = f"wss://api.openai.com/v1/realtime?model={encoded_model}"
        self.logger.info(f"OpenAI WebSocket URL: {url}")
        return url
    
    def _get_connection_params(self) -> Dict[str, Any]:
        """Get OpenAI-specific connection parameters including auth headers"""
        api_key = self.config.get('openai_api_key', os.getenv('OPENAI_API_KEY'))
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")
        
        # Log key presence (not the key itself)
        self.logger.debug(f"API key found: {len(api_key)} characters")
        
        # OpenAI expects the API key in the Authorization header
        # Note: websockets 15.x uses 'additional_headers' not 'extra_headers'
        return {
            'additional_headers': {
                "Authorization": f"Bearer {api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
        }
    
    async def _configure_session(self, context: Optional[ConversationContext] = None):
        """Send OpenAI-specific session configuration"""
        
        # Load and build system prompt
        prompt_id = self.config.get('prompt_id')
        
        # Try to get specific prompt by ID, or use context-based selection
        if prompt_id:
            prompt_info = self.prompt_loader.get_prompt_info(prompt_id)
            selected_prompt = prompt_info.system_prompt if prompt_info else None
        else:
            selected_prompt = self.prompt_loader.select_prompt({})
            
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
        
        # Send configuration to OpenAI
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
                        error_data = data.get("error", {})
                        error_msg = error_data.get("message", "Unknown error")
                        error_type = error_data.get("type", "unknown")
                        error_code = error_data.get("code", "")
                        
                        self.logger.error(f"❌ OpenAI API error: {error_type} - {error_msg}")
                        if error_code:
                            self.logger.error(f"   Error code: {error_code}")
                        
                        # Provide helpful context based on error type
                        if "invalid_request" in error_type:
                            self.logger.error("   Check: session configuration, prompt format, or model parameters")
                        elif "authentication" in error_type or "unauthorized" in error_type:
                            self.logger.error("   Check: API key validity and permissions")
                        elif "rate_limit" in error_type:
                            self.logger.error("   Rate limit exceeded - wait and retry")
                            
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
    
    async def close_session(self) -> Optional[ConversationContext]:
        """Close session and clean up OpenAI-specific state"""
        # Call parent close_session first
        context = await super().close_session()
        
        # Reset OpenAI-specific variables
        self.session_id = None
        self.conversation_id = None
        
        return context
    
    def add_conversation_turn(self, role: str, text: str, metadata: Optional[Dict] = None):
        """Add turn to conversation context"""
        self.context_manager.add_turn(role, text, metadata)
        
        # Rough token estimation for cost tracking (OpenAI-specific)
        token_estimate = len(text) // 4  # ~4 chars per token
        if not hasattr(self, 'current_session_tokens'):
            self.current_session_tokens = 0
        if not hasattr(self, 'current_session_cost'):
            self.current_session_cost = 0.0
            
        self.current_session_tokens += token_estimate
        
        # Rough cost estimation (example rates)
        if role == "user":
            self.current_session_cost += token_estimate * 0.00001  # Input tokens
        else:
            self.current_session_cost += token_estimate * 0.00003  # Output tokens