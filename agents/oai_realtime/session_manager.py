"""
OpenAI Realtime API Session Manager

Manages WebSocket connections to OpenAI Realtime API with intelligent
session lifecycle, cycling, and context injection.

Author: Karim Virani
Version: 1.0
Date: July 2025
"""

import asyncio
import json
import logging
import time
from enum import Enum
from typing import Optional, Dict, Any

try:
    import websockets
except ImportError:
    raise ImportError("websockets library required: pip install websockets")

from .context import ConversationContext, ContextManager


class SessionState(Enum):
    """Session lifecycle states"""
    IDLE = "idle"
    CONNECTING = "connecting" 
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"


class SessionManager:
    """Manage OpenAI Realtime API session lifecycle and cycling"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.state = SessionState.IDLE
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.session_start_time: Optional[float] = None
        self.context_manager = ContextManager(
            max_context_tokens=config.get('max_context_tokens', 2000),
            max_context_age=config.get('max_context_age', 3600)
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.session_max_duration = config.get('session_max_duration', 120.0)
        self.session_max_tokens = config.get('session_max_tokens', 50000)
        self.session_max_cost = config.get('session_max_cost', 5.00)
        self.api_key = config.get('openai_api_key', '')
        self.model = config.get('model', 'gpt-4o-realtime-preview')
        self.base_system_prompt = config.get('system_prompt', self._default_system_prompt())
        
        # Session metrics
        self.sessions_created = 0
        self.sessions_closed = 0
        self.total_session_duration = 0.0
        self.current_session_tokens = 0
        self.current_session_cost = 0.0
        
    def _default_system_prompt(self) -> str:
        """Default system prompt for the assistant"""
        return """You are a helpful robotic assistant. You can control robot movements, 
answer questions, and engage in natural conversation. Be concise but friendly.
Respond naturally to the user's speech and provide helpful information or assistance."""
        
    async def connect_session(self, context: Optional[ConversationContext] = None) -> bool:
        """Create new WebSocket session with optional context injection"""
        if self.state != SessionState.IDLE:
            self.logger.warn(f"Cannot connect from state {self.state}")
            return False
            
        if not self.api_key:
            self.logger.error("OpenAI API key not configured")
            return False
            
        try:
            self.state = SessionState.CONNECTING
            self.logger.info("Connecting to OpenAI Realtime API...")
            
            # WebSocket connection to OpenAI Realtime API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            url = f"wss://api.openai.com/v1/realtime?model={self.model}"
            self.websocket = await websockets.connect(url, extra_headers=headers)
            
            # Send session configuration
            await self._configure_session(context)
            
            self.state = SessionState.ACTIVE
            self.session_start_time = time.time()
            self.sessions_created += 1
            self.current_session_tokens = 0
            self.current_session_cost = 0.0
            
            self.logger.info(f"OpenAI Realtime session established (#{self.sessions_created})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to OpenAI: {e}")
            self.state = SessionState.IDLE
            self.websocket = None
            return False
            
    async def _configure_session(self, context: Optional[ConversationContext] = None):
        """Configure session with system prompt and context"""
        system_prompt = self.context_manager.build_system_prompt(
            self.base_system_prompt, 
            context
        )
        
        config_msg = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": system_prompt,
                "voice": self.config.get('voice', 'alloy'),
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": self.config.get('vad_threshold', 0.5),
                    "prefix_padding_ms": self.config.get('vad_prefix_padding', 300),
                    "silence_duration_ms": self.config.get('vad_silence_duration', 200)
                }
            }
        }
        
        await self.websocket.send(json.dumps(config_msg))
        self.logger.debug("Session configuration sent")
        
    async def close_session(self) -> Optional[ConversationContext]:
        """Close current session and return conversation context"""
        if self.state not in [SessionState.ACTIVE, SessionState.CONNECTING]:
            return None
            
        try:
            self.state = SessionState.CLOSING
            self.logger.info("Closing OpenAI Realtime session...")
            
            if self.websocket:
                await self.websocket.close()
                
            self.state = SessionState.CLOSED
            
            # Calculate session metrics
            if self.session_start_time:
                session_duration = time.time() - self.session_start_time
                self.total_session_duration += session_duration
                self.sessions_closed += 1
                
                self.logger.info(f"Session closed after {session_duration:.1f}s "
                               f"(tokens: ~{self.current_session_tokens}, "
                               f"cost: ~${self.current_session_cost:.2f})")
            
            # Return current context for next session
            context = self.context_manager.get_current_context()
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error closing session: {e}")
            return None
        finally:
            self.state = SessionState.IDLE
            self.websocket = None
            self.session_start_time = None
            
    def check_session_limits(self) -> bool:
        """Check if session should be rotated due to time/cost limits"""
        if not self.session_start_time or self.state != SessionState.ACTIVE:
            return False
            
        elapsed = time.time() - self.session_start_time
        
        # Check time limit
        if elapsed > self.session_max_duration:
            self.logger.info(f"Session time limit reached: {elapsed:.1f}s")
            return True
            
        # Check token limit (rough estimate)
        if self.current_session_tokens > self.session_max_tokens:
            self.logger.info(f"Session token limit reached: ~{self.current_session_tokens}")
            return True
            
        # Check cost limit (rough estimate)
        if self.current_session_cost > self.session_max_cost:
            self.logger.info(f"Session cost limit reached: ~${self.current_session_cost:.2f}")
            return True
            
        return False
        
    def add_conversation_turn(self, role: str, text: str, metadata: Optional[Dict] = None):
        """Add turn to conversation context"""
        self.context_manager.add_turn(role, text, metadata)
        
        # Rough token estimation for cost tracking
        token_estimate = len(text) // 4  # ~4 chars per token
        self.current_session_tokens += token_estimate
        
        # Rough cost estimation (example rates)
        if role == "user":
            self.current_session_cost += token_estimate * 0.00001  # Input tokens
        else:
            self.current_session_cost += token_estimate * 0.00002  # Output tokens
    
    def is_connected(self) -> bool:
        """Check if session is currently connected"""
        return self.state == SessionState.ACTIVE and self.websocket is not None
        
    def get_session_duration(self) -> float:
        """Get current session duration in seconds"""
        if not self.session_start_time:
            return 0.0
        return time.time() - self.session_start_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get session manager metrics"""
        base_metrics = {
            'state': self.state.value,
            'sessions_created': self.sessions_created,
            'sessions_closed': self.sessions_closed,
            'total_session_duration': self.total_session_duration,
            'current_session_duration': self.get_session_duration(),
            'current_session_tokens': self.current_session_tokens,
            'current_session_cost': self.current_session_cost,
            'average_session_duration': (self.total_session_duration / self.sessions_closed 
                                       if self.sessions_closed > 0 else 0.0)
        }
        
        # Add context stats
        context_stats = self.context_manager.get_context_stats()
        base_metrics.update({f'context_{k}': v for k, v in context_stats.items()})
        
        return base_metrics
    
    def update_configuration(self, new_config: Dict):
        """Update session configuration"""
        old_config = self.config.copy()
        self.config.update(new_config)
        
        # Update derived values
        self.session_max_duration = self.config.get('session_max_duration', 120.0)
        self.session_max_tokens = self.config.get('session_max_tokens', 50000)
        self.session_max_cost = self.config.get('session_max_cost', 5.00)
        
        self.logger.info("Session configuration updated")
        
    def reset_metrics(self):
        """Reset session metrics"""
        self.sessions_created = 0
        self.sessions_closed = 0
        self.total_session_duration = 0.0
        self.current_session_tokens = 0
        self.current_session_cost = 0.0
        self.context_manager.reset_context()
        
        self.logger.info("Session metrics reset")