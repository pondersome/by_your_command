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
from typing import Optional, Dict, Any, List

try:
    import websockets
except ImportError:
    raise ImportError("websockets library required: pip install websockets")

from .context import ConversationContext, ContextManager
from .prompt_loader import PromptLoader


class SessionState(Enum):
    """Session lifecycle states"""
    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"  # WebSocket connected, waiting for session.created
    ACTIVE = "active"        # Session ready for audio/messages
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
        
        # Named prompt system
        self.prompt_loader = PromptLoader()
        # Check if prompt_id is specified in config
        if config.get('prompt_id'):
            self._prompt_override = config.get('prompt_id')
            self.logger.info(f"Using configured prompt_id: {self._prompt_override}")
        # Fallback to config system_prompt if named prompts fail
        self.fallback_system_prompt = config.get('system_prompt', self._default_system_prompt())
        
        # Current prompt selection context
        self.prompt_context = {
            'user_age': config.get('user_age'),
            'environment': config.get('environment', 'normal'),
            'robot_name': config.get('robot_name', 'robot'),
            'agent_id': config.get('agent_id', 'openai_realtime')
        }
        
        # Session metrics
        self.sessions_created = 0
        self.sessions_closed = 0
        self.total_session_duration = 0.0
        self.current_session_tokens = 0
        self.current_session_cost = 0.0
        
    def _default_system_prompt(self) -> str:
        """Default system prompt for the assistant"""
        return """You are a helpful robotic assistant integrated into a voice-controlled robot system. 
You can engage in natural conversation, answer questions, and provide assistance.

Respond naturally and conversationally to user speech. Be concise but friendly and helpful.
You are designed to work through real-time voice interaction, so keep responses appropriate for spoken conversation.

When users ask questions or make requests, provide clear and helpful responses."""
        
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
            self.websocket = await websockets.connect(url, additional_headers=headers)
            
            # Send session configuration
            await self._configure_session(context)
            
            # Wait for session.created event to transition to ACTIVE
            self.state = SessionState.CONNECTED
            self.session_start_time = time.time()
            self.sessions_created += 1
            self.current_session_tokens = 0
            self.current_session_cost = 0.0
            
            self.logger.info(f"OpenAI session connecting (#{self.sessions_created}) - waiting for session.created")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to OpenAI: {e}")
            self.state = SessionState.IDLE
            self.websocket = None
            return False
            
    async def _configure_session(self, context: Optional[ConversationContext] = None):
        """Configure session with system prompt and context"""
        # Select appropriate prompt based on context and configuration
        try:
            selected_prompt = self.prompt_loader.select_prompt(self.prompt_context)
            self.logger.info("Using context-selected system prompt")
        except Exception as e:
            self.logger.warning(f"Prompt selection failed, using default: {e}")
            selected_prompt = self._default_system_prompt()
            
        # Build final system prompt with conversation context
        system_prompt = self.context_manager.build_system_prompt(
            selected_prompt, 
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
                    "type": "server_vad"
                }
            }
        }
        
        await self.websocket.send(json.dumps(config_msg))
        self.logger.info("📤 Session configuration sent:")
        self.logger.info(f"   {json.dumps(config_msg, indent=2)}")
        
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
        return self.state in [SessionState.CONNECTED, SessionState.ACTIVE] and self.websocket is not None
        
    def is_ready_for_audio(self) -> bool:
        """Check if session is ready to receive audio data"""
        return self.state == SessionState.ACTIVE and self.websocket is not None
        
    def handle_session_created(self, session_data: Dict):
        """Handle session.created event from OpenAI"""
        if self.state == SessionState.CONNECTED:
            self.state = SessionState.ACTIVE
            session_id = session_data.get('id', 'unknown')
            self.logger.info(f"✅ OpenAI session ready: {session_id} - READY FOR AUDIO")
            # Signal that session is ready for use
            return True  # Indicate session is now ready
        else:
            self.logger.warning(f"Received session.created in unexpected state: {self.state}")
            return False
        
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
        
        # Update prompt context if relevant keys changed
        prompt_context_keys = ['user_age', 'environment', 'robot_name', 'agent_id']
        for key in prompt_context_keys:
            if key in new_config:
                self.prompt_context[key] = new_config[key]
        
        self.logger.info("Session configuration updated")
        
    def update_prompt_context(self, context_updates: Dict[str, Any]):
        """Update prompt selection context"""
        old_context = self.prompt_context.copy()
        self.prompt_context.update(context_updates)
        
        changed_keys = [k for k, v in context_updates.items() if old_context.get(k) != v]
        if changed_keys:
            self.logger.info(f"Updated prompt context: {changed_keys}")
            
    def get_current_prompt_info(self) -> Dict[str, Any]:
        """Get information about currently selected prompt"""
        try:
            selected_prompt = self.prompt_loader.select_prompt(self.prompt_context)
            # Find which prompt this corresponds to
            for prompt_id, prompt_info in self.prompt_loader.prompts.items():
                if prompt_info.system_prompt == selected_prompt:
                    return {
                        'prompt_id': prompt_id,
                        'name': prompt_info.name,
                        'description': prompt_info.description,
                        'version': prompt_info.version,
                        'tested_with': prompt_info.tested_with,
                        'context': self.prompt_context.copy()
                    }
            return {'prompt_id': 'unknown', 'context': self.prompt_context.copy()}
        except Exception as e:
            return {'error': str(e), 'using_fallback': True}
            
    def reload_prompts(self):
        """Reload prompts from file"""
        try:
            self.prompt_loader.reload_prompts()
            self.logger.info("Prompts reloaded successfully")
        except Exception as e:
            self.logger.error(f"Error reloading prompts: {e}")
            
    def list_available_prompts(self) -> List[str]:
        """List all available prompt IDs"""
        return self.prompt_loader.list_prompts()
        
    async def switch_prompt(self, prompt_id: str = None, context_updates: Dict[str, Any] = None) -> bool:
        """Switch to a different prompt, optionally updating context"""
        try:
            # Update context if provided
            if context_updates:
                self.update_prompt_context(context_updates)
            
            # If specific prompt requested, temporarily override selection
            old_selection_override = getattr(self, '_prompt_override', None)
            if prompt_id:
                if prompt_id not in self.prompt_loader.prompts:
                    self.logger.error(f"Prompt '{prompt_id}' not found. Available: {self.list_available_prompts()}")
                    return False
                self._prompt_override = prompt_id
                self.logger.info(f"Overriding prompt selection to: {prompt_id}")
            
            # Get new prompt
            if hasattr(self, '_prompt_override') and self._prompt_override:
                new_prompt = self.prompt_loader.prompts[self._prompt_override].system_prompt
                prompt_info = f"override:{self._prompt_override}"
            else:
                new_prompt = self.prompt_loader.select_prompt(self.prompt_context)
                prompt_info = "context-selected"
            
            # If we have an active session, update it
            if self.state == SessionState.ACTIVE and self.websocket:
                # Build system prompt with current context
                system_prompt = self.context_manager.build_system_prompt(new_prompt, 
                    self.context_manager.get_current_context())
                
                # Update session with new prompt
                config_msg = {
                    "type": "session.update", 
                    "session": {
                        "instructions": system_prompt,
                        "modalities": ["text", "audio"],
                        "voice": self.config.get('voice', 'alloy'),
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "input_audio_transcription": {"model": "whisper-1"},
                        "turn_detection": {
                            "type": "server_vad"
                        }
                    }
                }
                
                await self.websocket.send(json.dumps(config_msg))
                self.logger.info(f"✅ Updated active session with new prompt ({prompt_info})")
                return True
            else:
                # No active session - change will take effect on next session
                self.logger.info(f"✅ Prompt updated ({prompt_info}) - will apply to next session")
                return True
                
        except Exception as e:
            # Restore old override if switch failed
            if old_selection_override is not None:
                self._prompt_override = old_selection_override
            elif hasattr(self, '_prompt_override'):
                delattr(self, '_prompt_override')
            self.logger.error(f"Failed to switch prompt: {e}")
            return False
    
    def clear_prompt_override(self):
        """Clear prompt override and return to context-based selection"""
        if hasattr(self, '_prompt_override'):
            old_override = self._prompt_override
            delattr(self, '_prompt_override')
            self.logger.info(f"Cleared prompt override '{old_override}' - using context-based selection")
            
    def get_effective_prompt_selection(self) -> Dict[str, Any]:
        """Get information about what prompt would be selected right now"""
        if hasattr(self, '_prompt_override') and self._prompt_override:
            prompt_info = self.prompt_loader.get_prompt_info(self._prompt_override)
            return {
                'selection_type': 'override',
                'prompt_id': self._prompt_override,
                'prompt_info': prompt_info.__dict__ if prompt_info else None,
                'context': self.prompt_context.copy()
            }
        else:
            return {
                'selection_type': 'context-based',
                **self.get_current_prompt_info()
            }
        
    def reset_metrics(self):
        """Reset session metrics"""
        self.sessions_created = 0
        self.sessions_closed = 0
        self.total_session_duration = 0.0
        self.current_session_tokens = 0
        self.current_session_cost = 0.0
        self.context_manager.reset_context()
        
        self.logger.info("Session metrics reset")