"""
Base Session Manager for Multi-Provider LLM Support

Provider-agnostic session management with common functionality for
WebSocket connections, context management, and session lifecycle.

Author: Karim Virani
Version: 1.0
Date: August 2025
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
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
    CONNECTED = "connected"  # WebSocket connected, waiting for session ready
    ACTIVE = "active"        # Session ready for audio/messages
    CLOSING = "closing"
    CLOSED = "closed"


class BaseSessionManager(ABC):
    """Provider-agnostic session manager base class"""
    
    def __init__(self, config: Dict):
        """Initialize common session management components"""
        self.config = config
        self.state = SessionState.IDLE
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.session_start_time: Optional[float] = None
        
        # Context and prompt management
        self.context_manager = ContextManager(
            max_context_tokens=config.get('max_context_tokens', 2000),
            max_context_age=config.get('conversation_timeout', config.get('max_context_age', 600.0))
        )
        self.prompt_loader = PromptLoader(config.get('prompts_file'))
        
        # Metrics
        self.sessions_created = 0
        self.total_session_duration = 0.0
        self.max_session_duration = config.get('session_max_duration', 120.0)
        self.max_session_tokens = config.get('session_max_tokens', 50000)
        
        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Session data
        self.current_session_id: Optional[str] = None
        self.session_created_at: Optional[float] = None
        
    async def connect_session(self, context: Optional[ConversationContext] = None) -> bool:
        """
        Establish connection with LLM provider
        
        Args:
            context: Optional conversation context to inject
            
        Returns:
            bool: True if connection successful
        """
        if self.state != SessionState.IDLE:
            self.logger.warning(f"Cannot connect - session in state: {self.state.value}")
            return False
            
        try:
            import random
            conn_id = random.randint(1000, 9999)
            self.state = SessionState.CONNECTING
            self.logger.info(f"🔌 [{conn_id}] Connecting to LLM provider...")
            
            # Get provider-specific WebSocket URL and connection parameters
            self.logger.debug(f"[{conn_id}] Getting WebSocket URL...")
            url = self._get_websocket_url()
            self.logger.debug(f"[{conn_id}] Got URL: {url}")
            
            # Get provider-specific connection parameters (headers, etc.)
            connect_params = self._get_connection_params()
            
            # Merge common and provider-specific parameters
            ws_params = {
                'ping_interval': 30,
                'ping_timeout': 20,
                'close_timeout': 10,
                'open_timeout': 30,  # Increase timeout for slower connections
                'max_size': 10 * 1024 * 1024,  # 10MB max message size
            }
            
            # Add provider-specific parameters (like extra_headers)
            if connect_params:
                ws_params.update(connect_params)
            
            # Connect with merged parameters
            self.logger.debug(f"[{conn_id}] Attempting WebSocket connection...")
            self.websocket = await websockets.connect(url, **ws_params)
            
            self.state = SessionState.CONNECTED
            self.logger.info(f"✅ [{conn_id}] WebSocket connected, configuring session...")
            
            # Provider-specific session configuration
            await self._configure_session(context)
            
            # Wait for session to be ready (provider-specific)
            if await self._wait_for_session_ready():
                self.state = SessionState.ACTIVE
                self.session_start_time = time.time()
                self.sessions_created += 1
                self.logger.info(f"✅ Session #{self.sessions_created} active")
                return True
            else:
                await self.close_session()
                return False
                
        except Exception as e:
            error_msg = str(e)
            if "3000" in error_msg and "invalid_request_error" in error_msg:
                self.logger.error(f"❌ Connection failed - Invalid request format or parameters: {e}")
                self.logger.error("Check: prompt configuration, model availability, or API format changes")
            elif "401" in error_msg or "403" in error_msg:
                self.logger.error(f"❌ Connection failed - Authentication issue: {e}")
                self.logger.error("Check: API key is valid and has proper permissions")
            elif "429" in error_msg:
                self.logger.error(f"❌ Connection failed - Rate limit exceeded: {e}")
            elif "timeout" in error_msg.lower():
                self.logger.error(f"❌ Connection failed - Timeout: {e}")
                if "opening handshake" in error_msg.lower():
                    self.logger.error("Network issue or API temporarily unavailable - will retry on next message")
                else:
                    self.logger.error("Check network connectivity and API status")
            else:
                self.logger.error(f"❌ [{conn_id if 'conn_id' in locals() else '????'}] Connection failed: {e}")
            
            self.state = SessionState.IDLE
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            return False
    
    async def close_session(self) -> Optional[ConversationContext]:
        """
        Close current session and return conversation context
        
        Returns:
            Optional[ConversationContext]: Preserved context if available
        """
        if self.state not in [SessionState.ACTIVE, SessionState.CONNECTING, SessionState.CONNECTED]:
            return None
            
        try:
            self.state = SessionState.CLOSING
            
            # Track session duration
            if self.session_start_time:
                duration = time.time() - self.session_start_time
                self.total_session_duration += duration
                self.logger.info(f"📊 Session duration: {duration:.1f}s")
                
            # Get current context before closing
            context = self.context_manager.get_current_context()
            
            # Close WebSocket connection
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
                
            self.state = SessionState.IDLE
            self.session_start_time = None
            self.current_session_id = None
            self.session_created_at = None
            self.logger.info("🔌 Session closed")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error closing session: {e}")
            self.state = SessionState.IDLE
            self.websocket = None
            self.session_start_time = None
            self.current_session_id = None
            self.session_created_at = None
            return None
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        if self.state not in [SessionState.CONNECTED, SessionState.ACTIVE]:
            return False
        if self.websocket is None:
            return False
        # Check websocket state for websockets 15.x compatibility
        try:
            # For websockets 15.x, check state attribute
            from websockets.protocol import State
            return self.websocket.state == State.OPEN
        except (AttributeError, ImportError):
            # Fallback for older versions
            return not getattr(self.websocket, 'closed', True)
    
    def is_ready_for_audio(self) -> bool:
        """Check if session is ready to receive audio"""
        return self.state == SessionState.ACTIVE and self.is_connected()
    
    def check_session_limits(self) -> bool:
        """
        Check if current session has reached any limits
        
        Returns:
            bool: True if any limit exceeded
        """
        if not self.session_start_time:
            return False
            
        # Check duration limit
        duration = time.time() - self.session_start_time
        if duration > self.max_session_duration:
            self.logger.info(f"⏰ Session duration limit reached: {duration:.1f}s > {self.max_session_duration}s")
            return True
            
        # Provider-specific limit checks can be added in subclasses
        return self._check_provider_limits()
    
    async def update_session_prompt(self, prompt_id: str = None, prompt_text: str = None) -> bool:
        """
        Update the session's system prompt
        
        Args:
            prompt_id: ID of prompt from prompts file
            prompt_text: Direct prompt text (overrides prompt_id)
            
        Returns:
            bool: True if update successful
        """
        if self.state != SessionState.ACTIVE:
            self.logger.warning("Cannot update prompt - session not active")
            return False
            
        try:
            # Load prompt
            if prompt_text:
                prompt = prompt_text
                prompt_info = "direct text"
            elif prompt_id:
                prompt_obj = self.prompt_loader.get_prompt_info(prompt_id)
                prompt = prompt_obj.system_prompt if prompt_obj else None
                prompt_info = f"id: {prompt_id}"
            else:
                # Use context-based selection or default
                prompt = self.prompt_loader.select_prompt({})
                prompt_info = "context-selected"
                
            if not prompt:
                self.logger.error("No prompt available for update")
                return False
                
            # Provider-specific prompt update
            success = await self._update_session_prompt(prompt)
            
            if success:
                self.logger.info(f"✅ Updated session prompt ({prompt_info})")
            else:
                self.logger.error(f"❌ Failed to update prompt ({prompt_info})")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error updating prompt: {e}")
            return False
    
    def add_context_item(self, text: str, role: str = "user"):
        """Add item to conversation context"""
        self.context_manager.add_turn(role, text)
    
    def reset_conversation_context(self):
        """Reset conversation context (called on conversation ID change)"""
        self.logger.info("🧹 Resetting conversation context")
        self.context_manager.reset_context()
        
        # If we have an active session, update it with fresh system prompt
        # (subclasses can override to do provider-specific updates)
        if self.state == SessionState.ACTIVE:
            asyncio.create_task(self.update_session_prompt())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get session metrics"""
        return {
            'sessions_created': self.sessions_created,
            'total_duration': self.total_session_duration,
            'avg_duration': self.total_session_duration / max(1, self.sessions_created),
            'current_state': self.state.value,
            'is_connected': self.is_connected()
        }
    
    # ============================================================
    # Abstract methods - must be implemented by provider subclasses
    # ============================================================
    
    @abstractmethod
    def _get_websocket_url(self) -> str:
        """
        Get provider-specific WebSocket URL
        
        Returns:
            str: WebSocket URL for the provider
        """
        pass
    
    @abstractmethod
    def _get_connection_params(self) -> Dict[str, Any]:
        """
        Get provider-specific connection parameters
        
        Returns:
            Dict: Extra parameters for websocket connection (e.g., extra_headers)
        """
        pass
    
    @abstractmethod
    async def _configure_session(self, context: Optional[ConversationContext] = None):
        """
        Send provider-specific session configuration
        
        Args:
            context: Optional conversation context to inject
        """
        pass
    
    @abstractmethod
    async def _wait_for_session_ready(self, timeout: float = 5.0) -> bool:
        """
        Wait for provider-specific session ready signal
        
        Args:
            timeout: Maximum time to wait for ready signal
            
        Returns:
            bool: True if session is ready
        """
        pass
    
    @abstractmethod
    async def _update_session_prompt(self, prompt: str) -> bool:
        """
        Update session with new prompt (provider-specific)
        
        Args:
            prompt: New system prompt text
            
        Returns:
            bool: True if update successful
        """
        pass
    
    @abstractmethod
    def _check_provider_limits(self) -> bool:
        """
        Check provider-specific session limits
        
        Returns:
            bool: True if any provider limit exceeded
        """
        pass