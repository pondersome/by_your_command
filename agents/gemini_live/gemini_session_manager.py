"""
Gemini Live API Session Manager

Gemini-specific implementation of the base session manager with
configuration for the Gemini Live API.

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
from datetime import datetime

try:
    from google import genai
except ImportError:
    raise ImportError("google-genai library required: pip install google-genai")

from ..common.base_session_manager import BaseSessionManager, SessionState
from ..common.context import ConversationContext


class GeminiSessionManager(BaseSessionManager):
    """Gemini Live-specific session manager implementation"""
    
    def __init__(self, config: Dict):
        """Initialize Gemini session manager"""
        super().__init__(config)
        
        # Gemini-specific attributes
        self.client: Optional[genai.Client] = None
        self.session = None  # Will be the actual session from context manager
        self.session_context = None  # The async context manager
        self.model_name: str = config.get('model', 'models/gemini-2.0-flash-exp')
        
        # Session timing for reconnection management
        self.connection_start_time: Optional[float] = None
        self.session_start_time: Optional[float] = None
        
        # Video support
        self.video_enabled = config.get('enable_video', False)
        self.last_video_frame: Optional[bytes] = None
        self.video_frame_timestamp: Optional[float] = None
        
        # Gemini has strict time limits (depends on modalities)
        if self.video_enabled:
            self.max_connection_duration = 1.5 * 60  # 1.5 minutes (leave buffer before 2 min limit)
            self.max_session_duration = 1.5 * 60     # Same for video sessions
        else:
            self.max_connection_duration = 9 * 60    # 9 minutes (leave buffer before 10 min limit)
            self.max_session_duration = 14 * 60      # 14 minutes (leave buffer before 15 min limit)
        
        # Initialize Gemini API
        self.api_key = config.get('gemini_api_key', os.getenv('GEMINI_API_KEY'))
        if not self.api_key:
            raise ValueError("Gemini API key not found in config or environment")
        
        # Client will be created when connecting
        
    def _get_websocket_url(self) -> str:
        """Not used for Gemini - using genai.live client instead"""
        # This is required by base class but not used in Gemini implementation
        return "gemini-live-session"
    
    def _get_connection_params(self) -> Dict[str, Any]:
        """Get Gemini-specific connection parameters"""
        # Gemini doesn't use WebSocket headers, configuration is done differently
        return {}
    
    async def connect_session(self, context: Optional[ConversationContext] = None) -> bool:
        """
        Create new Gemini Live session with optional context injection
        
        Overrides base class to use Gemini Live API instead of WebSocket
        """
        if self.state != SessionState.IDLE:
            self.logger.warning(f"Cannot connect - session in state: {self.state.value}")
            return False
            
        try:
            import random
            conn_id = random.randint(1000, 9999)
            self.state = SessionState.CONNECTING
            self.logger.info(f"🔌 [{conn_id}] Connecting to Gemini Live...")
            
            # Build configuration for Gemini Live session
            config = self._build_session_config(context)
            
            # Create Gemini Live session
            self.logger.debug(f"[{conn_id}] Creating Live session with model: {self.model_name}")
            
            # Verify API key is present
            if not self.api_key:
                raise ValueError("Gemini API key is not set")
            
            self.logger.debug(f"[{conn_id}] Using API key: {'*' * 10}{self.api_key[-4:] if len(self.api_key) > 4 else '****'}")
            
            # Create client if not already created (reuse for better performance)
            if not hasattr(self, 'client') or self.client is None:
                self.client = genai.Client(
                    api_key=self.api_key,
                    http_options={'api_version': 'v1beta'}
                )
                # Small delay on first client creation to ensure initialization
                await asyncio.sleep(0.5)
                self.logger.debug(f"[{conn_id}] Created new Gemini client")
            
            # Connect using async context manager with timeout
            self.logger.debug(f"[{conn_id}] Attempting connection to Gemini Live...")
            # Connect with configuration
            self.session_context = self.client.aio.live.connect(
                model=self.model_name,
                config=config
            )
            
            # Add timeout for connection
            self.session = await asyncio.wait_for(
                self.session_context.__aenter__(),
                timeout=30.0  # 30 second timeout
            )
            self.logger.debug(f"[{conn_id}] Session established successfully")
            
            # Track connection timing
            self.connection_start_time = time.time()
            self.session_start_time = time.time()
            
            self.state = SessionState.CONNECTED
            self.logger.info(f"✅ [{conn_id}] Gemini Live session connected")
            
            # Unlike OpenAI, Gemini doesn't need to wait for session.created event
            # The session is immediately ready after connection
            self.state = SessionState.ACTIVE
            self.sessions_created += 1
            self.logger.info(f"✅ Session #{self.sessions_created} active")
            
            # Note: start_stream() requires stream and mime_type arguments
            # We don't actually need to call this for basic audio streaming
            # The session is ready to receive audio after connection
            
            return True
            
        except asyncio.TimeoutError:
            self.logger.error(f"❌ [{conn_id if 'conn_id' in locals() else '????'}] Connection timeout after 30 seconds")
            self.logger.error("Possible causes:")
            self.logger.error("  1. Check your internet connection")
            self.logger.error("  2. Verify API key is valid: set GEMINI_API_KEY environment variable")
            self.logger.error("  3. Check if Gemini Live API is available in your region")
            self.logger.error(f"  4. Verify the model is correct: {self.model_name}")
            
            # Clean up the hanging connection attempt
            if hasattr(self, 'session_context') and self.session_context:
                try:
                    await self.session_context.__aexit__(None, None, None)
                except:
                    pass  # Ignore cleanup errors
                self.session_context = None
            
            # Reset state so retries can work
            self.state = SessionState.IDLE
            self.session = None
            self.connection_start_time = None
            self.session_start_time = None
            return False
            
        except Exception as e:
            error_msg = str(e)
            import traceback
            
            # Always log the actual error first
            self.logger.error(f"❌ [{conn_id if 'conn_id' in locals() else '????'}] Connection failed with {type(e).__name__}: {e}")
            
            # Log full traceback for debugging
            self.logger.error(f"Full error details:\n{traceback.format_exc()}")
            
            # Then provide helpful context
            if "quota" in error_msg.lower():
                self.logger.error("Likely cause: API quota exceeded")
            elif "api key" in error_msg.lower() or "invalid" in error_msg.lower() or "unauthorized" in error_msg.lower():
                self.logger.error("Likely cause: Authentication issue")
                self.logger.error(f"Current API key ends with: ...{self.api_key[-4:] if self.api_key and len(self.api_key) > 4 else '????'}")
            elif "model" in error_msg.lower():
                self.logger.error(f"Likely cause: Model '{self.model_name}' issue")
            elif "timed out" in error_msg.lower():
                self.logger.error("Likely cause: Network timeout")
            elif "dns" in error_msg.lower() or "getaddrinfo" in error_msg.lower():
                self.logger.error("Likely cause: DNS resolution failure - check internet connection")
            
            # Clean up any hanging connection
            if hasattr(self, 'session_context') and self.session_context:
                try:
                    await self.session_context.__aexit__(None, None, None)
                except:
                    pass  # Ignore cleanup errors
                self.session_context = None
            
            # Reset state for retries
            self.state = SessionState.IDLE
            self.session = None
            self.connection_start_time = None
            self.session_start_time = None
            return False
    
    def _build_session_config(self, context: Optional[ConversationContext] = None) -> Dict[str, Any]:
        """Build Gemini-specific session configuration"""
        
        # Load and build system prompt
        prompt_id = self.config.get('prompt_id')
        
        # Try to get specific prompt by ID, or use context-based selection
        if prompt_id:
            self.logger.info(f"Looking for prompt ID: {prompt_id}")
            prompt_info = self.prompt_loader.get_prompt_info(prompt_id)
            if prompt_info:
                selected_prompt = prompt_info.system_prompt
                self.logger.info(f"✅ Found prompt: {prompt_info.name}")
            else:
                selected_prompt = None
                self.logger.warning(f"❌ Prompt ID '{prompt_id}' not found in prompts.yaml")
                # List available prompts for debugging
                available = self.prompt_loader.list_prompts()
                self.logger.warning(f"Available prompts: {available[:5]}...")  # Show first 5
        else:
            selected_prompt = self.prompt_loader.select_prompt({})
            
        if not selected_prompt:
            selected_prompt = "You are a helpful assistant."
            self.logger.warning("Using fallback default prompt")
            
        # Build prompt with context if available
        system_prompt = self.context_manager.build_system_prompt(
            selected_prompt, 
            context
        )
        
        # Log the prompt for debugging
        self.logger.info(f"📝 Using prompt ID: {prompt_id}")
        self.logger.info(f"📝 System prompt (first 500 chars): {system_prompt[:500]}...")
        
        # Check if command-related content is in the expanded prompt
        command_indicators = ['bumper', 'tenhut', 'lookup', 'pan@', 'move@', 'Arm Positions', 'Motion Commands']
        has_command_content = any(indicator in system_prompt for indicator in command_indicators)
        
        if has_command_content:
            self.logger.info("✅ Prompt contains command recognition content")
        else:
            self.logger.info("ℹ️ Prompt does not contain movement commands (OK for non-command agents)")
        
        # Build Gemini Live configuration - simplified format based on docs
        # Determine response modality based on agent type
        agent_id = self.config.get('agent_id', '')
        voice_setting = self.config.get('voice', '')
        if 'command' in agent_id.lower() or not voice_setting:
            # Command agents or agents with no voice should use TEXT only
            response_modalities = ["TEXT"]
            self.logger.info("📝 Command agent detected or no voice - using TEXT response modality")
        else:
            # Conversational agents use AUDIO
            response_modalities = ["AUDIO"]
            self.logger.info("🔊 Conversational agent - using AUDIO response modality")
        
        config = {
            "response_modalities": response_modalities,
            "system_instruction": system_prompt,
            "input_audio_transcription": {}  # Enable transcription of user audio
        }
        
        # Add output transcription for conversational agents with audio
        if "AUDIO" in response_modalities:
            config["output_audio_transcription"] = {}
            self.logger.info("🎤 Enabled input and output audio transcription")
        
        # Add video configuration if enabled
        if self.video_enabled:
            # Video is an input modality, not a response modality
            # Response stays as AUDIO only
            self.logger.info("Video mode enabled - 2 minute session limit applies")
        
        # Enable proactive audio if configured
        # This lets the model decide when to respond vs staying silent
        # Note: This may need adjustment based on actual API
        if self.config.get('proactive_audio', False):
            self.logger.info("Proactive audio enabled - model will decide when to speak")
        
        self.logger.debug(f"Gemini session config created with voice: {self.config.get('voice')}")
        return config
    
    async def _configure_session(self, context: Optional[ConversationContext] = None):
        """Not used in Gemini - configuration happens during connect"""
        # This is required by base class but configuration happens in connect_session for Gemini
        pass
    
    async def _wait_for_session_ready(self, timeout: float = 5.0) -> bool:
        """Gemini sessions are ready immediately after connection"""
        # Unlike OpenAI, Gemini doesn't have a separate session.created event
        # The session is ready as soon as connect() returns
        return True
    
    async def _update_session_prompt(self, prompt: str) -> bool:
        """Update session with new prompt"""
        try:
            if not self.session:
                self.logger.error("No active session to update")
                return False
            
            # Gemini doesn't support mid-session prompt updates directly
            # Would need to reconnect with new configuration
            self.logger.warning("Gemini Live doesn't support mid-session prompt updates - need to reconnect")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to update Gemini prompt: {e}")
            return False
    
    def _check_provider_limits(self) -> bool:
        """
        Check Gemini-specific session limits
        
        Gemini has strict limits:
        - 10 minute connection limit
        - 15 minute text/voice session limit
        - 2 minute video session limit (not used here)
        """
        if not self.connection_start_time:
            return False
        
        connection_duration = time.time() - self.connection_start_time
        session_duration = time.time() - self.session_start_time
        
        # Check connection limit (10 minutes)
        if connection_duration > self.max_connection_duration:
            self.logger.info(f"⏰ Gemini connection limit approaching: {connection_duration:.1f}s > {self.max_connection_duration}s")
            return True
        
        # Check session limit (15 minutes for audio)
        if session_duration > self.max_session_duration:
            self.logger.info(f"⏰ Gemini session limit approaching: {session_duration:.1f}s > {self.max_session_duration}s")
            return True
        
        return False
    
    async def close_session(self) -> Optional[ConversationContext]:
        """Close session and clean up Gemini-specific state"""
        # Get context before closing
        context = await super().close_session()
        
        # Close Gemini Live session
        if self.session:
            try:
                # Close using async context manager
                if hasattr(self, 'session_context'):
                    await self.session_context.__aexit__(None, None, None)
            except Exception as e:
                self.logger.error(f"Error closing Gemini session: {e}")
            finally:
                self.session = None
                self.session_context = None
        
        # Reset Gemini-specific timing
        self.connection_start_time = None
        self.session_start_time = None
        
        return context
    
    def is_connected(self) -> bool:
        """Check if Gemini session is connected"""
        if self.state not in [SessionState.CONNECTED, SessionState.ACTIVE]:
            return False
        return self.session is not None and hasattr(self.session, 'send')
    
    def add_conversation_turn(self, role: str, text: str, metadata: Optional[Dict] = None):
        """Add turn to conversation context"""
        self.context_manager.add_turn(role, text, metadata)
        
        # Gemini doesn't track tokens/costs the same way as OpenAI
        # Could add custom metrics here if needed
    
    async def send_audio(self, audio_data: bytes) -> bool:
        """Send audio data to Gemini Live session"""
        if not self.session:
            self.logger.error("No active session for audio")
            return False
        
        # Check if connection is still alive
        if not await self._check_connection_health():
            self.logger.error("Session connection is dead, attempting reconnect...")
            await self._handle_connection_loss()
            return False
        
        try:
            # Use send_realtime_input for audio - it's the correct method for real-time audio
            if hasattr(self.session, 'send_realtime_input'):
                # Create the proper input format for real-time audio
                from google.genai import types
                
                # Send as real-time audio input using 'audio' parameter (not 'media')
                await self.session.send_realtime_input(
                    audio=types.Blob(
                        data=audio_data,
                        mime_type="audio/pcm;rate=16000"
                    )
                )
                self.logger.debug(f"Sent {len(audio_data)} bytes of audio via send_realtime_input")
            elif hasattr(self.session, 'send'):
                # Fallback to send() with keyword argument
                # The signature shows it takes 'input' as a keyword argument
                from google.genai import types
                
                # Try sending as a blob with the input keyword
                await self.session.send(
                    input=types.Blob(
                        data=audio_data,
                        mime_type="audio/pcm;rate=16000"
                    )
                )
                self.logger.debug(f"Sent {len(audio_data)} bytes of audio via send(input=...)")
            else:
                self.logger.error(f"Cannot find send method. Available: {dir(self.session)}")
                return False
            
            return True
        except Exception as e:
            error_str = str(e)
            if "keepalive" in error_str or "1011" in error_str or "close" in error_str:
                self.logger.error(f"WebSocket connection lost: {e}")
                await self._handle_connection_loss()
            else:
                self.logger.error(f"Failed to send audio: {e}")
            return False
    
    async def send_turn_complete(self) -> bool:
        """Send explicit turn complete signal to Gemini"""
        if not self.session:
            self.logger.error("No active session for turn_complete")
            return False
            
        try:
            # Send empty content with turn_complete flag
            from google.genai import types
            await self.session.send_client_content(
                turns=types.Content(parts=[]),
                turn_complete=True
            )
            self.logger.debug("Sent turn_complete signal")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send turn_complete: {e}")
            return False
    
    async def send_text(self, text: str) -> bool:
        """Send text to Gemini Live session"""
        if not self.session:
            self.logger.error("No active session for text")
            return False
        
        try:
            # Use send_client_content with proper format
            from google.genai import types
            await self.session.send_client_content(
                turns=types.Content(
                    parts=[types.Part(text=text)]
                ),
                turn_complete=True
            )
            self.logger.debug(f"Sent text via send_client_content: {text[:50]}...")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send text: {e}")
            return False
    
    async def interrupt_response(self) -> bool:
        """Interrupt ongoing Gemini response"""
        if not self.session:
            return False
        
        try:
            # Gemini Live has a single interrupt call that handles everything
            await self.session.interrupt()
            self.logger.debug("Sent interrupt to Gemini")
            return True
        except Exception as e:
            self.logger.error(f"Failed to interrupt response: {e}")
            return False
    
    async def send_video_frame(self, frame_data: bytes, mime_type: str = "image/jpeg") -> bool:
        """Send video frame to Gemini Live session
        
        Args:
            frame_data: Video frame bytes (JPEG, PNG, etc.)
            mime_type: MIME type of the frame
            
        Returns:
            bool: True if sent successfully
        """
        if not self.session:
            self.logger.error("No active session for video")
            return False
        
        if not self.video_enabled:
            self.logger.error("Video not enabled for this session")
            return False
        
        try:
            # Store the last frame for potential reinjection on reconnect
            self.last_video_frame = frame_data
            self.video_frame_timestamp = time.time()
            
            # Send video frame using the correct API
            if hasattr(self.session, 'send_realtime_input'):
                from google.genai import types
                await self.session.send_realtime_input(
                    media=types.Blob(data=frame_data, mime_type=mime_type)
                )
            else:
                # Fallback to older API style
                await self.session.send(
                    genai.types.LiveClientContent(
                        inline_data=genai.types.InlineData(
                            mime_type=mime_type,
                            data=frame_data
                        )
                    )
                )
            return True
        except Exception as e:
            self.logger.error(f"Failed to send video frame: {e}")
            return False
    
    def should_reinject_video_frame(self) -> bool:
        """Check if we should reinject the last video frame after reconnection"""
        if not self.video_enabled or not self.last_video_frame:
            return False
        
        # Reinject if frame is less than 5 seconds old
        if self.video_frame_timestamp:
            age = time.time() - self.video_frame_timestamp
            return age < 5.0
        
        return False
    
    async def _check_connection_health(self) -> bool:
        """Check if the WebSocket connection is still alive"""
        if not self.session:
            return False
        
        # For now, just check if session exists and state is active
        # The WebSocket state checking was too aggressive
        if self.state != SessionState.ACTIVE:
            return False
        
        # Only check WebSocket if we know it's actually closed
        if hasattr(self.session, '_ws') and self.session._ws:
            # Only return False if we're certain it's closed
            if hasattr(self.session._ws, 'closed') and self.session._ws.closed:
                self.logger.debug("WebSocket is closed")
                return False
            # Check for websockets library specific states
            if hasattr(self.session._ws, 'state'):
                import websockets
                if hasattr(websockets, 'protocol') and hasattr(websockets.protocol, 'State'):
                    # Only fail if explicitly CLOSED or CLOSING
                    if self.session._ws.state in [websockets.protocol.State.CLOSED, 
                                                   websockets.protocol.State.CLOSING]:
                        self.logger.debug(f"WebSocket state is {self.session._ws.state}")
                        return False
        
        return True
    
    async def _handle_connection_loss(self):
        """Handle lost connection by cleaning up and marking session as idle"""
        self.logger.warning("Handling connection loss - cleaning up session")
        
        # Clean up the dead session
        if self.session:
            try:
                if hasattr(self, 'session_context'):
                    await self.session_context.__aexit__(None, None, None)
            except:
                pass  # Ignore cleanup errors
            
            self.session = None
            self.session_context = None
        
        # Reset state
        self.state = SessionState.IDLE
        self.connection_start_time = None
        self.session_start_time = None
        
        self.logger.info("Session cleaned up after connection loss")