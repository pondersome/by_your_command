#!/usr/bin/env python3
"""
Gemini Live API Agent

Main agent implementation with intelligent session management, conversation
continuity, and video support. Integrates with ROS AI Bridge for zero-copy message handling.

Author: Karim Virani
Version: 1.0
Date: August 2025
"""

import asyncio
import base64
import json
import logging
import time
from typing import Optional, Dict, Any, List
import numpy as np

try:
    from google import genai
except ImportError:
    raise ImportError("google-genai library required: pip install google-genai")

from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
from sensor_msgs.msg import Image, CompressedImage

# Use refactored components
from .gemini_serializer import GeminiSerializer
from .gemini_session_manager import GeminiSessionManager
from ..common.base_session_manager import SessionState
from ..common.debug_interface import DebugInterface
from ..common import WebSocketBridgeInterface, PauseDetector, ConversationContext, ConversationMonitor


class GeminiLiveAgent:
    """
    Gemini Live API agent with intelligent session management and video support.
    
    Features:
    - Automatic reconnection for time-limited sessions (2/10/15 minutes)
    - Video stream support with frame caching for reconnection
    - Proactive audio mode for natural conversation flow
    - Seamless conversation continuity through context preservation
    - Zero-copy integration with ROS AI Bridge
    - Simplified interruption handling
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.bridge_interface: Optional[WebSocketBridgeInterface] = None  # Will be set in initialize()
        
        # Core components - using refactored classes
        self.serializer = GeminiSerializer()
        self.pause_detector = PauseDetector(
            pause_timeout=self.config.get('session_pause_timeout', 10.0)
        )
        self.session_manager = GeminiSessionManager(self.config)
        
        # Conversation monitoring
        conversation_timeout = self.config.get('conversation_timeout', 
                                              self.config.get('max_context_age', 600.0))
        self.conversation_monitor = ConversationMonitor(
            timeout=conversation_timeout,
            on_conversation_change=self._handle_conversation_change
        )
        
        # State
        self.running = False
        self.prepared_context: Optional[ConversationContext] = None
        self.session_creating = False  # Flag to prevent concurrent session creation
        
        # Video support
        self.video_enabled = self.config.get('enable_video', False)
        self.video_topic = self.config.get('video_topic', '/camera/image_compressed')
        self.last_video_frame: Optional[bytes] = None
        self.last_video_timestamp: Optional[float] = None
        self.video_frame_interval = self.config.get('video_frame_interval', 1.0)  # Send frame every second
        
        # Response tracking (simpler for Gemini - no manual triggering)
        self.expecting_response = False
        self.response_timeout_start = None
        self.response_timeout_seconds = 10.0
        
        # Assistant response accumulation
        self.assistant_transcript_buffer = ""
        
        # Debug interface for standalone testing
        self.debug_interface: Optional[DebugInterface] = None
        
        # Session ready event for synchronization
        self.session_ready = asyncio.Event()
        
        # Published topic configuration
        self.published_topics = {
            'audio_out': self.config.get('audio_out_topic', 'audio_out'),
            'transcript': self.config.get('transcript_topic', 'llm_transcript'),
            'command_detected': self.config.get('command_detected_topic', 'command_detected'),
            'interruption_signal': self.config.get('interruption_signal_topic', 'interruption_signal')
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.get('log_level', logging.INFO))
        
        # Metrics
        self.metrics = {
            'messages_processed': 0,
            'audio_chunks_sent': 0,
            'text_messages_sent': 0,
            'video_frames_sent': 0,
            'responses_received': 0,
            'interruptions': 0,
            'sessions_created': 0,
            'sessions_cycled_on_pause': 0,
            'sessions_cycled_on_limits': 0,
            'errors': 0
        }
        
        # Agent ID for multi-agent systems
        self.agent_id = self.config.get('agent_id', 'gemini_live')
        
        # Response processor task
        self._response_processor_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize agent connection to bridge"""
        self.logger.info("Initializing Gemini Live Agent...")
        
        # Start conversation monitoring
        await self.conversation_monitor.start_monitoring()
        
        # Connect to bridge interface
        await self._connect_to_bridge()
        
        # Initialize debug interface if no bridge connection
        if not self.bridge_interface:
            self.debug_interface = DebugInterface(self)
            await self.debug_interface.start()
            self.logger.info("Debug interface initialized for standalone testing")
    
    async def _connect_to_bridge(self):
        """Connect to the ROS AI Bridge via WebSocket"""
        try:
            self.logger.info("Connecting to ROS AI Bridge via WebSocket...")
            
            # Create WebSocket bridge interface
            from agents.common import WebSocketBridgeInterface
            self.bridge_interface = WebSocketBridgeInterface(self.config)
            
            # Attempt connection with retries
            success = await self.bridge_interface.connect_with_retry()
            
            if success:
                self.logger.info("✅ Successfully connected to bridge via WebSocket")
            else:
                self.logger.warning("❌ Failed to connect to bridge - running in standalone mode")
                self.bridge_interface = None
            
        except Exception as e:
            self.logger.error(f"Bridge connection error: {e}")
            self.logger.warning("Running in standalone mode without bridge connection")
            self.bridge_interface = None
            
    async def _ensure_session_ready(self):
        """Ensure we have an active session when needed"""
        if (self.session_manager.state == SessionState.IDLE and 
            self.prepared_context is not None and 
            not self.session_creating):
            
            self.session_creating = True
            try:
                # Create new session with prepared context
                success = await self.session_manager.connect_session(self.prepared_context)
                if success:
                    self.prepared_context = None
                    self.pause_detector.reset()
                    self.session_ready.set()  # Set immediately - Gemini sessions are ready right away
                    
                    # Start response processor
                    if self._response_processor_task is None or self._response_processor_task.done():
                        self.logger.info("🚀 Starting response processor")
                        self._response_processor_task = asyncio.create_task(self._continuous_response_processor())
                    
                    # Reinject last video frame if needed
                    if self.video_enabled and self.session_manager.should_reinject_video_frame():
                        await self.session_manager.send_video_frame(
                            self.session_manager.last_video_frame,
                            "image/jpeg"
                        )
                        self.logger.info("📹 Reinjected last video frame after reconnection")
                    
                    self.logger.info("✅ Session created with injected context")
                else:
                    self.logger.error("❌ Failed to create session with context")
            finally:
                self.session_creating = False
                
    async def _process_bridge_messages(self):
        """Process messages from bridge interface"""
        try:
            # Skip if bridge interface not connected yet
            if not self.bridge_interface or not self.bridge_interface.is_connected():
                return
                
            # Try to get message with short timeout
            envelope = await self.bridge_interface.get_inbound_message(timeout=0.1)
            
            if envelope is None:
                return
            
            if envelope:
                self.pause_detector.record_message(envelope.ros_msg_type)
                self.metrics['messages_processed'] += 1
                self.logger.info(f"[{self.conversation_monitor.current_conversation_id[-12:]}] 📨 Processing: {envelope.ros_msg_type}")
                
                # Handle different message types
                if envelope.ros_msg_type == "by_your_command/AudioDataUtterance":
                    await self._handle_audio_message(envelope)
                elif envelope.ros_msg_type == "std_msgs/String" and envelope.topic_name.endswith("/text_input"):
                    await self._handle_text_message(envelope)
                elif envelope.ros_msg_type in ["sensor_msgs/CompressedImage", "sensor_msgs/Image"]:
                    await self._handle_video_message(envelope)
                elif envelope.ros_msg_type == "std_msgs/String" and envelope.topic_name.endswith("/conversation_id"):
                    await self._handle_conversation_id_message(envelope)
                    
        except Exception as e:
            self.logger.error(f"Error processing bridge message: {e}")
            self.metrics['errors'] += 1
            
    async def _handle_audio_message(self, envelope):
        """Handle audio data utterance"""
        # Record activity for conversation monitoring
        self.conversation_monitor.record_activity()
        
        # Ensure session is ready
        await self._ensure_session_for_message()
        
        # Serialize audio for Gemini (returns raw bytes, not JSON)
        audio_bytes = await self.serializer.safe_serialize(envelope)
        
        if audio_bytes:
            chunk_id = envelope.raw_data.chunk_sequence if hasattr(envelope.raw_data, 'chunk_sequence') else 0
            
            # Wait for session ready
            if not self.session_ready.is_set():
                self.logger.info(f"⏳ Waiting for session ready before sending chunk #{chunk_id}...")
                await self.session_ready.wait()
            
            # Send audio directly to Gemini
            if self.session_manager.is_ready_for_audio():
                success = await self.session_manager.send_audio(audio_bytes)
                if success:
                    self.metrics['audio_chunks_sent'] += 1
                    
                    # Check if this is the last chunk
                    if hasattr(envelope.raw_data, 'is_end') and envelope.raw_data.is_end:
                        self.logger.info(f"🎤 Sent final audio chunk #{chunk_id}")
                        self.expecting_response = True
                        self.response_timeout_start = time.time()
                        
                        # Store metadata for context
                        utterance_metadata = self.serializer.get_utterance_metadata()
                        if utterance_metadata:
                            self.serializer.add_utterance_context(utterance_metadata)
                    else:
                        self.logger.debug(f"🎤 Sent audio chunk #{chunk_id}")
            else:
                self.logger.warning(f"Session not ready for audio chunk #{chunk_id}")
                
    async def _handle_text_message(self, envelope):
        """Handle text input message"""
        # Ensure session is ready
        await self._ensure_session_for_message()
        
        # Wait for session ready
        if not self.session_ready.is_set():
            self.logger.info("⏳ Waiting for session ready before sending text...")
            await self.session_ready.wait()
        
        # Extract and send text directly (Gemini doesn't need JSON wrapping)
        text = self.serializer.serialize_text(envelope)
        if text and self.session_manager.is_connected():
            success = await self.session_manager.send_text(text)
            if success:
                self.metrics['text_messages_sent'] += 1
                self.logger.info(f"💬 Sent text: {text[:100]}...")
                self.expecting_response = True
                self.response_timeout_start = time.time()
                
    async def _handle_video_message(self, envelope):
        """Handle video frame message"""
        if not self.video_enabled:
            return
            
        # Check frame rate limiting
        current_time = time.time()
        if self.last_video_timestamp:
            elapsed = current_time - self.last_video_timestamp
            if elapsed < self.video_frame_interval:
                return  # Skip this frame
        
        # Ensure session is ready
        await self._ensure_session_for_message()
        
        # Extract frame data
        if envelope.ros_msg_type == "sensor_msgs/CompressedImage":
            frame_data = envelope.raw_data.data
            mime_type = "image/jpeg"  # Compressed images are typically JPEG
        else:  # sensor_msgs/Image
            # Would need to convert raw image to JPEG/PNG
            self.logger.warning("Raw Image messages not yet supported - use CompressedImage")
            return
        
        # Send video frame
        if self.session_manager.is_connected():
            success = await self.session_manager.send_video_frame(frame_data, mime_type)
            if success:
                self.metrics['video_frames_sent'] += 1
                self.last_video_timestamp = current_time
                self.logger.debug(f"📹 Sent video frame ({len(frame_data)} bytes)")
                
    async def _handle_conversation_id_message(self, envelope):
        """Handle conversation ID change"""
        new_id = envelope.raw_data.data
        if new_id != self.conversation_monitor.current_conversation_id:
            self.logger.info(f"🔄 Conversation ID changed: {new_id}")
            self.conversation_monitor.set_conversation_id(new_id)
            
    async def _ensure_session_for_message(self):
        """Ensure we have an active session for incoming messages"""
        if self.session_manager.state == SessionState.IDLE:
            if self.prepared_context is not None:
                self.logger.info("🔄 Deferring to prepared context session creation")
                return
            
            if self.session_creating:
                self.logger.info("⏳ Session creation in progress, skipping")
                return
                
            self.session_creating = True
            try:
                self.logger.info("🔗 Creating Gemini session for incoming message...")
                success = await self.session_manager.connect_session()
                if success:
                    self.pause_detector.reset()
                    self.session_ready.set()  # Gemini sessions are ready immediately
                    
                    # Start response processor
                    if self._response_processor_task is None or self._response_processor_task.done():
                        self.logger.info("🚀 Starting response processor")
                        self._response_processor_task = asyncio.create_task(self._continuous_response_processor())
                    
                    self.logger.info("✅ Session created for incoming message")
                else:
                    self.logger.error("❌ Failed to create session")
            finally:
                self.session_creating = False
                
    async def _continuous_response_processor(self):
        """Process responses from Gemini Live session"""
        # For now, just keep it simple and non-blocking
        # The Gemini receive() API seems problematic
        self.logger.info("🎧 Gemini response processor started (simplified)")
        
        while self.running and self.session_manager.is_connected():
            # Just sleep and don't block
            await asyncio.sleep(1.0)
            
        self.logger.info("🛑 Gemini response processor stopped")
            
    async def _process_gemini_response(self, response):
        """Process response from Gemini Live"""
        try:
            # Gemini responses can be audio, text, or tool calls
            if isinstance(response, bytes):
                # Audio response
                await self._handle_audio_response(response)
            elif isinstance(response, str):
                # Text response
                await self._handle_text_response(response)
            elif isinstance(response, dict):
                # Structured response (tool calls, etc.)
                await self._handle_structured_response(response)
                
            self.metrics['responses_received'] += 1
            self.expecting_response = False
            self.response_timeout_start = None
            
        except Exception as e:
            self.logger.error(f"Error processing Gemini response: {e}")
            
    async def _handle_audio_response(self, audio_data: bytes):
        """Handle audio response from Gemini"""
        # Convert PCM16 bytes to AudioData message
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Publish audio
        if self.bridge_interface:
            await self.bridge_interface.put_outbound_message(
                topic=self.published_topics['audio_out'],
                msg_data={'int16_data': audio_array.tolist()},
                msg_type='audio_common_msgs/AudioData'
            )
            self.logger.debug(f"🔊 Published audio response ({len(audio_array)} samples)")
            
    async def _handle_text_response(self, text: str):
        """Handle text response from Gemini"""
        # Add to transcript buffer
        self.assistant_transcript_buffer += text
        
        # Publish transcript
        if self.bridge_interface:
            await self.bridge_interface.put_outbound_message(
                topic=self.published_topics['transcript'],
                msg_data={'data': f"Assistant: {text}"},
                msg_type='std_msgs/String'
            )
            
        # Add to conversation context
        self.session_manager.add_conversation_turn("assistant", text)
        self.logger.info(f"🤖 Assistant: {text[:100]}...")
        
    async def _handle_structured_response(self, response: Dict):
        """Handle structured response from Gemini (tool calls, etc.)"""
        response_type = response.get('type', 'unknown')
        
        if response_type == 'tool_call':
            # Handle tool/function call
            tool_name = response.get('tool_name')
            tool_args = response.get('arguments', {})
            self.logger.info(f"🔧 Tool call: {tool_name}({tool_args})")
            # Would implement tool execution here
            
        elif response_type == 'error':
            error_msg = response.get('message', 'Unknown error')
            self.logger.error(f"❌ Gemini error: {error_msg}")
            
        else:
            self.logger.debug(f"Received structured response: {response_type}")
            
    async def _handle_interruption(self):
        """Handle user interruption of assistant response"""
        try:
            self.logger.info("🛑 User interrupted - stopping response")
            self.metrics['interruptions'] += 1
            
            # Single call to interrupt in Gemini
            await self.session_manager.interrupt_response()
            
            # Publish interruption signal for audio player
            if self.bridge_interface:
                await self.bridge_interface.put_outbound_message(
                    topic=self.published_topics['interruption_signal'],
                    msg_data={'data': 'interrupt'},
                    msg_type='std_msgs/String'
                )
                
            # Clear expectations
            self.expecting_response = False
            self.response_timeout_start = None
            
        except Exception as e:
            self.logger.error(f"Error handling interruption: {e}")
            
    async def _cycle_session_on_pause(self):
        """Cycle session when pause detected"""
        self.logger.info("🔄 Cycling session on pause")
        
        # Close current session and get context
        context = await self.session_manager.close_session()
        self.prepared_context = context
        self.metrics['sessions_cycled_on_pause'] += 1
        
        self.logger.info("✅ Session cycled - ready for next speech")
        
    async def _cycle_session_on_limits(self):
        """Cycle session when approaching time limits"""
        self.logger.info("⏰ Cycling session due to time limits")
        
        try:
            context = await self.session_manager.close_session()
            
            # Immediately reconnect with context
            if context:
                success = await self.session_manager.connect_session(context)
                if success:
                    self.metrics['sessions_cycled_on_limits'] += 1
                    self.pause_detector.reset()
                    self.session_ready.set()
                    self.logger.info("✅ Session rotated seamlessly")
                    
                    # Reinject video frame if needed
                    if self.video_enabled and self.session_manager.should_reinject_video_frame():
                        await self.session_manager.send_video_frame(
                            self.session_manager.last_video_frame,
                            "image/jpeg"
                        )
                        self.logger.info("📹 Reinjected video frame after rotation")
                else:
                    self.logger.error("❌ Failed to rotate session")
                    self.prepared_context = context
                    
        except Exception as e:
            self.logger.error(f"Error rotating session: {e}")
            
    def _handle_conversation_change(self, old_id: str, new_id: str):
        """Handle conversation ID change"""
        self.logger.info(f"🔄 Conversation changed from {old_id} to {new_id}")
        # Reset conversation context
        self.session_manager.reset_conversation_context()
        
    async def _manage_response_processor(self):
        """Manage the response processor task lifecycle"""
        try:
            # Start processor if we have an active session
            if self.session_manager.state == SessionState.ACTIVE:
                if self._response_processor_task is None or self._response_processor_task.done():
                    # Only log if actually starting (not if it's already running)
                    if self._response_processor_task is None or self._response_processor_task.done():
                        self.logger.info("🚀 Starting response processor")
                        self._response_processor_task = asyncio.create_task(self._continuous_response_processor())
                    
        except Exception as e:
            self.logger.error(f"Error managing response processor: {e}")
            
    async def run(self):
        """Main agent loop"""
        self.running = True
        self.logger.info(f"🚀 Gemini Live Agent '{self.agent_id}' starting...")
        
        # Don't create session immediately - wait for first message
        self.prepared_context = None
        
        try:
            while self.running:
                # Ensure session is ready when needed
                await self._ensure_session_ready()
                
                # Process incoming messages from bridge
                await self._process_bridge_messages()
                
                # Check for session limits (critical for Gemini)
                if self.session_manager.check_session_limits():
                    await self._cycle_session_on_limits()
                
                # Check for conversation pause
                elif self.pause_detector.check_pause_condition():
                    if not self.expecting_response:
                        await self._cycle_session_on_pause()
                    elif self.response_timeout_start:
                        elapsed = time.time() - self.response_timeout_start
                        if elapsed > self.response_timeout_seconds:
                            self.logger.info(f"⏰ Response timeout after {elapsed:.1f}s - cycling")
                            await self._cycle_session_on_pause()
                
                # Manage response processor
                await self._manage_response_processor()
                
                # Small sleep to prevent busy loop
                await asyncio.sleep(0.01)
                
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Fatal error in agent loop: {e}")
        finally:
            await self.cleanup()
            
    async def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.logger.info("🧹 Cleaning up Gemini Live Agent...")
        
        # Stop response processor
        if self._response_processor_task is not None and not self._response_processor_task.done():
            self._response_processor_task.cancel()
            
        # Close session
        if self.session_manager.state == SessionState.ACTIVE:
            await self.session_manager.close_session()
            
        if self.bridge_interface:
            await self.bridge_interface.close()
            
        if self.debug_interface:
            await self.debug_interface.stop()
            
        self.logger.info("✅ Gemini Live Agent shutdown complete")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        return {
            **self.metrics,
            **self.session_manager.get_metrics(),
            'conversation_monitor': {
                'current_id': self.conversation_monitor.current_conversation_id,
                'duration': self.conversation_monitor.get_conversation_duration()
            }
        }