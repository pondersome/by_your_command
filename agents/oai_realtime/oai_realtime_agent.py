#!/usr/bin/env python3
"""
OpenAI Realtime API Agent

Main agent implementation with intelligent session management and conversation
continuity. Integrates with ROS AI Bridge for zero-copy message handling.

Author: Karim Virani
Version: 1.0
Date: July 2025
"""

import asyncio
import base64
import json
import logging
import time
from typing import Optional, Dict, Any, List
import numpy as np
import websockets

#import rclpy
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData

# Use refactored components
from .oai_serializer import OpenAISerializer
from .oai_session_manager import OpenAISessionManager
from ..common.base_session_manager import SessionState
from ..common.debug_interface import DebugInterface
from ..common import WebSocketBridgeInterface, PauseDetector, ConversationContext, ConversationMonitor


class OpenAIRealtimeAgent:
    """
    OpenAI Realtime API agent with intelligent session management.
    
    Features:
    - Cost-optimized session cycling on conversation pauses
    - Seamless conversation continuity through context preservation
    - Zero-copy integration with ROS AI Bridge
    - Comprehensive error handling and recovery
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.bridge_interface: Optional[WebSocketBridgeInterface] = None  # Will be set in initialize()
        
        # Core components - using refactored classes
        self.serializer = OpenAISerializer()
        self.pause_detector = PauseDetector(
            pause_timeout=self.config.get('session_pause_timeout', 10.0)
        )
        self.session_manager = OpenAISessionManager(self.config)
        
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
        
        # Response tracking for proper session cycling
        self.pending_responses = {
            'transcription': False,     # Waiting for user transcript
            'assistant_response': False, # Waiting for assistant to start responding
            'audio_complete': False     # Waiting for assistant response to complete
        }
        self.response_timeout_start = None  # When we started waiting for responses
        self.response_timeout_seconds = 10.0  # Max time to wait for responses
        self.last_waiting_log_time = 0  # Last time we logged "waiting for responses"
        self.waiting_log_interval = 5.0  # Log waiting message every 5 seconds
        
        # Assistant response accumulation
        self.assistant_transcript_buffer = ""
        self.last_assistant_item_id: Optional[str] = None
        
        # Debug interface for standalone testing
        self.debug_interface: Optional[DebugInterface] = None
        
        # Session ready event for synchronization
        self.session_ready = asyncio.Event()
        
        # Published topic configuration (support command extractor agent)
        # Using new naming convention: prompt_* for user input, response_* for agent output
        self.published_topics = {
            'response_voice': self.config.get('response_voice_topic',
                                             self.config.get('audio_out_topic', 'response_voice')),  # Fallback for compatibility
            'response_text': self.config.get('response_text_topic',
                                            self.config.get('transcript_topic', 'response_text')),  # Fallback for compatibility
            'response_cmd': self.config.get('response_cmd_topic',
                                           self.config.get('command_transcript_topic', '')),  # Command extractor output
            'prompt_transcript': self.config.get('prompt_transcript_topic', 'prompt_transcript'),  # NEW: User voice transcript
            'command_detected': self.config.get('command_detected_topic', 'command_detected'),  # Keep as-is
            'interruption_signal': self.config.get('interruption_signal_topic', 'interruption_signal')  # Keep as-is
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.get('log_level', logging.INFO))
        
        # Metrics
        self.metrics = {
            'messages_processed': 0,
            'messages_sent_to_openai': 0,
            'messages_sent_to_ros': 0,
            'sessions_cycled_on_pause': 0,
            'sessions_cycled_on_limits': 0,
            'total_runtime': 0.0
        }
        
        self.start_time: Optional[float] = None
        
    async def initialize(self):
        """Initialize agent components"""
        self.logger.info("Initializing OpenAI Realtime Agent...")
        
        # Setup conversation ID logging after we have a conversation monitor
        self._setup_conversation_logging()
        
        # Start conversation monitoring
        await self.conversation_monitor.start_monitoring()
        
        # Validate configuration
        api_key = self.config.get('openai_api_key', '')
        if not api_key:
            self.logger.error("❌ No OpenAI API key found in configuration")
            raise ValueError("OpenAI API key required in configuration")
        else:
            # Show masked API key for confirmation
            masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
            self.logger.info(f"🔑 OpenAI API key configured: {masked_key}")
            
        # Connect to bridge interface
        await self._connect_to_bridge()
        
        # Initialize debug interface if no bridge connection
        if not self.bridge_interface:
            self.debug_interface = DebugInterface(self)
            await self.debug_interface.start()
            self.logger.info("🔧 Debug interface enabled for standalone mode")
            
        self.logger.info("Agent initialized successfully")
        self.logger.info(f"🎭 Initial conversation ID: {self.conversation_monitor.current_conversation_id}")
        
        # Publish initial conversation ID after a small delay to ensure bridge is ready
        if self.bridge_interface and self.bridge_interface.is_connected():
            await asyncio.sleep(1.0)  # Give bridge time to set up publishers
            await self._publish_conversation_id(self.conversation_monitor.current_conversation_id)
        
    async def _connect_to_bridge(self):
        """Connect to the ROS AI Bridge via WebSocket for distributed deployment"""
        try:
            self.logger.info("Connecting to ROS AI Bridge via WebSocket...")
            
            # Create WebSocket bridge interface
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
        
    async def run(self):
        """Main agent loop - consumes from bridge, manages sessions"""
        self.running = True
        self.start_time = time.time()
        self.logger.info(f"[{self.conversation_monitor.current_conversation_id[-12:]}] Starting OpenAI Realtime Agent main loop...")
        
        try:
            while self.running:
                # Ensure session is ready when needed
                await self._ensure_session_ready()
                
                # Process incoming messages from bridge
                await self._process_bridge_messages()
                
                # Handle session lifecycle management
                await self._manage_session_lifecycle()
                
                # Process OpenAI responses
                await self._process_openai_responses()
                
                # Small sleep to prevent busy loop
                await asyncio.sleep(0.01)
                
        except Exception as e:
            self.logger.error(f"Agent main loop error: {e}", exc_info=True)
        finally:
            await self._cleanup()
            
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
                    self._reset_response_tracking()
                    self.session_ready.set()  # Set immediately - session is ready
                    
                    # Start response processor immediately after session creation
                    if not hasattr(self, '_response_processor_task') or self._response_processor_task.done():
                        self.logger.info("🚀 Starting continuous response processor task immediately")
                        self._response_processor_task = asyncio.create_task(self._continuous_response_processor())
                    
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
                # No message available
                return
            
            if envelope:
                self.pause_detector.record_message(envelope.ros_msg_type)
                self.metrics['messages_processed'] += 1
                self.logger.info(f"[{self.conversation_monitor.current_conversation_id[-12:]}] 📨 Processing message: {envelope.ros_msg_type} from {envelope.topic_name}")
                
                # Record activity for conversation monitoring if it's a voice chunk
                if envelope.ros_msg_type == "by_your_command/AudioDataUtterance":
                    self.conversation_monitor.record_activity()
                
                # Ensure we have a session for incoming messages
                if self.session_manager.state == SessionState.IDLE:
                    # Check if we have prepared context first (priority)
                    if self.prepared_context is not None:
                        self.logger.info("🔄 Deferring to prepared context session creation")
                        return  # Let _ensure_session_ready() handle it
                    
                    # Check if session creation is already in progress
                    if self.session_creating:
                        self.logger.info("⏳ Session creation in progress, skipping")
                        return
                        
                    self.session_creating = True
                    try:
                        self.logger.info("🔗 Creating OpenAI session for incoming message...")
                        success = await self.session_manager.connect_session()
                        if success:
                            self.pause_detector.reset()
                            self._reset_response_tracking()
                            self.session_ready.set()  # Set immediately - session is ready
                            
                            # Start response processor immediately after session creation
                            if not hasattr(self, '_response_processor_task') or self._response_processor_task.done():
                                self.logger.info("🚀 Starting continuous response processor task immediately")
                                self._response_processor_task = asyncio.create_task(self._continuous_response_processor())
                            
                            self.logger.info("✅ Session created for incoming message")
                        else:
                            self.logger.error("❌ Failed to create session - check configuration and API status")
                            return
                    finally:
                        self.session_creating = False
                
                # Handle AudioDataUtterance with metadata
                if envelope.ros_msg_type == "by_your_command/AudioDataUtterance":
                    # Serialize for OpenAI Realtime API (agent responsibility)
                    api_msg = await self.serializer.safe_serialize(envelope)
                    
                    chunk_id = envelope.raw_data.chunk_sequence
                    
                    if api_msg:
                        # Wait for session to be ready before sending audio
                        if not self.session_ready.is_set():
                            self.logger.info(f"⏳ Waiting for session ready before sending chunk #{chunk_id}...")
                            await self.session_ready.wait()
                            self.logger.info(f"✅ Session ready - proceeding with chunk #{chunk_id}")
                        
                        if self.session_manager.is_ready_for_audio():
                            await self.session_manager.websocket.send(json.dumps(api_msg))
                            self.metrics['messages_sent_to_openai'] += 1
                            self.logger.info(f"✅ SENT chunk #{chunk_id} ({len(json.dumps(api_msg))} bytes)")
                            
                            # Check if this is the end of utterance - commit audio buffer
                            if envelope.raw_data.is_utterance_end:
                                # Add small delay to ensure audio is processed by OpenAI
                                await asyncio.sleep(0.1)  # 100ms delay
                                
                                commit_msg = self.serializer.create_audio_buffer_commit()
                                await self.session_manager.websocket.send(json.dumps(commit_msg))
                                utterance_id = envelope.raw_data.utterance_id
                                self.logger.info(f"💾 Committed audio buffer for utterance {utterance_id}")
                                
                                # Log current session configuration for debugging
                                self.logger.info(f"📊 Session state: {self.session_manager.state.value}")
                                
                                # Set up response expectations
                                self._setup_response_expectations()
                        else:
                            current_state = self.session_manager.state.value
                            has_websocket = self.session_manager.websocket is not None
                            self.logger.warning(f"⏳ BLOCKED chunk #{chunk_id} - session not ready (state: {current_state}, websocket: {has_websocket})")
                        
                        # Store metadata for context injection
                        utterance_metadata = self.serializer.get_utterance_metadata()
                        if utterance_metadata:
                            self.serializer.add_utterance_context(utterance_metadata)
                            
                    elif api_msg:
                        self.logger.warning("⚠️ AudioDataUtterance serialized but no active session")
                    else:
                        self.logger.error("❌ Failed to serialize AudioDataUtterance")
                        
                elif envelope.ros_msg_type == "std_msgs/String" and envelope.topic_name.endswith("/conversation_id"):
                    # Handle external conversation ID changes
                    new_conversation_id = envelope.raw_data.data
                    if new_conversation_id != self.conversation_monitor.current_conversation_id:
                        self.logger.info(f"📨 Received external conversation ID: {new_conversation_id}")
                        self.conversation_monitor.handle_external_reset(new_conversation_id)
                        
                elif envelope.ros_msg_type == "std_msgs/String" and envelope.topic_name.endswith("/text_input"):
                    # Handle text input - inject as conversation item
                    text_content = envelope.raw_data.data
                    self.logger.info(f"📝 Received text input: '{text_content[:50]}...'")
                    
                    # Wait for session to be ready (reuses session creation logic above)
                    if not self.session_ready.is_set():
                        self.logger.info("⏳ Waiting for session ready before sending text...")
                        await self.session_ready.wait()
                        self.logger.info("✅ Session ready - proceeding with text input")
                    
                    # Send text as conversation item to OpenAI
                    await self._send_text_to_openai(envelope)
                        
                else:
                    # Handle other message types
                    api_msg = await self.serializer.safe_serialize(envelope)
                    
                    if api_msg and self.session_manager.is_connected():
                        await self.session_manager.websocket.send(json.dumps(api_msg))
                        self.metrics['messages_sent_to_openai'] += 1
                        self.logger.debug(f"Sent to OpenAI: {api_msg['type']}")
                    elif api_msg:
                        self.logger.warn("Message serialized but no active session")
                    
        except asyncio.TimeoutError:
            # No message - normal when no audio input
            pass
        except Exception as e:
            # Throttle error messages to avoid flooding
            if not hasattr(self, '_last_bridge_error_time') or time.time() - self._last_bridge_error_time > 5.0:
                self.logger.error(f"Error processing bridge message: {e}")
                self._last_bridge_error_time = time.time()
            
    async def _manage_session_lifecycle(self):
        """Handle session cycling and limits"""
        # Check for pause-based cycling (primary strategy)
        if (self.session_manager.state == SessionState.ACTIVE and 
            self.pause_detector.check_pause_condition()):
            
            await self._cycle_session_on_pause()
            
        # Check for limit-based cycling (fallback)
        elif self.session_manager.check_session_limits():
            await self._cycle_session_on_limits()
            
    async def _cycle_session_on_pause(self):
        """Cycle session due to pause detection - only if all responses complete"""
        # Check if we're still waiting for responses
        pending_count = sum(self.pending_responses.values())
        if pending_count > 0:
            # Check for timeout
            if self.response_timeout_start and time.time() - self.response_timeout_start > self.response_timeout_seconds:
                self.logger.warning(f"⏰ Response timeout after {self.response_timeout_seconds}s - forcing cycle")
                self._clear_response_expectations()
            else:
                # Throttle log messages - only log every 5 seconds
                current_time = time.time()
                if current_time - self.last_waiting_log_time >= self.waiting_log_interval:
                    self.logger.info(f"⏳ Pause detected but waiting for {pending_count} responses - delaying cycle")
                    self.last_waiting_log_time = current_time
                return
            
        # Check if session creation/cycling is already in progress
        if self.session_creating:
            self.logger.info("⏳ Session creation in progress, delaying cycle")
            return
            
        self.logger.info("🔄 Cycling session on pause (all responses complete)")
        
        context = await self.session_manager.close_session()
        self.prepared_context = context
        self.metrics['sessions_cycled_on_pause'] += 1
        
        # Reset response tracking and pause detector for next conversation
        self._reset_response_tracking()
        self.pause_detector.reset()
        
        self.logger.info("✅ Session cycled - ready for next speech")
        
    async def _cycle_session_on_limits(self):
        """Cycle session due to time/cost limits - seamless rotation"""
        # Check if session creation is already in progress
        if self.session_creating:
            self.logger.info("⏳ Session creation in progress, delaying limits cycle")
            return
            
        self.logger.info("🔄 Cycling session on limits (seamless rotation)")
        
        self.session_creating = True
        try:
            context = await self.session_manager.close_session() 
            
            # Immediately reconnect with context (no pause to wait)
            if context:
                success = await self.session_manager.connect_session(context)
                if success:
                    self.metrics['sessions_cycled_on_limits'] += 1
                    self.pause_detector.reset()
                    self._reset_response_tracking()
                    self.session_ready.set()  # Set immediately - session is ready
                    self.logger.info("✅ Session rotated seamlessly")
                else:
                    self.logger.error("❌ Failed to rotate session - preparing for next message")
                    self.prepared_context = context
        finally:
            self.session_creating = False
            
    async def _process_openai_responses(self):
        """Process responses from OpenAI WebSocket"""
        if not self.session_manager.is_connected():
            return
            
        try:
            # Check if we have a response processor task running
            if not hasattr(self, '_response_processor_task') or self._response_processor_task.done():
                # Start continuous response processor
                self.logger.info(f"🚀 Starting continuous response processor")
                self._response_processor_task = asyncio.create_task(self._continuous_response_processor())
                
        except Exception as e:
            self.logger.error(f"Error managing response processor: {e}")
            
    async def _continuous_response_processor(self):
        """Continuously process responses from OpenAI WebSocket"""
        try:
            self.logger.info("🎧 Starting continuous OpenAI response listener")
            
            while self.session_manager.is_connected() and self.running:
                try:
                    websocket = self.session_manager.websocket
                    if not websocket:
                        self.logger.warning("🎧 No websocket available, waiting...")
                        await asyncio.sleep(0.1)
                        continue
                        
                    # Continuously listen for messages (no timeout)
                    # Continuously listen - no debug spam
                    message = await websocket.recv()
                    self.logger.debug(f"🎧 Received message from OpenAI")
                    data = json.loads(message)
                    event_type = data.get('type', 'unknown')
                    
                    # Log events with appropriate detail level
                    if event_type in ['response.audio.delta', 'response.audio_transcript.delta']:
                        # Audio deltas are frequent - log at debug level
                        self.logger.debug(f"🎵 OpenAI: {event_type} ({len(message)} chars)")
                    elif event_type in ['session.updated', 'response.created', 'conversation.item.created']:
                        # Important events - log with detail
                        self.logger.info(f"🎯 OpenAI: {event_type}")
                        self.logger.debug(f"   Event data: {json.dumps(data, indent=2)[:200]}...")
                    else:
                        # Other events - standard logging
                        self.logger.info(f"🎯 OpenAI: {event_type}")
                        
                    await self._handle_openai_event(data)
                    
                except websockets.exceptions.ConnectionClosed:
                    self.logger.warning("⚠️ OpenAI connection closed")
                    break
                except json.JSONDecodeError as e:
                    self.logger.error(f"❌ Invalid JSON from OpenAI: {e}")
                except Exception as e:
                    self.logger.error(f"❌ Error in response processor: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ Fatal error in continuous response processor: {e}")
        finally:
            self.logger.info("🛑 Stopped OpenAI response listener")
            
    async def _handle_openai_event(self, data: Dict):
        """Handle individual OpenAI Realtime API events"""
        event_type = data.get("type", "")
        
        try:
            if event_type == "response.audio_transcript.done":
                await self._handle_assistant_transcript_complete(data)
                
            elif event_type == "response.audio.delta":
                await self._handle_audio_delta(data)
                
            elif event_type == "input_audio_buffer.speech_started":
                self.pause_detector.record_message("speech_started")
                self.logger.info("🎤 OpenAI detected speech start")
                # Handle user interruption during assistant response
                await self._handle_user_interruption()
                
            elif event_type == "conversation.item.created":
                # Check if this is the assistant's response starting
                item = data.get("item", {})
                if item.get("role") == "assistant":
                    # Store the assistant item ID for potential truncation
                    self.last_assistant_item_id = item.get("id")
                    self.logger.info(f"🤖 Assistant starting response (item: {self.last_assistant_item_id})")
                    # Don't mark complete here - wait for actual content
                self.logger.debug(f"📝 OpenAI created conversation item: role={item.get('role')}")
                
            elif event_type == "conversation.item.input_audio_transcription.delta":
                # Partial transcription - we can ignore these for cleaner logs
                pass
                
            elif event_type == "conversation.item.input_audio_transcription.completed":
                await self._handle_user_transcript(data)
                
            elif event_type == "input_audio_buffer.speech_stopped": 
                self.logger.debug("🎤 OpenAI detected speech stop")
                
            elif event_type == "input_audio_buffer.committed":
                self.logger.info("💾 OpenAI committed audio buffer")
                
            elif event_type == "response.created":
                self.logger.info("🤖 OpenAI creating response...")
                # Clear assistant transcript buffer for new response
                self.assistant_transcript_buffer = ""
                self._mark_response_complete('assistant_response')
                
            elif event_type == "response.output_item.added":
                self.logger.debug("📝 OpenAI added response item")
                
            elif event_type == "response.content_part.added":
                self.logger.debug("📋 OpenAI added response content part")
                
            elif event_type == "response.audio_transcript.delta":
                # Accumulate assistant transcript deltas (streaming text)
                delta_text = data.get("delta", "")
                if delta_text:
                    self.assistant_transcript_buffer += delta_text
                    self.logger.debug(f"📝 Assistant delta: +{len(delta_text)} chars")
                    
            elif event_type == "response.audio_transcript.done":
                await self._handle_assistant_transcript_complete(data)
                
            elif event_type == "response.audio.transcript.delta":
                # Legacy event type (keeping for compatibility)
                transcript = data.get("delta", "")
                self.logger.debug(f"📝 Legacy transcript delta: {transcript}")
                
            elif event_type == "response.audio.delta":
                await self._handle_audio_delta(data)
                
            elif event_type == "response.done":
                self.pause_detector.mark_llm_response_complete()
                self.logger.info("🤖 Assistant response complete")
                self._mark_response_complete('audio_complete')
                
            elif event_type == "session.created":
                # Session already marked ready during connection, this is just for logging
                session_info = data.get('session', {})
                session_id = session_info.get('id', 'unknown')
                self.logger.debug(f"📝 Received session.created confirmation: {session_id}")
                # Ensure session_ready is set (defensive programming)
                if not self.session_ready.is_set():
                    self.session_ready.set()
                
            elif event_type == "session.updated":
                session_config = data.get('session', {})
                turn_detection = session_config.get('turn_detection', {})
                self.logger.info(f"📝 OpenAI session updated - turn_detection: {turn_detection.get('type')}")
                
            elif event_type == "error":
                await self._handle_openai_error(data)
                
            else:
                # Log unhandled events
                self.logger.debug(f"🔍 Unhandled OpenAI event: {event_type}")
                if event_type not in ['response.audio.delta', 'response.audio_transcript.delta']:
                    self.logger.debug(f"   Event data: {json.dumps(data, indent=2)[:100]}...")
                
        except Exception as e:
            self.logger.error(f"Error handling OpenAI event {event_type}: {e}")
            
        
    async def _handle_audio_delta(self, data: Dict):
        """Handle audio response delta - convert base64 PCM to ROS audio"""
        audio_b64 = data.get("delta", "")
        if not audio_b64:
            self.logger.debug("🔊 Empty audio delta received")
            return
            
        if not self.bridge_interface:
            self.logger.debug("🔊 Audio delta received but no bridge interface")
            return
            
        try:
            # Decode base64 PCM audio data
            audio_data = base64.b64decode(audio_b64)
            if len(audio_data) == 0:
                self.logger.debug("🔊 Empty audio data after decoding")
                return
                
            # Convert to int16 array for ROS
            audio_array_24k = np.frombuffer(audio_data, dtype=np.int16)
            
            # Resample from 24kHz to 16kHz using simple 3:2 decimation
            # For every 3 samples at 24kHz, output 2 samples at 16kHz
            audio_array_16k = self._simple_resample_24k_to_16k(audio_array_24k)
            
            # Send audio as-is to preserve timing
            audio_data_dict = {"int16_data": audio_array_16k.tolist()}
            
            # Send to ROS via bridge if audio output is enabled
            if self.published_topics['response_voice']:  # Skip if topic is empty/disabled
                success = await self.bridge_interface.put_outbound_message(
                    self.published_topics['response_voice'], 
                    audio_data_dict, 
                    "audio_common_msgs/AudioData"
                )
                
                if success:
                    self.metrics['messages_sent_to_ros'] += 1
                    self.logger.debug(f"🔊 Audio delta sent: {len(audio_array_16k)} samples @ 16kHz (from {len(audio_array_24k)} @ 24kHz)")
                else:
                    self.logger.warning("⚠️ Failed to send audio delta to ROS")
            else:
                self.logger.debug("🔇 Audio output disabled for this agent")
                
        except Exception as e:
            self.logger.error(f"❌ Error processing audio delta: {e}")
            
        # Mark LLM response as active for pause detection
        self.pause_detector.mark_llm_response_active()
        
    async def _handle_user_transcript(self, data: Dict):
        """Handle completed user transcript"""
        transcript = data.get("transcript", "").strip()
        if transcript:
            self.session_manager.add_conversation_turn("user", transcript)
            self.logger.info(f"👤 User transcript: {transcript}")
            self._mark_response_complete('transcription')
            
            # NEW: Publish user transcript to ROS (was missing before!)
            if self.bridge_interface and self.published_topics.get('prompt_transcript'):
                transcript_data = {"data": transcript}
                success = await self.bridge_interface.put_outbound_message(
                    self.published_topics['prompt_transcript'],
                    transcript_data,
                    "std_msgs/String"
                )
                if success:
                    self.logger.info("📤 User transcript published to ROS")
            
            # Manually trigger response since OpenAI server VAD doesn't automatically respond reliably
            # Keep manual triggering for now while we investigate interruption separately
            if self.pending_responses.get('assistant_response', False):
                self.logger.info("🤖 Triggering OpenAI response generation")
                try:
                    response_msg = self.serializer.create_response_trigger()
                    await self.session_manager.websocket.send(json.dumps(response_msg))
                    self.logger.info("✅ Response generation triggered")
                except Exception as e:
                    self.logger.error(f"❌ Failed to trigger response: {e}")
        else:
            self.logger.warning("⚠️ Empty user transcript received")
            self._mark_response_complete('transcription')
            
    async def _handle_assistant_transcript_complete(self, data: Dict):
        """Handle completed assistant transcript"""
        # Get final transcript from buffer or data
        final_transcript = self.assistant_transcript_buffer.strip()
        if not final_transcript:
            # Fallback to transcript in event data
            final_transcript = data.get("transcript", "").strip()
            
        if final_transcript:
            # Add to conversation context
            self.session_manager.add_conversation_turn("assistant", final_transcript)
            self.logger.info(f"🤖 Assistant transcript: {final_transcript}")
            
            # Send transcript to ROS via WebSocket
            if self.bridge_interface:
                transcript_data = {"data": final_transcript}

                # Determine which topic to use based on agent configuration
                # Command agents publish to response_cmd, conversation agents to response_text
                if self.published_topics.get('response_cmd') and not self.published_topics.get('response_text'):
                    # This is a command agent - publish to response_cmd
                    topic = self.published_topics['response_cmd']
                    topic_name = "response_cmd"
                elif self.published_topics.get('response_text'):
                    # This is a conversation agent - publish to response_text
                    topic = self.published_topics['response_text']
                    topic_name = "response_text"
                else:
                    # Fallback - shouldn't happen
                    topic = None
                    topic_name = None

                if topic:
                    success = await self.bridge_interface.put_outbound_message(
                        topic,
                        transcript_data,
                        "std_msgs/String"
                    )

                    if success:
                        self.metrics['messages_sent_to_ros'] += 1
                        self.logger.info(f"📤 Assistant transcript sent to ROS on {topic_name}")

                        # For command extractor: check if this looks like a command
                        if (self.published_topics.get('command_detected') and
                            final_transcript.startswith("COMMAND:")):
                            # Publish command detected signal
                            command_signal = {"data": True}
                            await self.bridge_interface.put_outbound_message(
                                self.published_topics['command_detected'],
                                command_signal,
                                "std_msgs/Bool"
                            )
                            self.logger.info("🤖 Command detected and signaled")
                    else:
                        self.logger.warning(f"⚠️ Failed to send assistant transcript to ROS on {topic_name}")
                else:
                    self.logger.warning("⚠️ No output topic configured for assistant transcript")
        else:
            self.logger.warning("⚠️ Empty assistant transcript received")
            
        # Clear the buffer
        self.assistant_transcript_buffer = ""
        self.pause_detector.mark_llm_response_complete()
            
    async def _trigger_response(self):
        """Manually trigger OpenAI to generate a response (for testing/debug)"""
        if not self.session_manager.is_connected():
            self.logger.warning("⚠️ Cannot trigger response - no active session")
            return
            
        try:
            # Send response.create message to explicitly request a response
            response_msg = {
                "type": "response.create",
                "response": {
                    "modalities": ["text", "audio"],
                    "instructions": "Please respond to the user's message naturally and helpfully."
                }
            }
            
            await self.session_manager.websocket.send(json.dumps(response_msg))
            self.logger.info("🎯 Manually triggered OpenAI response generation")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to trigger response: {e}")
            
    async def _handle_openai_error(self, data: Dict):
        """Handle OpenAI API errors"""
        error_code = data.get("error", {}).get("code", "unknown")
        error_message = data.get("error", {}).get("message", "Unknown error")
        
        self.logger.error(f"🚨 OpenAI API error [{error_code}]: {error_message}")
        
        # Handle specific error types
        if error_code in ["invalid_api_key", "insufficient_quota"]:
            self.logger.error("❌ API key or quota issue - stopping agent")
            self.running = False
        elif error_code == "rate_limit_exceeded":
            self.logger.warn("⏰ Rate limited - will retry connection")
            # Close current session, will retry on next message
            await self.session_manager.close_session()
            
    def _setup_response_expectations(self):
        """Set up response tracking after committing audio buffer"""
        self.pending_responses = {
            'transcription': True,      # Expect user transcript
            'assistant_response': True, # Expect assistant to respond  
            'audio_complete': True      # Expect response completion
        }
        self.response_timeout_start = time.time()  # Start timeout timer
        self.last_waiting_log_time = 0  # Reset throttling timer
        self.logger.info("⏳ Expecting transcription + assistant response")
        
    def _mark_response_complete(self, response_type: str):
        """Mark a response type as complete"""
        if response_type in self.pending_responses:
            self.pending_responses[response_type] = False
            self.logger.info(f"✅ {response_type} complete")
            self._check_cycle_readiness()
            
    def _clear_response_expectations(self):
        """Clear all pending response expectations (for timeout recovery)"""
        self.pending_responses = {
            'transcription': False,
            'assistant_response': False,
            'audio_complete': False
        }
        self.response_timeout_start = None
        self.last_waiting_log_time = 0  # Reset throttling timer
        self.logger.info("🚫 Cleared all response expectations")
        
    async def _send_text_to_openai(self, envelope):
        """Send text input as a conversation item to OpenAI"""
        try:
            # Use existing serializer that already handles text format
            api_msg = self.serializer.serialize_text(envelope)
            
            if api_msg and self.session_manager.is_connected():
                await self.session_manager.websocket.send(json.dumps(api_msg))
                self.metrics['messages_sent_to_openai'] += 1
                self.logger.info(f"✅ Sent text to OpenAI: '{envelope.raw_data.data[:50]}...'")
                
                # Trigger response creation immediately for text input
                response_msg = self.serializer.create_response_trigger()
                await self.session_manager.websocket.send(json.dumps(response_msg))
                self.logger.info("🚀 Triggered response creation for text input")
                
                # Set up response expectations (text also expects responses)
                self._setup_response_expectations()
                
            else:
                self.logger.warning("⚠️ Text message ready but no active session")
                
        except Exception as e:
            self.logger.error(f"❌ Error sending text to OpenAI: {e}")
            
    async def send_text_to_llm(self, text: str) -> bool:
        """
        Send text input to the LLM as a conversation item.
        
        This is a common interface method that should be implemented by all agents.
        Different LLM providers may handle text differently internally.
        
        Args:
            text: The text message to send to the LLM
            
        Returns:
            bool: True if message was sent successfully
        """
        # For OpenAI, we need the full envelope for metadata
        # Create a minimal envelope for the text
        envelope = type('TextEnvelope', (object,), {
            'raw_data': type('RawData', (object,), {'data': text})(),
            'ros_msg_type': 'std_msgs/String',
            'topic_name': 'text_input'
        })()
        
        try:
            await self._send_text_to_openai(envelope)
            return True
        except Exception as e:
            self.logger.error(f"Failed to send text to LLM: {e}")
            return False
            
    def _check_cycle_readiness(self):
        """Check if all responses are complete and we can cycle session"""
        self._continue_check_cycle_readiness()
        
    def _simple_resample_24k_to_16k(self, audio_data):
        """Simple 3:2 decimation from 24kHz to 16kHz
        
        For every 3 samples at 24kHz, we output 2 samples at 16kHz.
        Uses linear interpolation for smoother resampling.
        """
        if len(audio_data) == 0:
            return np.array([], dtype=np.int16)
            
        # Calculate output size (2/3 of input)
        output_size = int(len(audio_data) * 2 / 3)
        output = np.zeros(output_size, dtype=np.int16)
        
        # Resample using linear interpolation
        for i in range(output_size):
            # Map output index to input index
            input_idx = i * 1.5  # 3/2 ratio
            idx_int = int(input_idx)
            fraction = input_idx - idx_int
            
            # Linear interpolation
            if idx_int + 1 < len(audio_data):
                sample1 = audio_data[idx_int]
                sample2 = audio_data[idx_int + 1]
                output[i] = int(sample1 * (1 - fraction) + sample2 * fraction)
            elif idx_int < len(audio_data):
                output[i] = audio_data[idx_int]
                
        return output
        
    def _continue_check_cycle_readiness(self):
        """Continue the cycle readiness check logic"""
        all_complete = not any(self.pending_responses.values())
        pending_count = sum(self.pending_responses.values())
        
        if all_complete:
            self.logger.info("🔄 All responses complete - ready to cycle session")
            self.response_timeout_start = None  # Clear timeout timer
            # Reset pause detector since we're ready for the next utterance
            self.pause_detector.reset()
        else:
            self.logger.debug(f"⏳ Still waiting for {pending_count} responses")
            
    def _reset_response_tracking(self):
        """Reset response tracking for new utterance"""
        self.pending_responses = {
            'transcription': False,
            'assistant_response': False, 
            'audio_complete': False
        }
        self.response_timeout_start = None  # Clear timeout timer
        # Clear assistant transcript buffer
        self.assistant_transcript_buffer = ""
            
    async def _cleanup(self):
        """Clean up resources"""
        if self.start_time:
            self.metrics['total_runtime'] = time.time() - self.start_time
            
        self.logger.info("🧹 Cleaning up OpenAI Realtime Agent...")
        
        # Stop conversation monitoring
        await self.conversation_monitor.stop_monitoring()
        
        # Stop response processor task
        if hasattr(self, '_response_processor_task') and not self._response_processor_task.done():
            self._response_processor_task.cancel()
            try:
                await self._response_processor_task
            except asyncio.CancelledError:
                pass
        
        if self.session_manager.state == SessionState.ACTIVE:
            await self.session_manager.close_session()
            
        if self.bridge_interface:
            await self.bridge_interface.close()
            
        if self.debug_interface:
            await self.debug_interface.stop()
        
        # Log final metrics
        final_metrics = self.get_metrics()
        self.logger.info(f"📊 Final metrics: {final_metrics}")
        self.logger.info("✅ Agent cleanup complete")
        
    async def stop(self):
        """Stop the agent gracefully"""
        self.logger.info("🛑 Stop signal received")
        self.running = False
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive agent metrics"""
        # Combine all metrics
        combined_metrics = self.metrics.copy()
        
        # Add component metrics
        combined_metrics.update({
            'serializer': self.serializer.get_metrics(),
            'pause_detector': self.pause_detector.get_metrics(),
            'session_manager': self.session_manager.get_metrics(),
            'conversation_monitor': self.conversation_monitor.get_metrics(),
            'bridge_interface': self.bridge_interface.get_metrics() if self.bridge_interface else {}
        })
        
        # Calculate runtime
        if self.start_time:
            combined_metrics['current_runtime'] = time.time() - self.start_time
            
        return combined_metrics
        
    def get_status_summary(self) -> str:
        """Get human-readable status summary"""
        session_status = f"Session: {self.session_manager.state.value}"
        bridge_status = self.bridge_interface.get_status_summary() if self.bridge_interface else "Bridge: Disconnected"
        pause_status = self.pause_detector.get_status_summary()
        conv_status = self.conversation_monitor.get_status_summary()
        message_stats = f"Processed: {self.metrics['messages_processed']}"
        
        return f"{session_status} | {bridge_status} | {pause_status} | {conv_status} | {message_stats}"
        
    def update_configuration(self, new_config: Dict):
        """Update agent configuration at runtime"""
        old_config = self.config.copy()
        self.config.update(new_config)
        
        # Update components
        if 'session_pause_timeout' in new_config:
            self.pause_detector.update_pause_timeout(new_config['session_pause_timeout'])
            
        self.session_manager.update_configuration(new_config)
        
        self.logger.info("⚙️ Configuration updated")
        
    async def switch_system_prompt(self, prompt_id: str = None, context_updates: Dict[str, Any] = None) -> bool:
        """Switch system prompt during runtime
        
        Args:
            prompt_id: Specific prompt to switch to (overrides context-based selection)
            context_updates: Updates to prompt selection context (e.g., {'user_age': 8})
            
        Returns:
            bool: True if switch was successful
        """
        success = await self.session_manager.switch_prompt(prompt_id, context_updates)
        
        if success:
            prompt_info = self.session_manager.get_effective_prompt_selection()
            self.logger.info(f"🔄 System prompt switched: {prompt_info['selection_type']} -> {prompt_info.get('prompt_id', 'unknown')}")
        
        return success
        
    def get_current_system_prompt_info(self) -> Dict[str, Any]:
        """Get information about currently active system prompt"""
        return self.session_manager.get_effective_prompt_selection()
        
    def list_available_system_prompts(self) -> List[str]:
        """List all available system prompt IDs"""
        return self.session_manager.list_available_prompts()
        
    def clear_system_prompt_override(self):
        """Clear any prompt override and return to context-based selection"""
        self.session_manager.clear_prompt_override()
        self.logger.info("🔄 Cleared system prompt override - using context-based selection")
        
    def reload_system_prompts(self):
        """Reload system prompts from prompts.yaml file"""
        self.session_manager.reload_prompts()
        self.logger.info("🔄 System prompts reloaded from file")
        
    # Debug interface methods for standalone testing
    async def debug_inject_audio(self, audio_data: List[int], utterance_id: str = None, 
                               confidence: float = 0.95, is_utterance_end: bool = False) -> bool:
        """Inject audio data for testing (standalone mode only)"""
        if not self.debug_interface:
            self.logger.warning("Debug interface not available - agent connected to bridge")
            return False
            
        return await self.debug_interface.inject_audio_data(
            audio_data, utterance_id, confidence, is_utterance_end
        )
        
    async def debug_inject_text(self, text: str) -> bool:
        """Inject text message for testing (standalone mode only)"""
        if not self.debug_interface:
            self.logger.warning("Debug interface not available - agent connected to bridge")
            return False
            
        return await self.debug_interface.inject_text_message(text)
        
    def get_debug_stats(self) -> Dict[str, Any]:
        """Get debug interface statistics"""
        if not self.debug_interface:
            return {"debug_interface": "not_available"}
            
        return self.debug_interface.get_stats()
        
    def is_standalone_mode(self) -> bool:
        """Check if agent is running in standalone mode"""
        return self.bridge_interface is None and self.debug_interface is not None
        
    def _setup_conversation_logging(self):
        """Add conversation ID to all log messages"""
        # Create a custom filter that adds conversation ID
        class ConversationFilter(logging.Filter):
            def __init__(self, monitor):
                self.monitor = monitor
                
            def filter(self, record):
                # Add conversation ID to log record
                record.conversation_id = self.monitor.current_conversation_id[-12:]  # Last 12 chars
                return True
                
        # Add filter to logger
        conv_filter = ConversationFilter(self.conversation_monitor)
        self.logger.addFilter(conv_filter)
        
        # For now, just log conversation ID in messages rather than changing formatter
        # This avoids conflicts with existing log formats
                
    def _handle_conversation_change(self, old_id: str, new_id: str, is_external: bool = False):
        """Handle conversation ID change callback"""
        self.logger.info(f"🎭 CONVERSATION CHANGE: {old_id[-12:]} → {new_id[-12:]}")
        
        # Clear conversation context
        self.session_manager.reset_conversation_context()
        
        # If we have an active session, we'll keep it but with cleared context
        if self.session_manager.state == SessionState.ACTIVE:
            self.logger.info("🔄 Active session will continue with fresh context")
            
        # Only publish conversation ID if this was an internal change (timeout)
        # External changes should not be re-published to avoid loops
        if not is_external and self.bridge_interface and self.bridge_interface.is_connected():
            asyncio.create_task(self._publish_conversation_id(new_id))
            
    async def _publish_conversation_id(self, conversation_id: str):
        """Publish conversation ID to ROS topic"""
        try:
            conv_msg = {"data": conversation_id}
            success = await self.bridge_interface.put_outbound_message(
                "conversation_id",  # Use relative topic name for bridge compatibility
                conv_msg,
                "std_msgs/String"
            )
            if success:
                self.logger.info(f"📤 Published conversation ID: {conversation_id}")
            else:
                self.logger.warning("Failed to publish conversation ID")
        except Exception as e:
            self.logger.error(f"Error publishing conversation ID: {e}")
            
    async def _handle_user_interruption(self):
        """Handle user interruption during assistant response"""
        try:
            # Check if assistant is currently responding (has active response)
            if not self.pause_detector.is_llm_speaking():
                self.logger.debug("👂 User speech detected, but assistant not speaking - no interruption needed")
                return
                
            self.logger.info("⚡ User interruption detected - canceling assistant response")
            
            # 1. Cancel current response
            cancel_msg = self.serializer.create_response_cancel()
            await self.session_manager.websocket.send(json.dumps(cancel_msg))
            self.logger.info("🛑 Sent response.cancel")
            
            # 2. Clear output audio buffer to stop playback immediately  
            clear_msg = self.serializer.create_audio_buffer_clear()
            await self.session_manager.websocket.send(json.dumps(clear_msg))
            self.logger.info("🔇 Cleared output audio buffer")
            
            # 3. Truncate conversation item to prevent partial text in context
            if self.last_assistant_item_id:
                truncate_msg = self.serializer.create_conversation_truncate(
                    self.last_assistant_item_id,
                    audio_end_ms=0  # Start from beginning of audio
                )
                await self.session_manager.websocket.send(json.dumps(truncate_msg))
                self.logger.info(f"✂️ Truncated conversation item: {self.last_assistant_item_id}")
            
            # 4. Clear assistant transcript buffer to prevent partial text pollution
            if self.assistant_transcript_buffer:
                self.logger.info(f"🧹 Clearing partial transcript: '{self.assistant_transcript_buffer[:50]}...'")
                self.assistant_transcript_buffer = ""
            
            # 5. Signal ROS audio player to clear its queue
            if self.published_topics.get('interruption_signal') and self.bridge_interface:
                interrupt_signal = {"data": True}
                await self.bridge_interface.put_outbound_message(
                    self.published_topics['interruption_signal'],
                    interrupt_signal,
                    "std_msgs/Bool"
                )
                self.logger.info("📡 Sent interruption signal to audio player")
            
            # 6. Mark LLM response as complete since we're interrupting it
            self.pause_detector.mark_llm_response_complete()
            self._mark_response_complete('assistant_response')
            self._mark_response_complete('audio_complete')
            
            self.logger.info("✅ Interruption handled - ready for new user input")
            
        except Exception as e:
            self.logger.error(f"❌ Error handling user interruption: {e}")


# Convenience function for standalone usage
async def create_and_run_agent(config: Dict) -> OpenAIRealtimeAgent:
    """Create and run agent with proper error handling"""
    agent = OpenAIRealtimeAgent(config)
    
    try:
        await agent.initialize()
        await agent.run()
    except KeyboardInterrupt:
        agent.logger.info("⌨️ Keyboard interrupt received")
    except Exception as e:
        agent.logger.error(f"❌ Agent error: {e}", exc_info=True)
    finally:
        await agent.stop()
        
    return agent