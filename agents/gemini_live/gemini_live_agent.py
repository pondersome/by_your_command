#!/usr/bin/env python3
"""
Gemini Live Agent - Hybrid Implementation

Based on OpenAI agent's bridge interface pattern but with Gemini-specific
receive generator coordination through minimal middleware.

Author: Karim Virani
Version: 2.0
Date: August 2025
"""

import asyncio
import json
import logging
import time
from typing import Optional, Dict, Any
import numpy as np

from ..common import (
    WebSocketBridgeInterface,
    ConversationMonitor,
    ConversationContext,
    PauseDetector
)
from .gemini_session_manager import GeminiSessionManager
from .gemini_serializer import GeminiSerializer
from .receive_coordinator import ReceiveCoordinator
from ..common.base_session_manager import SessionState


class GeminiLiveAgent:
    """
    Gemini Live agent with clean architecture.
    
    Uses OpenAI's proven bridge interface pattern with Gemini-specific
    receive generator coordination through minimal middleware.
    """
    
    def __init__(self, config: Dict):
        """Initialize agent with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Agent identification
        self.agent_id = config.get('agent_id', 'gemini_live')
        
        # Core components (following OpenAI pattern)
        self.bridge_interface: Optional[WebSocketBridgeInterface] = None
        self.session_manager = GeminiSessionManager(config)
        self.serializer = GeminiSerializer()
        
        # Conversation monitoring
        conversation_timeout = config.get('conversation_timeout', 600.0)
        self.conversation_monitor = ConversationMonitor(
            timeout=conversation_timeout,
            on_conversation_change=self._handle_conversation_change
        )
        
        # Pause detection for session cycling
        pause_timeout = config.get('session_pause_timeout', 10.0)
        self.logger.info(f"Setting pause detector timeout to {pause_timeout}s")
        self.pause_detector = PauseDetector(
            pause_timeout=pause_timeout
        )
        
        # NEW: Receive coordinator (the minimal middleware)
        self.receive_coordinator: Optional[ReceiveCoordinator] = None
        
        # State tracking
        self.running = False
        self.session_creating = False
        self.start_time: Optional[float] = None
        
        # Image/Video support
        self.video_enabled = config.get('enable_video', False)
        self.latest_image_frame = None  # Store latest image frame
        self.latest_image_timestamp = None
        self.image_frames_received = 0
        self.image_frames_sent = 0
        self.max_image_age = config.get('max_image_age', 5.0)  # Max age in seconds for image to be considered fresh
        
        if self.video_enabled:
            self.logger.info("📷 Video support ENABLED - will subscribe to camera topic")
        else:
            self.logger.info("📷 Video support DISABLED")
        
        # Metrics
        self.metrics = {
            'messages_processed': 0,
            'audio_chunks_sent': 0,
            'text_messages_sent': 0,
            'image_frames_received': 0,
            'image_frames_sent': 0,
            'responses_received': 0,
            'sessions_created': 0,
            'errors': 0
        }
        
        # Published topics configuration
        # Using new naming convention: prompt_* for user input, response_* for agent output
        # Determine topic based on agent type
        is_command_agent = 'command' in self.agent_id.lower()
        
        self.published_topics = {
            'response_voice': '' if is_command_agent else config.get('response_voice_topic', 
                                                                     config.get('audio_out_topic', 'response_voice')),
            'response_text': '' if is_command_agent else config.get('response_text_topic',
                                                                    config.get('transcript_topic', 'response_text')),
            'response_cmd': config.get('response_cmd_topic', 'response_cmd') if is_command_agent else '',
            'prompt_transcript': config.get('prompt_transcript_topic', 'prompt_transcript'),
            'interruption_signal': config.get('interruption_signal_topic', 'interruption_signal')
        }
        
        self.logger.info(f"Initialized Gemini Live Agent '{self.agent_id}'")
        
    async def initialize(self):
        """Initialize agent components"""
        self.logger.info("Initializing Gemini Live Agent...")
        
        # Start conversation monitoring
        await self.conversation_monitor.start_monitoring()
        
        # Connect to ROS AI Bridge (following OpenAI pattern exactly)
        await self._connect_to_bridge()
        
        # Initialize the receive coordinator with bridge and session manager
        self.receive_coordinator = ReceiveCoordinator(
            self.bridge_interface,
            self.session_manager,
            self.published_topics
        )
        
        self.logger.info("✅ Gemini Live Agent initialized")
        
    async def _connect_to_bridge(self):
        """Connect to ROS AI Bridge via WebSocket (following OpenAI pattern)"""
        try:
            self.logger.info("Connecting to ROS AI Bridge via WebSocket...")
            
            # Create bridge config (matching OpenAI pattern)
            bridge_config = {
                'agent_id': self.agent_id,
                'enable_video': self.video_enabled,  # Pass video flag for subscription
                'bridge_connection': {
                    'host': 'localhost',
                    'port': 8765,
                    'reconnect_interval': 5.0,
                    'max_reconnect_attempts': 10
                }
            }
            
            # Create bridge interface with config dict
            self.bridge_interface = WebSocketBridgeInterface(bridge_config)
            
            # Connect with initial attempt tracking
            # Registration happens automatically during connect
            success = await self.bridge_interface.connect_with_retry()
            
            if success:
                self.logger.info("✅ Connected to ROS AI Bridge")
            else:
                raise ConnectionError("Failed to connect to bridge after retries")
                
        except Exception as e:
            self.logger.error(f"Bridge connection error: {e}")
            self.logger.warning("Running in standalone mode without bridge connection")
            self.bridge_interface = None
            
    async def run(self):
        """Main agent loop (simplified from OpenAI)"""
        self.running = True
        self.start_time = time.time()
        self.logger.info(f"🚀 Gemini Live Agent '{self.agent_id}' starting...")
        
        try:
            while self.running:
                # CRITICAL CHANGE: Ensure session is ready BEFORE processing messages (OpenAI pattern)
                await self._ensure_session_ready()
                
                # Process incoming messages from bridge
                await self._process_bridge_messages()
                
                # Check for session limits (Gemini has strict time limits)
                if self.session_manager.check_session_limits():
                    await self._cycle_session_on_limits()
                
                # Check for conversation pause
                elif self.pause_detector.check_pause_condition():
                    await self._cycle_session_on_pause()
                
                # Small sleep to prevent busy loop
                await asyncio.sleep(0.01)
                
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Fatal error in agent loop: {e}")
        finally:
            await self.cleanup()
            
    async def _process_bridge_messages(self):
        """Process messages from bridge (simplified from OpenAI)"""
        try:
            # Skip if bridge not connected
            if not self.bridge_interface or not self.bridge_interface.is_connected():
                return
                
            # Get message with short timeout
            envelope = await self.bridge_interface.get_inbound_message(timeout=0.1)
            
            if envelope is None:
                return
                
            if envelope:
                self.pause_detector.record_message(envelope.ros_msg_type)
                self.metrics['messages_processed'] += 1
                # Reduce spam - only log non-image messages at info level
                if "Image" not in envelope.ros_msg_type:
                    self.logger.info(f"📨 Processing: {envelope.ros_msg_type}")
                else:
                    self.logger.debug(f"📨 Processing: {envelope.ros_msg_type}")
                
                # Handle image frames separately (store latest, don't send immediately)
                if envelope.ros_msg_type in ["sensor_msgs/Image", "sensor_msgs/CompressedImage"]:
                    # Check if this is a triggered frame
                    triggered = envelope.metadata and envelope.metadata.get('triggered_by') == 'voice_detection'
                    if triggered:
                        self.logger.info(f"🎯📷 Received TRIGGERED image frame from bridge! Age: {envelope.metadata.get('frame_age_ms')}ms")
                    else:
                        self.logger.debug(f"📷 Received compressed image frame from bridge!")
                    await self._handle_image_frame(envelope)
                    return
                
                # Session should already be ready from _ensure_session_ready() in main loop
                if self.session_manager.state != SessionState.ACTIVE:
                    self.logger.warning("⚠️ Message received but no active session - skipping")
                    return
                
                # If we have a stored image and video is enabled, include it with this interaction
                if self.video_enabled and self.latest_image_frame and self.receive_coordinator:
                    # Send the latest image frame before the audio/text
                    self.logger.info(f"🎯 Sending image with {envelope.ros_msg_type} interaction ({len(self.latest_image_frame)} bytes)")
                    await self._send_latest_image_to_session()
                else:
                    # Debug why image wasn't sent
                    if not self.video_enabled:
                        self.logger.debug("❌ Video not enabled")
                    elif not self.latest_image_frame:
                        self.logger.warning("❌ No stored image frame available")
                    elif not self.receive_coordinator:
                        self.logger.error("❌ No receive coordinator")
                    else:
                        self.logger.error("❌ Unknown reason for not sending image")
                
                # Delegate to receive coordinator (the middleware)
                await self.receive_coordinator.handle_message(envelope)
                
        except Exception as e:
            self.logger.error(f"Error processing bridge message: {e}")
            self.metrics['errors'] += 1
            
    async def _ensure_session_ready(self):
        """Ensure we have an active session when needed (OpenAI pattern)"""
        if self.session_manager.state == SessionState.IDLE and not self.session_creating:
            self.session_creating = True
            try:
                self.logger.info("🔗 Creating Gemini session proactively...")
                success = await self.session_manager.connect_session()
                if success:
                    self.pause_detector.reset()
                    self.metrics['sessions_created'] += 1
                    self.logger.info("✅ Session ready for per-turn communication")
                else:
                    self.logger.error("❌ Failed to create session")
            finally:
                self.session_creating = False
                
    async def _cycle_session_on_limits(self):
        """Cycle session when approaching Gemini's time limits"""
        self.logger.info("🔄 Cycling session due to time limits")
        
        # Get context before closing
        context = await self.session_manager.close_session()
        
        # Create new session with context
        if context:
            success = await self.session_manager.connect_session(context)
            if success:
                self.logger.info("✅ Session cycled with context preserved")
            else:
                self.logger.error("❌ Failed to cycle session")
                
    async def _cycle_session_on_pause(self):
        """Cycle session on conversation pause"""
        self.logger.info("🔄 Cycling session on pause")
        
        # Close current session
        await self.session_manager.close_session()
        
        # Don't create new session yet - wait for next message
        self.pause_detector.reset()
        self.logger.info("Session closed, waiting for next interaction")
        
    def _handle_conversation_change(self, old_id: str, new_id: str, is_external: bool):
        """Handle conversation ID change"""
        self.logger.info(f"🔄 Conversation changed: {old_id} → {new_id}")
        # Reset conversation context
        self.session_manager.reset_conversation_context()
    
    async def _handle_image_frame(self, envelope):
        """Handle incoming image frame from camera
        
        Store the latest frame for use when audio/text interaction occurs.
        This implements the "latest frame" pattern to avoid overwhelming Gemini.
        """
        try:
            self.image_frames_received += 1
            
            # Extract image data from ROS message
            image_msg = envelope.raw_data
            
            # Handle CompressedImage (JPEG data) or regular Image
            if hasattr(image_msg, 'data'):
                # Check if data is placeholder string (temporary fix)
                if isinstance(image_msg.data, str) and "skipped" in image_msg.data:
                    self.logger.debug(f"Skipping placeholder image data: {image_msg.data}")
                    return
                    
                # Store the raw image data (JPEG bytes for CompressedImage)
                self.latest_image_frame = bytes(image_msg.data)
                self.latest_image_timestamp = envelope.timestamp
                
                # Log periodically to avoid spam
                if self.image_frames_received % 10 == 1:
                    if hasattr(image_msg, 'format'):
                        # CompressedImage
                        self.logger.info(f"📷 Stored compressed frame #{self.image_frames_received} "
                                       f"({image_msg.format}, {len(image_msg.data)} bytes)")
                    else:
                        # Regular Image
                        self.logger.info(f"📷 Stored image frame #{self.image_frames_received} "
                                       f"({image_msg.width}x{image_msg.height}, {image_msg.encoding})")
            else:
                self.logger.warning("Image message missing data field")
                
        except Exception as e:
            self.logger.error(f"Error handling image frame: {e}")
    
    async def _send_latest_image_to_session(self):
        """Send the latest stored image frame to Gemini session
        
        This is called when an audio/text interaction starts, providing
        visual context for the conversation.
        """
        if not self.latest_image_frame:
            self.logger.warning("❌ No image frame to send")
            return
        if not self.session_manager.session:
            self.logger.warning("❌ No active Gemini session for image")
            return
            
        try:
            # Check if image is recent enough (configurable max age)
            if self.latest_image_timestamp:
                age = time.time() - self.latest_image_timestamp
                if age > self.max_image_age:
                    self.logger.warning(f"⏰ Image frame too old ({age:.1f}s > {self.max_image_age}s), skipping")
                    return
                else:
                    self.logger.info(f"✅ Image age OK: {age:.1f}s")
            
            # Send image using session.send() exactly like Google's example
            import base64
            
            # Convert raw JPEG bytes to base64 string (Google's example pattern)
            image_b64 = base64.b64encode(self.latest_image_frame).decode()
            
            # Send using the unified realtime API (same as audio)
            await self.session_manager.session.send(
                input={
                    "mime_type": "image/jpeg",
                    "data": image_b64
                }
            )
            
            self.image_frames_sent += 1
            self.logger.info(f"🖼️ Sent image frame to Gemini via session.send() (frame #{self.image_frames_sent})")
            
        except Exception as e:
            self.logger.error(f"Error sending image to session: {e}")
        
    async def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.logger.info("🧹 Cleaning up Gemini Live Agent...")
        
        # Stop conversation monitoring
        await self.conversation_monitor.stop_monitoring()
        
        # Clean up receive coordinator
        if self.receive_coordinator:
            await self.receive_coordinator.cleanup()
        
        # Close session
        if self.session_manager.state == SessionState.ACTIVE:
            await self.session_manager.close_session()
            
        # Close bridge
        if self.bridge_interface:
            await self.bridge_interface.close()
            
        self.logger.info("✅ Gemini Live Agent cleanup complete")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        combined_metrics = self.metrics.copy()
        
        # Add component metrics
        combined_metrics.update({
            'session_manager': self.session_manager.get_metrics(),
            'conversation_monitor': self.conversation_monitor.get_metrics(),
            'bridge_interface': self.bridge_interface.get_metrics() if self.bridge_interface else {},
            'receive_coordinator': self.receive_coordinator.get_metrics() if self.receive_coordinator else {}
        })
        
        # Add runtime
        if self.start_time:
            combined_metrics['runtime'] = time.time() - self.start_time
            
        return combined_metrics