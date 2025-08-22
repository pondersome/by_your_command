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
        
        # Metrics
        self.metrics = {
            'messages_processed': 0,
            'audio_chunks_sent': 0,
            'text_messages_sent': 0,
            'responses_received': 0,
            'sessions_created': 0,
            'errors': 0
        }
        
        # Published topics configuration
        self.published_topics = {
            'audio_out': config.get('audio_out_topic', 'audio_out'),
            'transcript': config.get('transcript_topic', 'llm_transcript'),
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
                self.logger.info(f"📨 Processing: {envelope.ros_msg_type}")
                
                # Ensure session exists
                await self._ensure_session()
                
                # Delegate to receive coordinator (the middleware)
                await self.receive_coordinator.handle_message(envelope)
                
        except Exception as e:
            self.logger.error(f"Error processing bridge message: {e}")
            self.metrics['errors'] += 1
            
    async def _ensure_session(self):
        """Ensure we have an active session"""
        if self.session_manager.state == SessionState.IDLE and not self.session_creating:
            self.session_creating = True
            try:
                self.logger.info("🔗 Creating Gemini session...")
                success = await self.session_manager.connect_session()
                if success:
                    self.pause_detector.reset()
                    self.metrics['sessions_created'] += 1
                    self.logger.info("✅ Session created")
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