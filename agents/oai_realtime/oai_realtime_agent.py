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
from typing import Optional, Dict, Any
import numpy as np

#import rclpy
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData

from ros_ai_bridge import ROSAIBridge, MessageEnvelope, AgentInterface
from .serializers import OpenAIRealtimeSerializer
from .pause_detector import PauseDetector
from .session_manager import SessionManager, SessionState
from .context import ConversationContext


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
        self.bridge_interface = None  # Will be set in initialize()
        
        # Core components
        self.serializer = OpenAIRealtimeSerializer()
        self.pause_detector = PauseDetector(
            pause_timeout=self.config.get('session_pause_timeout', 10.0)
        )
        self.session_manager = SessionManager(self.config)
        
        # State
        self.running = False
        self.prepared_context: Optional[ConversationContext] = None
        
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
        
        # Validate configuration
        if not self.config.get('openai_api_key'):
            raise ValueError("OpenAI API key required in configuration")
            
        # TODO: Connect to bridge through IPC mechanism
        # For now, this is a placeholder - actual bridge communication needs implementation
        self.logger.warning("Bridge connection not yet implemented - running in standalone mode")
            
        self.logger.info("Agent initialized successfully")
        
    async def run(self):
        """Main agent loop - consumes from bridge, manages sessions"""
        self.running = True
        self.start_time = time.time()
        self.logger.info("Starting OpenAI Realtime Agent main loop...")
        
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
            self.prepared_context is not None):
            
            # Create new session with prepared context
            success = await self.session_manager.connect_session(self.prepared_context)
            if success:
                self.prepared_context = None
                self.pause_detector.reset()
                self.logger.info("Session created with injected context")
                
    async def _process_bridge_messages(self):
        """Process messages from bridge interface"""
        try:
            # Skip if bridge interface not connected yet
            if not self.bridge_interface:
                return
                
            # Try to get message with short timeout
            envelope = await self.bridge_interface.get_inbound_message(timeout=0.1)
            
            if envelope:
                self.pause_detector.record_message(envelope.ros_msg_type)
                self.metrics['messages_processed'] += 1
                
                # Ensure we have a session for incoming messages
                if self.session_manager.state == SessionState.IDLE:
                    success = await self.session_manager.connect_session()
                    if success:
                        self.pause_detector.reset()
                        self.logger.info("Session created for incoming message")
                
                # Serialize message for OpenAI API
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
        """Cycle session due to pause detection - aggressive cost optimization"""
        self.logger.info("🔄 Cycling session on pause (cost optimization)")
        
        context = await self.session_manager.close_session()
        self.prepared_context = context
        self.metrics['sessions_cycled_on_pause'] += 1
        
        # Reset pause detector for next conversation
        self.pause_detector.reset()
        
        self.logger.info("✅ Session cycled - ready for next speech")
        
    async def _cycle_session_on_limits(self):
        """Cycle session due to time/cost limits - seamless rotation"""
        self.logger.info("🔄 Cycling session on limits (seamless rotation)")
        
        context = await self.session_manager.close_session() 
        
        # Immediately reconnect with context (no pause to wait)
        if context:
            success = await self.session_manager.connect_session(context)
            if success:
                self.metrics['sessions_cycled_on_limits'] += 1
                self.pause_detector.reset()
                self.logger.info("✅ Session rotated seamlessly")
            else:
                self.logger.error("❌ Failed to rotate session - preparing for next message")
                self.prepared_context = context
            
    async def _process_openai_responses(self):
        """Process responses from OpenAI WebSocket"""
        if not self.session_manager.is_connected():
            return
            
        try:
            # Non-blocking check for messages
            websocket = self.session_manager.websocket
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                data = json.loads(message)
                await self._handle_openai_event(data)
            except asyncio.TimeoutError:
                # No message - normal
                pass
                
        except Exception as e:
            self.logger.error(f"Error processing OpenAI response: {e}")
            
    async def _handle_openai_event(self, data: Dict):
        """Handle individual OpenAI Realtime API events"""
        event_type = data.get("type", "")
        
        try:
            if event_type == "response.audio_transcript.done":
                await self._handle_transcript_done(data)
                
            elif event_type == "response.audio.delta":
                await self._handle_audio_delta(data)
                
            elif event_type == "input_audio_buffer.speech_started":
                self.pause_detector.record_message("speech_started")
                self.logger.debug("👤 User started speaking")
                
            elif event_type == "conversation.item.input_audio_transcription.completed":
                await self._handle_user_transcript(data)
                
            elif event_type == "response.done":
                self.pause_detector.mark_llm_response_complete()
                self.logger.debug("🤖 Assistant response complete")
                
            elif event_type == "error":
                await self._handle_openai_error(data)
                
            else:
                self.logger.debug(f"Unhandled OpenAI event: {event_type}")
                
        except Exception as e:
            self.logger.error(f"Error handling OpenAI event {event_type}: {e}")
            
    async def _handle_transcript_done(self, data: Dict):
        """Handle completed assistant transcript"""
        transcript = data.get("transcript", "").strip()
        if transcript:
            # Add to conversation context
            self.session_manager.add_conversation_turn("assistant", transcript)
            
            # Send transcript to ROS
            transcript_msg = String(data=transcript)
            success = self.bridge_interface.put_outbound_message(
                "/llm_transcript", 
                transcript_msg, 
                "std_msgs/String"
            )
            
            if success:
                self.metrics['messages_sent_to_ros'] += 1
                self.logger.debug(f"🤖 Assistant: {transcript[:50]}...")
                
        self.pause_detector.mark_llm_response_complete()
        
    async def _handle_audio_delta(self, data: Dict):
        """Handle audio response delta"""
        audio_b64 = data.get("delta", "")
        if audio_b64:
            try:
                # Decode and convert to ROS audio message
                audio_data = base64.b64decode(audio_b64)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_msg = AudioData(int16_data=audio_array.tolist())
                
                success = self.bridge_interface.put_outbound_message(
                    "/audio_out", 
                    audio_msg, 
                    "audio_common_msgs/AudioData"
                )
                
                if success:
                    self.metrics['messages_sent_to_ros'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing audio delta: {e}")
                
        self.pause_detector.mark_llm_response_active()
        
    async def _handle_user_transcript(self, data: Dict):
        """Handle completed user transcript"""
        transcript = data.get("transcript", "").strip()
        if transcript:
            self.session_manager.add_conversation_turn("user", transcript)
            self.logger.debug(f"👤 User: {transcript[:50]}...")
            
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
            
    async def _cleanup(self):
        """Clean up resources"""
        if self.start_time:
            self.metrics['total_runtime'] = time.time() - self.start_time
            
        self.logger.info("🧹 Cleaning up OpenAI Realtime Agent...")
        
        if self.session_manager.state == SessionState.ACTIVE:
            await self.session_manager.close_session()
            
        await self.bridge_interface.close()
        
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
            'session_manager': self.session_manager.get_metrics()
        })
        
        # Calculate runtime
        if self.start_time:
            combined_metrics['current_runtime'] = time.time() - self.start_time
            
        return combined_metrics
        
    def get_status_summary(self) -> str:
        """Get human-readable status summary"""
        session_status = f"Session: {self.session_manager.state.value}"
        pause_status = self.pause_detector.get_status_summary()
        message_stats = f"Processed: {self.metrics['messages_processed']}"
        
        return f"{session_status} | {pause_status} | {message_stats}"
        
    def update_configuration(self, new_config: Dict):
        """Update agent configuration at runtime"""
        old_config = self.config.copy()
        self.config.update(new_config)
        
        # Update components
        if 'session_pause_timeout' in new_config:
            self.pause_detector.update_pause_timeout(new_config['session_pause_timeout'])
            
        self.session_manager.update_configuration(new_config)
        
        self.logger.info("⚙️ Configuration updated")


# Convenience function for standalone usage
async def create_and_run_agent(bridge: ROSAIBridge, config: Dict) -> OpenAIRealtimeAgent:
    """Create and run agent with proper error handling"""
    agent = OpenAIRealtimeAgent(bridge, config)
    
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