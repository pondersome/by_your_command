"""
Receive Coordinator - Minimal Middleware for Gemini

Manages the lifecycle of Gemini's receive generators, coordinating
between the streaming bridge interface and Gemini's turn-based pattern.

Key responsibility: Create receive generator AFTER sending input, not before.

Author: Karim Virani
Version: 1.0
Date: August 2025
"""

import asyncio
import logging
from typing import Optional, Dict, Any
import numpy as np
from google.genai import types


class ReceiveCoordinator:
    """
    Minimal middleware to coordinate Gemini's receive generator pattern.
    
    The core issue: Gemini's receive() generator must be created AFTER
    sending input, not before. This coordinator manages that lifecycle.
    """
    
    def __init__(self, bridge_interface, session_manager, published_topics: Dict[str, str]):
        """
        Initialize the coordinator.
        
        Args:
            bridge_interface: WebSocket bridge for communication
            session_manager: Gemini session manager
            published_topics: Topic names for publishing responses
        """
        self.bridge = bridge_interface
        self.session_manager = session_manager
        self.published_topics = published_topics
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self.receiving = False
        self.receive_task: Optional[asyncio.Task] = None
        self.first_chunk_sent = False  # Track if we've sent first audio chunk
        
        # Metrics
        self.metrics = {
            'audio_chunks_sent': 0,
            'text_messages_sent': 0,
            'responses_received': 0,
            'audio_responses': 0,
            'text_responses': 0,
            'turns_completed': 0
        }
        
    async def handle_message(self, envelope):
        """
        Handle incoming message from bridge.
        
        This is the main entry point - routes messages and manages
        the receive generator lifecycle.
        """
        msg_type = envelope.ros_msg_type
        
        try:
            if msg_type == "by_your_command/AudioDataUtterance":
                await self._handle_audio_chunk(envelope)
            elif msg_type == "std_msgs/String":
                await self._handle_text_message(envelope)
            elif msg_type == "conversation_id":
                # Conversation ID changes don't need Gemini interaction
                self.logger.info(f"Conversation ID: {envelope.raw_data.data}")
                
        except Exception as e:
            self.logger.error(f"Error handling {msg_type}: {e}")
            
    async def _handle_audio_chunk(self, envelope):
        """
        Handle audio chunk from bridge.
        
        Key pattern: Stream audio directly to Gemini, create receiver
        after FIRST chunk.
        """
        # Extract audio data
        if hasattr(envelope.raw_data, 'int16_data'):
            # Convert int16 array to bytes
            audio_array = np.array(envelope.raw_data.int16_data, dtype=np.int16)
            audio_bytes = audio_array.tobytes()
        else:
            self.logger.error("No audio data in envelope")
            return
            
        # Check for interruption (new audio while receiving)
        if self.receiving and envelope.raw_data.chunk_sequence == 0:
            self.logger.info("🛑 User interrupted - cancelling current response")
            await self._handle_interruption()
            # Continue to process the new audio
            
        # Send audio chunk to Gemini (streaming, no buffering!)
        success = await self.session_manager.send_audio(audio_bytes)
        
        if success:
            self.metrics['audio_chunks_sent'] += 1
            chunk_id = envelope.raw_data.chunk_sequence if hasattr(envelope.raw_data, 'chunk_sequence') else 0
            
            # Critical: Start receiver after FIRST chunk (not before!)
            if not self.first_chunk_sent:
                self.first_chunk_sent = True
                self.logger.info(f"🎤 First audio chunk #{chunk_id} sent, starting receiver")
                await self._start_receive_cycle()
            else:
                self.logger.debug(f"🎤 Audio chunk #{chunk_id} sent")
                
            # Check if this is the last chunk
            if hasattr(envelope.raw_data, 'is_end') and envelope.raw_data.is_end:
                self.logger.info(f"🎤 Final chunk #{chunk_id} - Gemini will auto-detect end and respond")
                self.first_chunk_sent = False  # Reset for next utterance
        else:
            self.logger.error("Failed to send audio to Gemini")
            
    async def _handle_text_message(self, envelope):
        """
        Handle text message from bridge.
        
        Text is atomic - send immediately and create receiver.
        """
        text = envelope.raw_data.data
        
        # Send text to Gemini
        success = await self.session_manager.send_text(text)
        
        if success:
            self.metrics['text_messages_sent'] += 1
            self.logger.info(f"💬 Sent text: {text[:50]}...")
            
            # Cancel any existing receiver (text interrupts audio)
            if self.receiving and self.receive_task:
                self.logger.info("Cancelling audio receiver for text input")
                self.receive_task.cancel()
                self.receiving = False
                
            # Start new receiver for text response
            await self._start_receive_cycle()
        else:
            self.logger.error("Failed to send text to Gemini")
            
    async def _start_receive_cycle(self):
        """
        Start a new receive cycle for the current input.
        
        Critical: This creates a NEW receive generator for this turn.
        One generator per conversation turn, not persistent.
        """
        if self.receiving:
            self.logger.debug("Already receiving, skipping start")
            return
            
        self.receiving = True
        self.receive_task = asyncio.create_task(self._receive_responses())
        self.logger.info("📡 Started receive cycle")
        
    async def _receive_responses(self):
        """
        Process responses for the current conversation turn.
        
        Receives until turn_complete signal, then stops.
        """
        try:
            session = self.session_manager.session
            if not session:
                self.logger.error("No active session for receiving")
                return
                
            self.logger.info("🎧 Creating receive generator for this turn")
            
            # Process responses from this turn's generator
            async for response in session.receive():
                self.metrics['responses_received'] += 1
                
                # Handle audio data
                if hasattr(response, 'data') and response.data:
                    await self._handle_audio_response(response.data)
                    
                # Handle text
                elif hasattr(response, 'text') and response.text:
                    await self._handle_text_response(response.text)
                    
                # Check for completion signals
                if hasattr(response, 'server_content') and response.server_content:
                    server_content = response.server_content
                    
                    if hasattr(server_content, 'generation_complete') and server_content.generation_complete:
                        self.logger.debug("Generation complete signal")
                        
                    if hasattr(server_content, 'turn_complete') and server_content.turn_complete:
                        self.logger.info("✅ Turn complete - ending receive cycle")
                        self.metrics['turns_completed'] += 1
                        break
                        
        except asyncio.CancelledError:
            self.logger.info("Receive cycle cancelled")
        except Exception as e:
            self.logger.error(f"Error in receive cycle: {e}")
        finally:
            self.receiving = False
            self.logger.info("📡 Receive cycle ended")
            
    async def _handle_audio_response(self, audio_data: bytes):
        """Handle audio response from Gemini"""
        self.metrics['audio_responses'] += 1
        
        # Convert PCM bytes to int16 array for ROS
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Publish to bridge
        if self.bridge:
            await self.bridge.put_outbound_message(
                topic=self.published_topics['audio_out'],
                msg_data={'int16_data': audio_array.tolist()},
                msg_type='audio_common_msgs/AudioData'
            )
            self.logger.debug(f"🔊 Published audio ({len(audio_array)} samples)")
            
    async def _handle_text_response(self, text: str):
        """Handle text response from Gemini"""
        self.metrics['text_responses'] += 1
        
        # Publish transcript to bridge
        if self.bridge:
            await self.bridge.put_outbound_message(
                topic=self.published_topics['transcript'],
                msg_data={'data': f"Assistant: {text}"},
                msg_type='std_msgs/String'
            )
            self.logger.info(f"🤖 Assistant: {text[:100]}...")
            
        # Add to conversation context
        self.session_manager.add_conversation_turn("assistant", text)
        
    async def _handle_interruption(self):
        """Handle user interruption during response"""
        # Cancel receive task
        if self.receive_task and not self.receive_task.done():
            self.receive_task.cancel()
            
        # Send interrupt to Gemini
        await self.session_manager.interrupt_response()
        
        # Send interruption signal to audio player
        if self.bridge:
            await self.bridge.put_outbound_message(
                topic=self.published_topics.get('interruption_signal', 'interruption_signal'),
                msg_data={'data': True},
                msg_type='std_msgs/Bool'
            )
            
        # Reset state
        self.receiving = False
        self.first_chunk_sent = False
        
    async def cleanup(self):
        """Clean up resources"""
        if self.receive_task and not self.receive_task.done():
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass
                
        self.logger.info("Receive coordinator cleaned up")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get coordinator metrics"""
        return self.metrics.copy()