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
    Per-turn receive coordinator following Gemini's proven pattern.
    
    Creates receive generator AFTER sending input for each turn.
    Ends generator on turn_complete, creates new one for next turn.
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
        
        # State tracking - per-turn approach
        self.current_turn_task: Optional[asyncio.Task] = None
        self.receiving = False
        
        # Buffer for accumulating text fragments (command agent)
        self.text_buffer = ""
        
        # Metrics
        self.metrics = {
            'audio_chunks_sent': 0,
            'text_messages_sent': 0,
            'responses_received': 0,
            'audio_responses': 0,
            'text_responses': 0,
            'turns_completed': 0,
            'sessions_started': 0
        }
        
    async def start_turn_receive(self, session):
        """
        Start receive generator for current turn AFTER input is sent.
        Only cancel previous task if it has already received responses.
        """
        # If previous task exists and hasn't finished, give it time to complete
        if self.current_turn_task and not self.current_turn_task.done():
            self.logger.info("⏳ Previous turn still active - giving it 2s to complete naturally")
            try:
                # Wait up to 2 seconds for natural completion
                await asyncio.wait_for(self.current_turn_task, timeout=2.0)
                self.logger.info("✅ Previous turn completed naturally")
            except asyncio.TimeoutError:
                self.logger.info("⏰ Previous turn timeout - cancelling to start new one")
                self.current_turn_task.cancel()
                try:
                    await self.current_turn_task
                except asyncio.CancelledError:
                    pass
            except Exception as e:
                self.logger.error(f"Error waiting for previous turn: {e}")
        
        self.receiving = True
        self.current_turn_task = asyncio.create_task(self._receive_turn_responses(session))
        # Track responses on the task
        self.current_turn_task._responses_received = 0
        self.logger.info(f"🚀 Started receive generator for turn (session: {id(session)})")
        
    async def _receive_turn_responses(self, session):
        """
        Receive responses for current turn, ending on turn_complete.
        This matches the pattern from test scripts.
        """
        session_id = id(session)
        response_count = 0
        
        try:
            self.logger.info(f"🎧 Creating turn receive generator (session: {session_id})")
            
            async for response in session.receive():
                response_count += 1
                self.metrics['responses_received'] += 1
                
                # Track responses on current task
                if hasattr(self.current_turn_task, '_responses_received'):
                    self.current_turn_task._responses_received = response_count
                
                # Log ALL attributes of the response for debugging
                self.logger.info(f"📥 DIAGNOSTIC: Turn response #{response_count}: {type(response)} (session: {session_id})")
                
                # Debug: Show all attributes
                if hasattr(response, '__dict__'):
                    attrs = {k: type(v).__name__ for k, v in response.__dict__.items() if not k.startswith('_')}
                    self.logger.info(f"  Response attributes: {attrs}")
                
                # Handle audio data
                if hasattr(response, 'data') and response.data:
                    self.logger.info(f"🔊 Audio response: {len(response.data)} bytes")
                    await self._handle_audio_response(response.data)
                    
                # Handle text
                elif hasattr(response, 'text') and response.text:
                    self.logger.info(f"📝 Text response: {response.text[:100]}...")
                    await self._handle_text_response(response.text)
                    
                # Handle tool calls (Gemini uses these for function execution)
                elif hasattr(response, 'tool_call') and response.tool_call:
                    self.logger.info(f"🔧 Tool call response: {response.tool_call}")
                    # Tool calls might contain transcripts or commands
                    
                # Handle user content (potential transcript source)
                elif hasattr(response, 'user_content') and response.user_content:
                    self.logger.info(f"👤 User content: {response.user_content}")
                    # This might be where user transcripts come from
                    
                # Check for server content (transcriptions and turn signals)
                if hasattr(response, 'server_content') and response.server_content:
                    server_content = response.server_content
                    
                    # Handle input transcription (user's speech)
                    if hasattr(server_content, 'input_transcription') and server_content.input_transcription:
                        transcription = server_content.input_transcription
                        # Extract text from transcription object
                        if hasattr(transcription, 'text'):
                            user_text = transcription.text
                        else:
                            user_text = str(transcription)
                        self.logger.info(f"👤 User transcript: {user_text}")
                        if user_text and user_text.strip() and user_text.strip() != '.':
                            await self._handle_user_transcript(user_text)
                    
                    # Handle output transcription (assistant's speech)
                    if hasattr(server_content, 'output_transcription') and server_content.output_transcription:
                        transcription = server_content.output_transcription
                        # Extract text from transcription object
                        if hasattr(transcription, 'text'):
                            assistant_text = transcription.text
                        else:
                            assistant_text = str(transcription)
                        self.logger.info(f"🤖 Assistant transcript: {assistant_text}")
                        # We already handle text responses, this is just for logging
                    
                    if hasattr(server_content, 'generation_complete') and server_content.generation_complete:
                        self.logger.info("✅ Generation complete signal")
                        
                    if hasattr(server_content, 'turn_complete') and server_content.turn_complete:
                        self.logger.info(f"✅ Turn complete - ending this generator (session: {session_id})")
                        self.metrics['turns_completed'] += 1
                        
                        # Process buffered text for command agents
                        await self._process_command_buffer()
                        
                        break  # END this generator, as per test scripts
                        
        except asyncio.CancelledError:
            self.logger.info(f"📡 Turn receive cancelled (session: {session_id})")
            raise
        except Exception as e:
            self.logger.error(f"❌ Error in turn receive (session: {session_id}): {e}")
        finally:
            self.receiving = False
            self.logger.info(f"📡 Turn receive ended - got {response_count} responses (session: {session_id})")
        
    async def handle_message(self, envelope):
        """
        Handle incoming message from bridge.
        
        Simplified - just routes messages to Gemini. Persistent receive loop
        handles all responses automatically.
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
        
        Simplified - just send audio to Gemini. Persistent receive loop
        handles responses automatically.
        """
        # Extract audio data
        if hasattr(envelope.raw_data, 'int16_data'):
            # Convert int16 array to bytes
            audio_array = np.array(envelope.raw_data.int16_data, dtype=np.int16)
            audio_bytes = audio_array.tobytes()
        else:
            self.logger.error("No audio data in envelope")
            return
            
        # Check if session exists
        if not self.session_manager.session:
            self.logger.warning("No active session - audio chunk dropped")
            return
            
        chunk_id = envelope.raw_data.chunk_sequence if hasattr(envelope.raw_data, 'chunk_sequence') else 0
        
        # Send audio chunk to Gemini first
        success = await self.session_manager.send_audio(audio_bytes)
        
        if success:
            self.metrics['audio_chunks_sent'] += 1
            self.logger.debug(f"🎤 Audio chunk #{chunk_id} sent")
            
            # Start receive generator AFTER sending first chunk (test pattern)
            if chunk_id == 0:
                self.logger.info(f"🚀 Starting receive generator after first audio chunk")
                await self.start_turn_receive(self.session_manager.session)
            
            # Log final chunk and add silence hack
            if hasattr(envelope.raw_data, 'is_utterance_end') and envelope.raw_data.is_utterance_end:
                self.logger.info(f"🎤 Final chunk #{chunk_id} - adding 1s silence for Gemini VAD")
                await self._send_silence_hack()
        else:
            self.logger.error("Failed to send audio to Gemini")
            
    async def _handle_text_message(self, envelope):
        """
        Handle text message from bridge.
        
        Simplified - just send text to Gemini. Persistent receive loop
        handles responses automatically.
        """
        text = envelope.raw_data.data
        
        # Send text to Gemini first
        success = await self.session_manager.send_text(text)
        
        if success:
            self.metrics['text_messages_sent'] += 1
            self.logger.info(f"💬 Sent text: {text[:50]}...")
            
            # Start receive generator AFTER sending text (test pattern)
            self.logger.info(f"🚀 Starting receive generator after text")
            await self.start_turn_receive(self.session_manager.session)
        else:
            self.logger.error("Failed to send text to Gemini")
            
    async def _send_silence_hack(self):
        """
        Send 1 second of silence to trigger Gemini's server-side VAD.
        This is a hack to work around Gemini Live's apparent need for 
        silence to detect utterance completion.
        """
        if not self.session_manager.session:
            return
            
        try:
            # Create 1 second of silence at 16kHz, 16-bit PCM
            sample_rate = 16000
            duration = 1.0
            num_samples = int(sample_rate * duration)
            silence_bytes = bytes([0, 0] * num_samples)  # PCM16 silence
            
            self.logger.info(f"🔇 Sending {len(silence_bytes)} bytes of silence to trigger Gemini VAD")
            success = await self.session_manager.send_audio(silence_bytes)
            
            if success:
                self.logger.info("✅ Silence hack sent successfully")
            else:
                self.logger.warning("❌ Failed to send silence hack")
                
        except Exception as e:
            self.logger.error(f"Error in silence hack: {e}")
            
    async def _handle_audio_response(self, audio_data: bytes):
        """Handle audio response from Gemini"""
        self.metrics['audio_responses'] += 1
        
        # Check if audio output is configured (not empty)
        # Using new naming: response_voice for agent audio output
        audio_topic = self.published_topics.get('response_voice', '')
        if not audio_topic:
            self.logger.debug("🔇 Audio output disabled - skipping audio response")
            return
        
        # Convert PCM bytes to int16 array for ROS
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Publish to bridge
        if self.bridge:
            await self.bridge.put_outbound_message(
                topic=audio_topic,
                msg_data={'int16_data': audio_array.tolist()},
                msg_type='audio_common_msgs/AudioData'
            )
            self.logger.debug(f"🔊 Published audio ({len(audio_array)} samples)")
            
    async def _handle_user_transcript(self, text: str):
        """Handle user transcript from Gemini STT"""
        # Determine agent type
        agent_id = self.session_manager.config.get('agent_id', '')
        
        # NEW: Publish user transcript to prompt_transcript for ALL agents
        prompt_transcript_topic = self.published_topics.get('prompt_transcript', 'prompt_transcript')
        if self.bridge and prompt_transcript_topic:
            await self.bridge.put_outbound_message(
                topic=prompt_transcript_topic,
                msg_data={'data': text},
                msg_type='std_msgs/String'
            )
            self.logger.info(f"📤 Published user transcript to {prompt_transcript_topic}: {text}")
        
        # Add to conversation context for ALL agents (helps with context management)
        self.session_manager.add_conversation_turn("user", text)
        
        # Log differently based on agent type
        if 'command' in agent_id.lower():
            self.logger.info(f"📝 Command agent received user input: {text}")
        else:
            self.logger.info(f"👤 Conversational agent received user input: {text}")
    
    async def _handle_text_response(self, text: str):
        """Handle text response from Gemini"""
        self.metrics['text_responses'] += 1
        
        # Always log the full text for debugging
        self.logger.info(f"📝 RAW TEXT fragment from Gemini: '{text}'")
        
        # Determine which topic to use based on agent type
        agent_id = self.session_manager.config.get('agent_id', '')
        
        # Command extractors need special handling for fragments
        if 'command' in agent_id.lower():
            # Accumulate text fragments
            self.text_buffer += text
            self.logger.info(f"📝 Buffer now: '{self.text_buffer}'")
            
            # Don't publish fragments - wait for complete response
            return
        else:
            # Conversational agent - use response_text topic from config
            # Using new naming: response_text for agent text output
            topic = self.published_topics.get('response_text', 'response_text')
            output_text = f"Assistant: {text}"
            
            # Check for command acknowledgment pattern
            if text.startswith("OK - ") or text.startswith("OK-"):
                self.logger.warning(f"🎯 COMMAND ACK DETECTED: {text}")
            
            # Check if response mentions visual content (debugging for image recognition)
            visual_keywords = ['see', 'image', 'photo', 'shows', 'appears', 'visible', 
                              'wooden', 'sword', 'coin', 'hammer', 'book', 'table',
                              'color', 'object', 'background', 'foreground']
            mentions_visual = any(keyword in text.lower() for keyword in visual_keywords)
            
            if mentions_visual:
                self.logger.warning(f"🎯 VISUAL RESPONSE DETECTED: {text[:200]}...")
            else:
                self.logger.info(f"🤖 Assistant: {text[:100]}...")
        
        # Publish to bridge
        if self.bridge and topic:
            await self.bridge.put_outbound_message(
                topic=topic,
                msg_data={'data': output_text},
                msg_type='std_msgs/String'
            )
            
        # Add to conversation context (only for conversational agents)
        if 'command' not in agent_id.lower():
            self.session_manager.add_conversation_turn("assistant", text)
        
    async def _process_command_buffer(self):
        """Process accumulated text buffer for command agents"""
        agent_id = self.session_manager.config.get('agent_id', '')
        
        # Only process for command agents
        if 'command' not in agent_id.lower() or not self.text_buffer:
            return
            
        # Full response is now in buffer
        full_text = self.text_buffer.strip()
        self.text_buffer = ""  # Clear buffer for next turn
        
        if not full_text:
            return
            
        # Clean up markdown JSON wrapper if present
        if full_text.startswith('```json') and full_text.endswith('```'):
            # Extract JSON from markdown code block
            lines = full_text.split('\n')
            # Remove first line (```json) and last line (```)
            json_lines = lines[1:-1] if len(lines) > 2 else lines
            full_text = '\n'.join(json_lines).strip()
            self.logger.info(f"📝 Cleaned markdown wrapper from JSON response")
            
        self.logger.info(f"📝 Complete response: '{full_text}'")
        
        # Publish whatever the command agent generated - let command_processor decide
        # Using new naming: response_cmd for command output
        topic = self.published_topics.get('response_cmd', 'response_cmd')
        
        if self.bridge:
            await self.bridge.put_outbound_message(
                topic=topic,
                msg_data={'data': full_text},
                msg_type='std_msgs/String'
            )
            self.logger.info(f"🎯 Published to {topic}: {full_text[:100]}...")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.current_turn_task and not self.current_turn_task.done():
            self.current_turn_task.cancel()
            try:
                await self.current_turn_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Receive coordinator cleaned up")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get coordinator metrics"""
        return self.metrics.copy()