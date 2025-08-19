"""
Debug Interface for LLM Agents

Provider-agnostic debug interface for injecting test data directly into agents
without requiring ROS bridge connection.

Author: Karim Virani
Version: 1.0
Date: August 2025
"""

import asyncio
import json
import logging
import numpy as np
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class DebugMessageEnvelope:
    """Debug message envelope that mimics WebSocket bridge messages"""
    msg_type: str           # 'topic', 'debug'
    topic_name: str         # Topic name
    ros_msg_type: str      # Message type
    timestamp: float       # Unix timestamp
    metadata: Dict[str, Any]
    raw_data: Any          # Message data


class DebugInterface:
    """Provider-agnostic debug interface for direct agent interaction"""
    
    def __init__(self, agent):
        """
        Initialize debug interface
        
        Args:
            agent: The LLM agent instance (must have serializer and session_manager)
        """
        self.agent = agent
        self.logger = logging.getLogger(self.__class__.__name__)
        self.message_queue = asyncio.Queue()
        self.running = False
        
        # Stats
        self.messages_injected = 0
        self.responses_received = 0
        
    async def start(self):
        """Start debug interface"""
        self.running = True
        self.logger.info("🔧 Debug interface started")
        
    async def stop(self):
        """Stop debug interface"""
        self.running = False
        self.logger.info("🔧 Debug interface stopped")
        
    async def inject_audio_data(self, audio_data: List[int], 
                               utterance_id: str = None,
                               confidence: float = 0.95,
                               is_utterance_end: bool = False) -> bool:
        """
        Inject audio data directly into agent
        
        Args:
            audio_data: List of int16 audio samples
            utterance_id: Optional utterance ID
            confidence: Confidence score
            is_utterance_end: Whether this marks end of utterance
            
        Returns:
            bool: True if injection successful
        """
        try:
            if utterance_id is None:
                utterance_id = f"debug_utt_{int(time.time() * 1000)}"
                
            # Create debug message envelope
            envelope = DebugMessageEnvelope(
                msg_type="topic",
                topic_name="/voice_chunks",
                ros_msg_type="by_your_command/AudioDataUtterance",
                timestamp=time.time(),
                metadata={
                    "utterance_id": utterance_id,
                    "confidence": confidence,
                    "debug_injected": True
                },
                raw_data=type('obj', (object,), {
                    'audio_data': audio_data,
                    'int16_data': audio_data,  # Support both field names
                    'utterance_id': utterance_id,
                    'start_time': time.time(),
                    'confidence': confidence,
                    'is_utterance_end': is_utterance_end,
                    'chunk_sequence': 0
                })()
            )
            
            success = await self._process_debug_message(envelope)
            
            if success:
                self.messages_injected += 1
                self.logger.info(f"✅ Injected audio: utterance={utterance_id}, "
                               f"samples={len(audio_data)}, end={is_utterance_end}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error injecting audio: {e}")
            return False
            
    async def inject_text_message(self, text: str) -> bool:
        """
        Inject text message directly into agent
        
        Args:
            text: Text message to inject
            
        Returns:
            bool: True if injection successful
        """
        try:
            envelope = DebugMessageEnvelope(
                msg_type="topic",
                topic_name="/text_input",
                ros_msg_type="std_msgs/String",
                timestamp=time.time(),
                metadata={"debug_injected": True},
                raw_data=type('obj', (object,), {'data': text})()
            )
            
            success = await self._process_debug_message(envelope)
            
            if success:
                self.messages_injected += 1
                self.logger.info(f"✅ Injected text message: '{text[:50]}...'")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error injecting text message: {e}")
            return False
            
    async def _process_debug_message(self, envelope: DebugMessageEnvelope) -> bool:
        """
        Process debug message through agent pipeline
        
        Args:
            envelope: Debug message envelope
            
        Returns:
            bool: True if processing successful
        """
        try:
            # Check if agent has required components
            if not hasattr(self.agent, 'serializer'):
                self.logger.error("Agent missing serializer - cannot process debug messages")
                return False
                
            if not hasattr(self.agent, 'session_manager'):
                self.logger.error("Agent missing session_manager - cannot process debug messages")
                return False
            
            # Ensure session is active
            if not self.agent.session_manager.is_connected():
                self.logger.warning("No active session - creating one for debug")
                success = await self.agent.session_manager.connect_session()
                if not success:
                    self.logger.error("Failed to create session for debug message")
                    return False
            
            # Use agent's serializer (provider-specific)
            api_msg = await self.agent.serializer.safe_serialize(envelope)
            
            if not api_msg:
                self.logger.warning(f"Could not serialize {envelope.ros_msg_type}")
                return False
            
            # Send through agent's session manager
            if self.agent.session_manager.websocket:
                await self.agent.session_manager.websocket.send(json.dumps(api_msg))
                
                # Update agent metrics if available
                if hasattr(self.agent, 'metrics'):
                    metric_key = 'messages_sent_to_' + self.agent.__class__.__name__.lower()
                    if metric_key in self.agent.metrics:
                        self.agent.metrics[metric_key] += 1
                    else:
                        self.agent.metrics['messages_sent'] = self.agent.metrics.get('messages_sent', 0) + 1
                
                self.logger.debug(f"Sent debug {envelope.ros_msg_type} to LLM provider")
                return True
            else:
                self.logger.error("No WebSocket connection available")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing debug message: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get debug interface statistics"""
        return {
            "running": self.running,
            "messages_injected": self.messages_injected,
            "responses_received": self.responses_received,
            "queue_size": self.message_queue.qsize()
        }


# ============================================================
# Convenience functions for creating test audio data
# ============================================================

def create_test_audio_sine_wave(frequency: int = 440, duration: float = 1.0, 
                               sample_rate: int = 16000) -> List[int]:
    """
    Create test sine wave audio data
    
    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        List[int]: PCM16 audio samples
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    sine_wave = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    pcm_data = (sine_wave * 32767).astype(np.int16)
    return pcm_data.tolist()


def create_test_audio_white_noise(duration: float = 1.0, 
                                 sample_rate: int = 16000,
                                 amplitude: float = 0.1) -> List[int]:
    """
    Create test white noise audio data
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Amplitude factor (0.0 to 1.0)
        
    Returns:
        List[int]: PCM16 audio samples
    """
    num_samples = int(sample_rate * duration)
    noise = np.random.randn(num_samples) * amplitude
    
    # Convert to 16-bit PCM
    pcm_data = (noise * 32767).astype(np.int16)
    return pcm_data.tolist()


def create_test_audio_silence(duration: float = 1.0,
                             sample_rate: int = 16000) -> List[int]:
    """
    Create test silence audio data
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        List[int]: PCM16 audio samples (all zeros)
    """
    num_samples = int(sample_rate * duration)
    return [0] * num_samples