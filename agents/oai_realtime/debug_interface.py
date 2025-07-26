"""
Debug Interface for OpenAI Realtime Agent

Provides a way to inject audio data and messages directly into the agent
for testing without requiring ROS bridge connection.

Author: Karim Virani
Version: 1.0
Date: July 2025
"""

import asyncio
import base64
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
    """Debug interface for direct agent interaction"""
    
    def __init__(self, agent):
        self.agent = agent
        self.logger = logging.getLogger(__name__)
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
        """Inject audio data directly into agent"""
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
                raw_data={
                    "audio_data": audio_data,
                    "utterance_id": utterance_id,
                    "start_time": time.time(),
                    "confidence": confidence,
                    "chunk_sequence": 0,
                    "is_utterance_end": is_utterance_end
                }
            )
            
            # Inject into agent's processing pipeline
            success = await self._process_debug_message(envelope)
            
            if success:
                self.messages_injected += 1
                self.logger.info(f"✅ Injected audio data: {utterance_id} ({len(audio_data)} samples)")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error injecting audio data: {e}")
            return False
            
    async def inject_text_message(self, text: str) -> bool:
        """Inject text message directly into agent"""
        try:
            envelope = DebugMessageEnvelope(
                msg_type="topic",
                topic_name="/text_input",
                ros_msg_type="std_msgs/String",
                timestamp=time.time(),
                metadata={"debug_injected": True},
                raw_data={"data": text}
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
        """Process debug message through agent pipeline"""
        try:
            # Record message activity for pause detection
            self.agent.pause_detector.record_message(envelope.ros_msg_type)
            self.agent.metrics['messages_processed'] += 1
            
            # Ensure we have a session for incoming messages
            from .session_manager import SessionState
            if self.agent.session_manager.state == SessionState.IDLE:
                success = await self.agent.session_manager.connect_session()
                if success:
                    self.agent.pause_detector.reset()
                    self.logger.info("Session created for debug message")
                else:
                    self.logger.error("Failed to create session for debug message")
                    return False
            
            # Handle different message types
            if envelope.ros_msg_type == "by_your_command/AudioDataUtterance":
                # Serialize for OpenAI Realtime API
                api_msg = await self._serialize_audio_utterance(envelope)
                
                if api_msg and self.agent.session_manager.is_connected():
                    await self.agent.session_manager.websocket.send(json.dumps(api_msg))
                    self.agent.metrics['messages_sent_to_openai'] += 1
                    self.logger.debug("Sent debug AudioDataUtterance to OpenAI")
                    
                    # Store metadata for context injection
                    self.agent.serializer.current_utterance_metadata = envelope.metadata
                    
                    return True
                    
            elif envelope.ros_msg_type == "std_msgs/String":
                # Handle text input (could be converted to audio or sent as text)
                text_data = envelope.raw_data["data"]
                
                # For now, just log it - could be extended to convert to speech
                self.logger.info(f"Debug text input: {text_data}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error processing debug message: {e}")
            return False
            
    async def _serialize_audio_utterance(self, envelope: DebugMessageEnvelope) -> Optional[Dict]:
        """Serialize debug audio utterance for OpenAI API"""
        try:
            audio_data = envelope.raw_data["audio_data"]
            
            # Convert to PCM bytes
            if isinstance(audio_data, list):
                pcm_bytes = np.array(audio_data, dtype=np.int16).tobytes()
            else:
                pcm_bytes = audio_data  # Already bytes
                
            base64_audio = base64.b64encode(pcm_bytes).decode()
            
            return {
                "type": "input_audio_buffer.append",
                "audio": base64_audio
            }
            
        except Exception as e:
            self.logger.error(f"Error serializing debug audio: {e}")
            return None
            
    def get_stats(self) -> Dict[str, Any]:
        """Get debug interface statistics"""
        return {
            "running": self.running,
            "messages_injected": self.messages_injected,
            "responses_received": self.responses_received,
            "queue_size": self.message_queue.qsize()
        }


# Convenience functions for creating test audio data
def create_test_audio_sine_wave(frequency: int = 440, duration: float = 1.0, 
                               sample_rate: int = 16000) -> List[int]:
    """Create test sine wave audio data"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    sine_wave = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    audio_data = (sine_wave * 32767).astype(np.int16)
    return audio_data.tolist()


def create_test_audio_noise(duration: float = 1.0, sample_rate: int = 16000) -> List[int]:
    """Create test white noise audio data"""
    samples = int(sample_rate * duration)
    noise = np.random.uniform(-1, 1, samples)
    
    # Convert to 16-bit PCM
    audio_data = (noise * 32767).astype(np.int16)
    return audio_data.tolist()


def load_wav_file(file_path: str, target_sample_rate: int = 16000) -> Optional[List[int]]:
    """Load WAV file and convert to test audio data"""
    try:
        import scipy.io.wavfile as wav
        
        sample_rate, audio_data = wav.read(file_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Resample if needed (simple approach)
        if sample_rate != target_sample_rate:
            # Basic resampling - for production use scipy.signal.resample
            ratio = target_sample_rate / sample_rate
            new_length = int(len(audio_data) * ratio)
            audio_data = np.interp(
                np.linspace(0, len(audio_data) - 1, new_length),
                np.arange(len(audio_data)),
                audio_data
            )
            
        # Ensure 16-bit PCM
        if audio_data.dtype != np.int16:
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
                
        return audio_data.tolist()
        
    except ImportError:
        print("scipy not available - cannot load WAV files")
        return None
    except Exception as e:
        print(f"Error loading WAV file: {e}")
        return None


# Example usage functions
async def demo_debug_interface(agent):
    """Demonstrate debug interface usage"""
    debug = DebugInterface(agent)
    await debug.start()
    
    try:
        print("🔧 Testing debug interface...")
        
        # Test 1: Inject sine wave
        print("1. Injecting test sine wave...")
        sine_wave = create_test_audio_sine_wave(440, 2.0)  # 2 seconds of 440Hz
        success = await debug.inject_audio_data(sine_wave, "test_sine", is_utterance_end=True)
        print(f"   Sine wave injection: {'✅' if success else '❌'}")
        
        # Wait a moment
        await asyncio.sleep(1)
        
        # Test 2: Inject text
        print("2. Injecting test text...")
        success = await debug.inject_text_message("Hello, this is a debug test message!")
        print(f"   Text injection: {'✅' if success else '❌'}")
        
        # Test 3: Show stats
        print("3. Debug interface stats:")
        stats = debug.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
    finally:
        await debug.stop()


if __name__ == "__main__":
    # Standalone test
    print("🔧 Debug Interface Test Module")
    print("=" * 40)
    
    # Test audio generation
    print("Testing audio generation...")
    sine_data = create_test_audio_sine_wave(440, 0.1)  # 100ms
    noise_data = create_test_audio_noise(0.1)  # 100ms
    
    print(f"Sine wave: {len(sine_data)} samples")
    print(f"Noise: {len(noise_data)} samples")
    print("✅ Audio generation working")