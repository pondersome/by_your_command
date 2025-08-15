#!/usr/bin/env python3
"""
Gemini Live Agent using Pipecat

Implements a multimodal agent using Pipecat's GeminiMultimodalLiveLLMService.
Follows the multi-agent architecture pattern established with OpenAI agents.

Key Features:
- Simple linear pipeline per agent instance
- WebSocket connection to ROS AI Bridge
- Support for conversation, command extraction, and scene description modes
- Frame-based data flow with Pipecat abstractions

Author: Karim Virani
Version: 3.0
Date: August 2025
"""

import asyncio
import logging
import os
import base64
import json
import yaml
import argparse
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime
import cv2

# Pipecat imports
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    StartFrame,
    EndFrame,
    AudioRawFrame,
    TextFrame,
    ImageRawFrame,
    TranscriptionFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)
# Try to import Pipecat's VAD
try:
    from pipecat.audio.vad.vad_analyzer import VADAudioFrameProcessor
except ImportError:
    VADAudioFrameProcessor = None
# Try to import Gemini service, fall back to direct implementation if needed
try:
    from pipecat.services.gemini_multimodal_live.gemini import (
        GeminiMultimodalLiveLLMService,
        GeminiMultimodalModalities,
        InputParams,
    )
except ImportError as e:
    # If import fails due to Google Cloud STT, we'll need to work around it
    import sys
    import types
    
    # Create a dummy google.cloud module to satisfy the import
    if 'google.cloud' not in sys.modules:
        google_cloud = types.ModuleType('google.cloud')
        sys.modules['google.cloud'] = google_cloud
        # Add dummy speech_v2 to avoid import error
        google_cloud.speech_v2 = types.ModuleType('google.cloud.speech_v2')
        sys.modules['google.cloud.speech_v2'] = google_cloud.speech_v2
    
    # Now try again
    from pipecat.services.gemini_multimodal_live.gemini import (
        GeminiMultimodalLiveLLMService,
        GeminiMultimodalModalities,
        InputParams,
    )

# ROS Bridge interface (from OpenAI agent)
from agents.oai_realtime.websocket_bridge import WebSocketBridgeInterface

# Context management (will port from OpenAI agent)
from agents.oai_realtime.context import ConversationContext
from agents.oai_realtime.conversation_monitor import ConversationMonitor

# Proven prompt loading system
from agents.oai_realtime.prompt_loader import PromptLoader


class ROSInputProcessor(FrameProcessor):
    """Convert ROS messages to Pipecat frames"""
    
    def __init__(self, bridge_interface: WebSocketBridgeInterface):
        super().__init__()
        self.bridge = bridge_interface
        self.logger = logging.getLogger(__name__)
        self._running = True
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Pass through frames - actual conversion happens in message loop"""
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        
    async def start_message_loop(self, task: PipelineTask):
        """Read messages from ROS bridge and convert to frames"""
        self.logger.info("Starting ROS message loop")
        frame_count = 0
        
        while self._running:
            try:
                # Get message from bridge
                envelope = await self.bridge.get_inbound_message(timeout=0.1)
                
                if envelope:
                    self.logger.debug(f"Processing envelope: {envelope.ros_msg_type}")
                    frame = self._convert_to_frame(envelope)
                    if frame:
                        frame_count += 1
                        frame_type = type(frame).__name__
                        
                        # Log ALL frames for debugging
                        if isinstance(frame, AudioRawFrame):
                            source = getattr(frame, 'source', 'unknown')
                            self.logger.info(f"🎵 Queuing AudioRawFrame #{frame_count} to pipeline (source: {source}, {len(frame.audio)} bytes)")
                        elif isinstance(frame, ImageRawFrame):
                            self.logger.info(f"🖼️ Queuing ImageRawFrame #{frame_count} to pipeline ({frame.size})")
                        else:
                            self.logger.info(f"📝 Queuing {frame_type} #{frame_count} to pipeline")
                            
                        await task.queue_frame(frame)
                        self.logger.debug(f"✅ Successfully queued {frame_type} #{frame_count}")
                    else:
                        self.logger.debug(f"❌ No frame created from envelope: {envelope.ros_msg_type}")
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Message loop error: {e}", exc_info=True)
                
    def _convert_to_frame(self, envelope) -> Optional[Frame]:
        """Convert ROS message envelope to Pipecat frame"""
        try:
            msg_type = envelope.ros_msg_type
            self.logger.debug(f"Converting message type: {msg_type}")
            
            if msg_type == "by_your_command/AudioDataUtterance":
                # Convert audio data to AudioRawFrame
                audio_data = envelope.raw_data.int16_data
                if audio_data:
                    # Convert int16 array to bytes
                    audio_bytes = np.array(audio_data, dtype=np.int16).tobytes()
                    self.logger.debug(f"Converting audio utterance: {len(audio_data)} samples -> {len(audio_bytes)} bytes")
                    # Create AudioRawFrame with ID for Pipecat compatibility
                    frame = AudioRawFrame(
                        audio=audio_bytes,
                        sample_rate=16000,
                        num_channels=1
                    )
                    # Add ID if not present (Pipecat version compatibility)
                    if not hasattr(frame, 'id'):
                        frame.id = hash(audio_bytes[:100]) % 1000000  # Simple hash-based ID
                    # Mark as user input so we don't echo it back
                    frame.source = "user_input"
                    return frame
                    
            elif msg_type == "sensor_msgs/Image":
                # Convert ROS Image to ImageRawFrame
                try:
                    # Extract image data from ROS message
                    ros_image = envelope.raw_data
                    height = ros_image.height
                    width = ros_image.width
                    encoding = ros_image.encoding
                    
                    # Convert ROS image data to numpy array
                    if encoding == "rgb8":
                        # RGB8 format
                        image_array = np.frombuffer(ros_image.data, dtype=np.uint8)
                        image_array = image_array.reshape((height, width, 3))
                    elif encoding == "bgr8":
                        # BGR8 format - convert to RGB
                        image_array = np.frombuffer(ros_image.data, dtype=np.uint8)
                        image_array = image_array.reshape((height, width, 3))
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                    else:
                        self.logger.warning(f"Unsupported image encoding: {encoding}")
                        return None
                    
                    # Create ImageRawFrame for Pipecat
                    self.logger.debug(f"Converting image frame: {width}x{height} {encoding}")
                    return ImageRawFrame(
                        image=image_array,
                        size=(width, height),
                        format="RGB"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Image conversion error: {e}")
                    return None
                
            elif msg_type == "std_msgs/String":
                # Convert string to TextFrame
                text = envelope.raw_data.data
                if text:
                    return TextFrame(text=text)
                    
        except Exception as e:
            self.logger.error(f"Frame conversion error: {e}")
            
        return None
        
    def stop(self):
        """Stop the message loop"""
        self._running = False


class ROSOutputProcessor(FrameProcessor):
    """Convert Pipecat frames to ROS messages"""
    
    def __init__(self, bridge_interface: WebSocketBridgeInterface, agent_config: Dict):
        super().__init__()
        self.bridge = bridge_interface
        self.config = agent_config
        self.logger = logging.getLogger(__name__)
        
        # Output topic configuration
        self.audio_out_topic = agent_config.get('audio_out_topic', 'audio_out')
        self.transcript_topic = agent_config.get('transcript_topic', 'llm_transcript')
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Convert and send frames to ROS"""
        await super().process_frame(frame, direction)
        
        try:
            if isinstance(frame, AudioRawFrame) and self.audio_out_topic:
                # Only send Gemini's response audio to speakers, not user input
                frame_source = getattr(frame, 'source', 'gemini_response')
                
                if frame_source == 'user_input':
                    # Don't echo user input back to speakers
                    self.logger.debug(f"Skipping user input AudioRawFrame (not echoing)")
                else:
                    # Send Gemini's audio response to ROS speakers
                    audio_array = np.frombuffer(frame.audio, dtype=np.int16)
                    self.logger.info(f"Sending Gemini AudioRawFrame to ROS: {len(audio_array)} samples at {frame.sample_rate}Hz")
                    
                    # Convert to the correct format for audio_common_msgs/AudioData
                    # The ROS message expects 'int16_data' field with int16 values
                    message = {
                        'int16_data': audio_array.tolist(),
                        'sample_rate': frame.sample_rate
                    }
                    await self.bridge.put_outbound_message(
                        self.audio_out_topic,
                        message,
                        'audio_common_msgs/AudioData'
                    )
                
            elif isinstance(frame, TranscriptionFrame):
                # Send transcription to ROS
                self.logger.info(f"Sending TranscriptionFrame to ROS: '{frame.text}'")
                message = {'data': frame.text}
                await self.bridge.put_outbound_message(
                    self.transcript_topic,
                    message,
                    'std_msgs/String'
                )
                
            elif isinstance(frame, TextFrame):
                # Send text responses to ROS
                if hasattr(frame, 'role') and frame.role == 'assistant':
                    message = {'data': frame.text}
                    await self.bridge.put_outbound_message(
                        self.transcript_topic,
                        message,
                        'std_msgs/String'
                    )
                    
        except Exception as e:
            self.logger.error(f"Output processing error: {e}")
            
        # Always pass frame through
        await self.push_frame(frame, direction)


class VideoFrameThrottler(FrameProcessor):
    """Throttle video frames to control costs"""
    
    def __init__(self, fps: float = 1.0, dynamic: bool = False):
        super().__init__()
        self.min_interval = 1.0 / fps
        self.last_frame_time = 0
        self.dynamic = dynamic
        self.logger = logging.getLogger(__name__)
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Throttle image frames"""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, ImageRawFrame):
            import time
            now = time.time()
            
            if now - self.last_frame_time >= self.min_interval:
                await self.push_frame(frame, direction)
                self.last_frame_time = now
                self.logger.debug(f"Passed image frame (throttled to {1/self.min_interval} fps)")
            else:
                self.logger.debug("Dropped image frame (throttling)")
        else:
            # Pass non-image frames through immediately
            await self.push_frame(frame, direction)


class GeminiLiveAgent:
    """Gemini Live Agent using Pipecat pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.bridge_interface: Optional[WebSocketBridgeInterface] = None
        self.pipeline: Optional[Pipeline] = None
        self.task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None
        
        # Processors
        self.ros_input: Optional[ROSInputProcessor] = None
        self.ros_output: Optional[ROSOutputProcessor] = None
        
        # Context management (Phase 4)
        self.context_manager: Optional[ConversationContext] = None
        
        # Proven prompt loading system
        self.prompt_loader: Optional[PromptLoader] = None
        
        # Metrics
        self.metrics = {
            'messages_processed': 0,
            'frames_sent': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
    async def initialize(self):
        """Initialize agent components"""
        try:
            # Connect to ROS bridge
            self.bridge_interface = WebSocketBridgeInterface(self.config)
            success = await self.bridge_interface.connect_with_retry()
            if not success:
                raise RuntimeError("Failed to connect to ROS bridge")
            self.logger.info("Connected to ROS bridge")
            
            # Create pipeline
            await self._create_pipeline()
            self.logger.info("Pipeline created")
            
        except Exception as e:
            self.logger.error(f"Initialization error: {e}", exc_info=True)
            raise
            
    async def _create_pipeline(self):
        """Create the Pipecat pipeline"""
        
        # Get configuration
        agent_type = self.config.get('agent_type', 'conversation')
        # Use full model path including 'models/' prefix
        model = self.config.get('model', 'models/gemini-2.0-flash-exp')
        api_key = self.config.get('api_key', os.getenv('GEMINI_API_KEY'))
        
        if not api_key:
            raise ValueError("Gemini API key required")
            
        # Initialize proven prompt loader
        self.prompt_loader = PromptLoader()
        
        # Load system prompt using proven loader
        prompt_id = self.config.get('prompt_id', 'default')
        if prompt_id in self.prompt_loader.prompts:
            system_prompt = self.prompt_loader.prompts[prompt_id].system_prompt
            self.logger.info(f"Loaded prompt '{prompt_id}' using PromptLoader")
        else:
            self.logger.warning(f"Prompt '{prompt_id}' not found in PromptLoader, using default")
            system_prompt = self.prompt_loader.select_prompt()
        
        # Configure modalities based on agent type
        modalities = self._get_modalities(agent_type)
        
        # Create Gemini LLM service
        gemini_params = InputParams(modalities=modalities)
        
        # Create Gemini LLM service with debug logging
        self.logger.info(f"Creating GeminiMultimodalLiveLLMService with model: {model}, modalities: {modalities}")
        
        gemini_llm = GeminiMultimodalLiveLLMService(
            api_key=api_key,
            model=model,
            system_instruction=system_prompt,
            params=gemini_params,
            transcribe_user_audio=True,  # Built-in STT
            voice_id=self.config.get('voice_id', 'Kore'),
        )
        
        self.logger.info("GeminiMultimodalLiveLLMService created successfully")
        
        # Create processors
        self.ros_input = ROSInputProcessor(self.bridge_interface)
        self.ros_output = ROSOutputProcessor(self.bridge_interface, self.config)
        
        # Build pipeline based on agent type
        pipeline_components = [
            self.ros_input,
        ]
        
        # Add Pipecat's VAD for proper turn management (if available)
        if VADAudioFrameProcessor:
            self.logger.info("Adding Pipecat VAD processor for turn management")
            vad_processor = VADAudioFrameProcessor()
            pipeline_components.append(vad_processor)
        else:
            self.logger.warning("Pipecat VAD not available - relying on external VAD")
        
        # Add video throttler for conversation agent
        if agent_type == 'conversation' and 'image' in self.config.get('modalities', []):
            video_fps = self.config.get('video_fps', 1.0)
            pipeline_components.append(VideoFrameThrottler(fps=video_fps))
            
        # Add Gemini LLM
        pipeline_components.append(gemini_llm)
        
        # Add output processor
        pipeline_components.append(self.ros_output)
        
        # Create pipeline
        self.pipeline = Pipeline(pipeline_components)
        
        # Create task with params
        params = PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        )
        self.task = PipelineTask(self.pipeline, params=params)
        
    def _get_modalities(self, agent_type: str):
        """Get modalities based on agent type"""
        
        # Default modalities from config
        config_modalities = self.config.get('modalities', ['audio', 'text'])
        
        # Check what modalities we need
        has_audio = 'audio' in config_modalities
        has_image = 'image' in config_modalities and agent_type == 'conversation'
        
        # For Gemini Live, we need to determine the best modality setting
        # Check available enum members first
        available_modalities = [attr for attr in dir(GeminiMultimodalModalities) if not attr.startswith('_')]
        self.logger.debug(f"Available GeminiMultimodalModalities: {available_modalities}")
        
        # Try to find the best match for our configuration
        if has_audio and has_image:
            # Prefer multimodal if available
            for multimodal_option in ['AUDIO_VIDEO', 'AUDIO_IMAGE', 'MULTIMODAL', 'ALL']:
                if hasattr(GeminiMultimodalModalities, multimodal_option):
                    self.logger.info(f"Using multimodal: {multimodal_option}")
                    return getattr(GeminiMultimodalModalities, multimodal_option)
            # Fall back to AUDIO if no multimodal option
            self.logger.info("No multimodal option found, falling back to AUDIO")
            return GeminiMultimodalModalities.AUDIO
        elif has_audio:
            return GeminiMultimodalModalities.AUDIO
        elif has_image and hasattr(GeminiMultimodalModalities, 'IMAGE'):
            return GeminiMultimodalModalities.IMAGE
        else:
            return GeminiMultimodalModalities.TEXT
        
            
    async def run(self):
        """Run the agent"""
        try:
            if not self.pipeline:
                await self.initialize()
                
            # Create runner
            self.runner = PipelineRunner()
            
            # Start ROS message loop
            message_loop = asyncio.create_task(
                self.ros_input.start_message_loop(self.task)
            )
            
            # Queue start frame to initialize pipeline
            self.logger.info("Queueing StartFrame to initialize pipeline")
            await self.task.queue_frame(StartFrame())
            
            # Run pipeline
            self.logger.info(f"Starting Gemini Live agent - type: {self.config.get('agent_type', 'conversation')}")
            await self.runner.run(self.task)
            
            # Cancel message loop when pipeline ends
            message_loop.cancel()
            
        except Exception as e:
            self.logger.error(f"Agent runtime error: {e}", exc_info=True)
            raise
        finally:
            await self.shutdown()
            
    async def shutdown(self):
        """Shutdown agent cleanly"""
        self.logger.info("Shutting down agent")
        
        # Stop processors
        if self.ros_input:
            self.ros_input.stop()
            
        # Close bridge connection
        if self.bridge_interface:
            await self.bridge_interface.close()
            
        # Log metrics
        runtime = datetime.now() - self.metrics['start_time']
        self.logger.info(f"Agent metrics - Runtime: {runtime}, Messages: {self.metrics['messages_processed']}, Errors: {self.metrics['errors']}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Gemini Live Multimodal Agent")
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--agent-type',
        type=str,
        choices=['conversation', 'command', 'scene'],
        default='conversation',
        help='Agent type'
    )
    parser.add_argument(
        '--prompt-id',
        type=str,
        help='Prompt ID from prompts.yaml'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] [%(name)s] %(levelname)s: %(message)s'
    )
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
        # Get agent-specific config
        agent_config = config.get('gemini_live_agent', {})
        
        # Override with command line arguments
        if args.agent_type:
            agent_config['agent_type'] = args.agent_type
        if args.prompt_id:
            agent_config['prompt_id'] = args.prompt_id
            
        # Add API key from environment if not in config
        if not agent_config.get('api_key'):
            agent_config['api_key'] = os.getenv('GEMINI_API_KEY')
            
        if not agent_config.get('api_key'):
            print("Error: GEMINI_API_KEY not set")
            print("Please set the GEMINI_API_KEY environment variable or add it to the config file")
            print("You can set it by running: export GEMINI_API_KEY='your-api-key'")
            return 1
            
    except Exception as e:
        print(f"Configuration error: {e}")
        return 1
        
    # Create and run agent
    try:
        agent = GeminiLiveAgent(agent_config)
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Fatal error: {e}")
        logging.exception("Fatal error")
        return 1
        
    return 0


if __name__ == '__main__':
    exit(main())