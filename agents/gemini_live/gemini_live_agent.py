#!/usr/bin/env python3
"""
Gemini Live Agent with Pipecat Pipeline

Multimodal agent for real-time voice and vision processing using Google's Gemini Live API
and Pipecat's pipeline architecture. Integrates with ROS2 through the existing WebSocket bridge.

Author: Karim Virani
Version: 1.0
Date: August 2025
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, AsyncGenerator
from dataclasses import dataclass
import base64
import numpy as np

# Pipecat imports
try:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext  # Use as base class example
    from pipecat.frames.frames import Frame, AudioRawFrame, TextFrame, ImageRawFrame
    # Note: Pipecat uses different frame names than we expected
    AudioFrame = AudioRawFrame  # Alias for compatibility
    ImageFrame = ImageRawFrame  # Alias for compatibility
    
    # Base processor class
    from pipecat.processors.frame_processor import FrameProcessor as Processor
except ImportError as e:
    raise ImportError(f"Pipecat required: pip install pipecat-ai. Error: {e}")

# Import reusable components from OpenAI agent
from ..oai_realtime.websocket_bridge import WebSocketBridgeInterface
from ..oai_realtime.context import ConversationContext, ContextManager

# Import custom processors
from .gemini_bridge_processor import GeminiLiveBridgeProcessor
from .visual_processor import VisualProcessor
from .serializers import GeminiSerializer


@dataclass
class GeminiConfig:
    """Configuration for Gemini Live agent"""
    agent_id: str = "gemini_live"
    agent_type: str = "multimodal"  # conversation|command|visual|multimodal
    api_key: str = ""
    model: str = "gemini-2.0-flash-exp"
    
    # Modalities
    modalities: List[str] = None
    
    # Audio settings
    audio_input_sample_rate: int = 16000
    audio_output_sample_rate: int = 24000
    voice: str = "default"
    
    # Video settings
    video_enabled: bool = True
    video_fps: float = 1.0
    video_max_fps: float = 10.0
    video_min_fps: float = 0.1
    video_resolution: str = "480p"
    video_dynamic_fps: bool = True
    
    # Session settings
    session_timeout: float = 300.0
    reconnect_attempts: int = 5
    
    # Topics
    voice_topic: str = "voice_chunks"
    camera_topic: str = "camera/image_raw"
    audio_out_topic: str = "audio_out"
    transcript_topic: str = "llm_transcript"
    command_topic: str = "command_transcript"
    scene_topic: str = "scene_description"
    
    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ["audio", "vision", "text"]


class ROSInputProcessor(Processor):
    """Base processor for ROS input handling"""
    
    def __init__(self, agent: 'GeminiLiveAgent', topic_name: str):
        super().__init__()
        self.agent = agent
        self.topic_name = topic_name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    async def process(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Process frames - to be implemented by subclasses"""
        yield frame


class ROSVoiceInputProcessor(ROSInputProcessor):
    """Process voice chunks from ROS topics"""
    
    async def process(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Convert ROS AudioDataUtterance to Pipecat AudioFrame"""
        if hasattr(frame, 'envelope'):
            envelope = frame.envelope
            if envelope.ros_msg_type == "by_your_command/AudioDataUtterance":
                audio_data = envelope.raw_data
                
                # Convert int16 data to bytes for Gemini
                if audio_data.int16_data:
                    pcm_bytes = np.array(audio_data.int16_data, dtype=np.int16).tobytes()
                    
                    # Create Pipecat AudioFrame
                    audio_frame = AudioFrame(
                        audio=pcm_bytes,
                        sample_rate=self.agent.config.audio_input_sample_rate,
                        channels=1
                    )
                    
                    # Add metadata
                    audio_frame.utterance_id = audio_data.utterance_id
                    audio_frame.is_utterance_end = audio_data.is_utterance_end
                    
                    yield audio_frame


class ROSCameraInputProcessor(ROSInputProcessor):
    """Process camera frames from ROS topics"""
    
    async def process(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Convert ROS Image to Pipecat ImageFrame"""
        if hasattr(frame, 'envelope'):
            envelope = frame.envelope
            if envelope.ros_msg_type == "sensor_msgs/Image":
                image_msg = envelope.raw_data
                
                # Convert ROS image to numpy array
                # This is simplified - in practice would use cv_bridge
                if image_msg.encoding == "rgb8":
                    height = image_msg.height
                    width = image_msg.width
                    channels = 3
                    img_array = np.frombuffer(image_msg.data, dtype=np.uint8)
                    img_array = img_array.reshape((height, width, channels))
                    
                    # Create Pipecat ImageFrame
                    image_frame = ImageFrame(
                        image=img_array,
                        size=(width, height),
                        format="rgb"
                    )
                    
                    # Add timestamp metadata
                    image_frame.timestamp = envelope.timestamp
                    
                    yield image_frame


class ResponseRouterProcessor(Processor):
    """Route Gemini responses to appropriate outputs"""
    
    def __init__(self, agent: 'GeminiLiveAgent'):
        super().__init__()
        self.agent = agent
        self.logger = logging.getLogger(f"{__name__}.ResponseRouter")
        
    async def process(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Route frames based on type"""
        if isinstance(frame, AudioFrame):
            # Audio response - route to speaker output
            yield frame
            
        elif isinstance(frame, TextFrame):
            # Text response - determine type and route
            if hasattr(frame, 'response_type'):
                if frame.response_type == 'transcript':
                    await self._publish_transcript(frame.text)
                elif frame.response_type == 'command':
                    await self._publish_command(frame.text)
                elif frame.response_type == 'scene':
                    await self._publish_scene(frame.text)
            yield frame
            
    async def _publish_transcript(self, text: str):
        """Publish conversation transcript to ROS"""
        if self.agent.bridge_interface:
            await self.agent.bridge_interface.publish_message(
                self.agent.config.transcript_topic,
                {"data": text},
                "std_msgs/String"
            )
            
    async def _publish_command(self, command: str):
        """Publish extracted command to ROS"""
        if self.agent.bridge_interface:
            await self.agent.bridge_interface.publish_message(
                self.agent.config.command_topic,
                {"data": command},
                "std_msgs/String"
            )
            
    async def _publish_scene(self, description: str):
        """Publish scene description to ROS"""
        if self.agent.bridge_interface:
            await self.agent.bridge_interface.publish_message(
                self.agent.config.scene_topic,
                {"data": description},
                "std_msgs/String"
            )


class ROSAudioOutputProcessor(Processor):
    """Output audio to ROS topics"""
    
    def __init__(self, agent: 'GeminiLiveAgent'):
        super().__init__()
        self.agent = agent
        self.logger = logging.getLogger(f"{__name__}.AudioOutput")
        
    async def process(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        """Publish audio frames to ROS"""
        if isinstance(frame, AudioFrame):
            # Convert audio bytes to int16 array for ROS
            audio_array = np.frombuffer(frame.audio, dtype=np.int16)
            
            # Publish to ROS topic
            if self.agent.bridge_interface:
                await self.agent.bridge_interface.publish_message(
                    self.agent.config.audio_out_topic,
                    {"int16_data": audio_array.tolist()},
                    "audio_common_msgs/AudioData"
                )
                
        yield frame  # Pass through for potential downstream processors


class GeminiLiveAgent:
    """
    Gemini Live agent using Pipecat pipeline architecture.
    
    Features:
    - Multimodal processing (voice + vision)
    - Flexible pipeline composition
    - ROS2 integration via WebSocket bridge
    - Support for multiple deployment configurations
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # Initialize configuration
        config_dict = config or {}
        self.config = GeminiConfig(**{k: v for k, v in config_dict.items() 
                                      if k in GeminiConfig.__annotations__})
        
        # Initialize components
        self.bridge_interface: Optional[WebSocketBridgeInterface] = None
        self.pipeline: Optional[Pipeline] = None
        self.serializer = GeminiSerializer()
        
        # Context management (reuse from OpenAI agent)
        self.context_manager = ContextManager(
            max_context_tokens=config_dict.get('max_context_tokens', 2000),
            max_context_age=config_dict.get('max_context_age', 600.0)
        )
        
        # State
        self.running = False
        self.session_active = False
        
        # Metrics
        self.metrics = {
            'messages_processed': 0,
            'frames_processed': 0,
            'audio_frames_sent': 0,
            'image_frames_sent': 0,
            'responses_received': 0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(config_dict.get('log_level', logging.INFO))
        
    async def initialize(self):
        """Initialize the agent and create pipeline"""
        self.logger.info(f"Initializing Gemini Live agent: {self.config.agent_id}")
        
        # Initialize bridge interface
        bridge_config = {
            'bridge_connection': {
                'host': 'localhost',
                'port': 8765
            },
            'agent_id': self.config.agent_id
        }
        self.bridge_interface = WebSocketBridgeInterface(bridge_config)
        
        # Create Pipecat pipeline
        await self._create_pipeline()
        
        # Connect to bridge
        await self.bridge_interface.connect()
        
        self.logger.info("Gemini Live agent initialized successfully")
        
    async def _create_pipeline(self):
        """Create the Pipecat processing pipeline"""
        
        # Create processors
        voice_input = ROSVoiceInputProcessor(self, self.config.voice_topic)
        camera_input = ROSCameraInputProcessor(self, self.config.camera_topic) if self.config.video_enabled else None
        
        # Create Gemini bridge processor
        gemini_bridge = GeminiLiveBridgeProcessor(
            api_key=self.config.api_key,
            model=self.config.model,
            modalities=self.config.modalities,
            config=self.config
        )
        
        # Visual processor for scene analysis (if needed)
        visual_processor = VisualProcessor(self.config) if self.config.video_enabled else None
        
        # Response routing
        response_router = ResponseRouterProcessor(self)
        
        # Audio output
        audio_output = ROSAudioOutputProcessor(self)
        
        # Build pipeline based on configuration
        processors = []
        
        # Input processors
        processors.append(voice_input)
        if camera_input and self.config.video_enabled:
            processors.append(camera_input)
            
        # Core processing
        processors.append(gemini_bridge)
        
        # Visual analysis (if applicable)
        if visual_processor and self.config.agent_type in ["visual", "multimodal"]:
            processors.append(visual_processor)
            
        # Output processors
        processors.append(response_router)
        processors.append(audio_output)
        
        # Create pipeline (Pipecat expects a list)
        self.pipeline = Pipeline(processors)
        
        self.logger.info(f"Created pipeline with {len(processors)} processors")
        
    async def run(self):
        """Main agent execution loop"""
        self.running = True
        self.logger.info(f"Starting Gemini Live agent: {self.config.agent_id}")
        
        try:
            # Initialize if not already done
            if not self.bridge_interface:
                await self.initialize()
                
            # Create pipeline runner
            pipeline_runner = self.pipeline.run() if self.pipeline else None
            
            # Start pipeline task
            if pipeline_runner:
                pipeline_task = asyncio.create_task(pipeline_runner)
                self.logger.info("Pipeline runner started")
            else:
                pipeline_task = asyncio.create_task(asyncio.sleep(0))  # Dummy task
                
            # Start message processing from bridge
            bridge_task = asyncio.create_task(self._process_bridge_messages())
            
            # Initialize pipeline with StartFrame after it's running
            if pipeline_runner:
                await self._start_pipeline()
            
            # Wait for both tasks
            await asyncio.gather(pipeline_task, bridge_task)
            
        except Exception as e:
            self.logger.error(f"Agent error: {e}", exc_info=True)
        finally:
            self.running = False
            await self.shutdown()
            
    async def _start_pipeline(self):
        """Initialize the Pipecat pipeline with StartFrame"""
        if self.pipeline:
            # Wait a moment for pipeline to be ready
            await asyncio.sleep(0.1)
            
            # Send StartFrame to initialize the pipeline
            from pipecat.frames.frames import StartFrame
            start_frame = StartFrame()
            await self.pipeline.push_frame(start_frame)
            self.logger.info("Pipeline initialized with StartFrame")
            
    async def _process_bridge_messages(self):
        """Process messages from ROS bridge"""
        while self.running:
            try:
                # Get message from bridge
                envelope = await self.bridge_interface.get_inbound_message(timeout=1.0)
                
                if envelope:
                    self.metrics['messages_processed'] += 1
                    
                    # Convert envelope to appropriate Pipecat frame type
                    frame = await self._envelope_to_frame(envelope)
                    
                    # Inject frame into pipeline
                    if self.pipeline and frame:
                        await self.pipeline.push_frame(frame)
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing bridge message: {e}")
                
    async def _envelope_to_frame(self, envelope: Dict[str, Any]) -> Optional[Frame]:
        """Convert WebSocket envelope to appropriate Pipecat frame"""
        try:
            msg_type = envelope.get('msg_type', '')
            
            if msg_type == 'by_your_command/AudioDataUtterance':
                # Convert audio data to AudioRawFrame
                audio_data = envelope.get('int16_data', [])
                if audio_data:
                    # Convert list to numpy array
                    audio_array = np.array(audio_data, dtype=np.int16)
                    # Convert to bytes for Pipecat
                    audio_bytes = audio_array.tobytes()
                    return AudioRawFrame(
                        audio=audio_bytes,
                        sample_rate=16000,
                        num_channels=1
                    )
                    
            elif msg_type == 'sensor_msgs/Image':
                # Convert image data to ImageRawFrame
                data = envelope.get('data', [])
                if data:
                    # Image data is usually in uint8 format
                    image_array = np.array(data, dtype=np.uint8)
                    width = envelope.get('width', 0)
                    height = envelope.get('height', 0)
                    encoding = envelope.get('encoding', 'rgb8')
                    
                    # Reshape based on encoding
                    if encoding == 'rgb8':
                        image_array = image_array.reshape((height, width, 3))
                    elif encoding == 'bgr8':
                        # Convert BGR to RGB
                        image_array = image_array.reshape((height, width, 3))
                        image_array = image_array[:, :, ::-1]  # BGR to RGB
                    
                    return ImageRawFrame(
                        image=image_array,
                        size=(width, height),
                        format="RGB"
                    )
                    
            elif msg_type == 'std_msgs/String':
                # Convert text to TextFrame
                text = envelope.get('data', '')
                if text:
                    return TextFrame(text=text)
                    
            else:
                self.logger.debug(f"Unknown message type: {msg_type}")
                
        except Exception as e:
            self.logger.error(f"Error converting envelope to frame: {e}")
            
        return None
    
    async def shutdown(self):
        """Clean shutdown of agent"""
        self.logger.info("Shutting down Gemini Live agent")
        
        # Stop pipeline
        if self.pipeline:
            # Pipecat pipelines don't have a stop() method
            # Cleanup is handled by individual processors
            pass
            
        # Disconnect from bridge
        if self.bridge_interface:
            # TODO: Add disconnect method to WebSocketBridgeInterface
            # await self.bridge_interface.disconnect()
            pass
            
        self.logger.info(f"Agent shutdown complete. Metrics: {self.metrics}")


async def main():
    """Standalone test entry point"""
    import yaml
    import os
    
    # Load configuration
    config_path = os.path.join(
        os.path.dirname(__file__), 
        "../../config/gemini_live_agent.yaml"
    )
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            agent_config = config.get('gemini_live_agent', {})
    else:
        agent_config = {}
        
    # Override with environment variables
    if os.environ.get('GEMINI_API_KEY'):
        agent_config['api_key'] = os.environ['GEMINI_API_KEY']
        
    # Create and run agent
    agent = GeminiLiveAgent(agent_config)
    await agent.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())