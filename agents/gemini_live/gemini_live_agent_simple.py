#!/usr/bin/env python3
"""
Simplified Gemini Live Agent using google-genai directly

Uses the Google GenAI SDK's Multimodal Live API for real-time voice and vision processing.
Integrates with ROS2 through the existing WebSocket bridge.

Author: Karim Virani
Version: 2.0
Date: August 2025
"""

import asyncio
import logging
import os
import base64
import json
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

# Google GenAI imports
from google import genai
from google.genai import types

# ROS Bridge interface  
from agents.oai_realtime.websocket_bridge import WebSocketBridgeInterface

# Pipecat imports for pipeline structure
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    Frame, 
    StartFrame,
    AudioRawFrame, 
    TextFrame, 
    ImageRawFrame
)


class GeminiLiveProcessor(FrameProcessor):
    """Simple processor that connects to Gemini Live API"""
    
    def __init__(self, api_key: str, model: str = "models/gemini-2.0-flash-exp"):
        super().__init__()
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.session = None
        self.logger = logging.getLogger(__name__)
        
    async def process_frame(self, frame: Frame, direction=None):
        """Process frames through Gemini Live API"""
        
        # Always pass through frame first
        await self.push_frame(frame, direction)
        
        # Initialize session on StartFrame
        if isinstance(frame, StartFrame):
            await self._init_session()
            return  # Don't process StartFrame further
            
        # Process audio frames
        if isinstance(frame, AudioRawFrame):
            await self._process_audio(frame)
            
        # Process image frames  
        elif isinstance(frame, ImageRawFrame):
            await self._process_image(frame)
            
        # Process text frames
        elif isinstance(frame, TextFrame):
            await self._process_text(frame)
        
    async def _init_session(self):
        """Initialize Gemini Live session"""
        try:
            self.logger.info(f"Initializing Gemini Live session with model: {self.model}")
            
            config = types.LiveConnectConfig(
                response_modalities=["AUDIO", "TEXT"],
                system_instruction="You are a helpful robot assistant with vision and hearing capabilities."
            )
            
            # Connect to Live API (using async context manager)
            self.session = await self.client.aio.live.connect(
                model=self.model,
                config=config
            )
            
            # Start response handler
            self.response_task = asyncio.create_task(self._handle_responses())
            
            self.logger.info("Gemini Live session initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to init Gemini session: {e}", exc_info=True)
            raise
            
    async def _process_audio(self, frame: AudioRawFrame):
        """Send audio to Gemini"""
        if self.session and frame.audio:
            try:
                # Convert audio bytes to base64
                audio_b64 = base64.b64encode(frame.audio).decode('utf-8')
                
                # Send to Gemini
                await self.session.send(
                    {"audio": {"data": audio_b64}},
                    end_of_turn=False
                )
                
            except Exception as e:
                self.logger.error(f"Error sending audio: {e}")
                
    async def _process_image(self, frame: ImageRawFrame):
        """Send image to Gemini"""
        if self.session and frame.image is not None:
            try:
                # Convert image to JPEG base64
                import cv2
                
                if isinstance(frame.image, np.ndarray):
                    _, buffer = cv2.imencode('.jpg', frame.image)
                    image_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send to Gemini
                    await self.session.send(
                        {"image": {"data": image_b64, "mime_type": "image/jpeg"}},
                        end_of_turn=False
                    )
                    
            except Exception as e:
                self.logger.error(f"Error sending image: {e}")
                
    async def _process_text(self, frame: TextFrame):
        """Send text to Gemini"""
        if self.session and frame.text:
            try:
                # Send text message
                await self.session.send(frame.text, end_of_turn=True)
                
            except Exception as e:
                self.logger.error(f"Error sending text: {e}")
                
    async def _handle_responses(self):
        """Handle responses from Gemini"""
        if not self.session:
            return
            
        try:
            async for response in self.session.receive():
                # Handle text responses
                if hasattr(response, 'text') and response.text:
                    text_frame = TextFrame(text=response.text)
                    await self.push_frame(text_frame)
                    
                # Handle audio responses
                if hasattr(response, 'audio') and response.audio:
                    audio_bytes = base64.b64decode(response.audio)
                    audio_frame = AudioRawFrame(
                        audio=audio_bytes,
                        sample_rate=24000,
                        num_channels=1
                    )
                    await self.push_frame(audio_frame)
                    
        except Exception as e:
            self.logger.error(f"Error handling responses: {e}")


class ROSInputProcessor(FrameProcessor):
    """Simple pass-through processor (messages are queued externally)"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._started = False
        
    async def process_frame(self, frame: Frame, direction=None):
        """Just pass frames through"""
        # Handle StartFrame to mark processor as started
        if isinstance(frame, StartFrame):
            self._started = True
            
        await self.push_frame(frame, direction)


class ROSOutputProcessor(FrameProcessor):
    """Convert Pipecat frames to ROS messages"""
    
    def __init__(self, bridge_interface: WebSocketBridgeInterface):
        super().__init__()
        self.bridge = bridge_interface
        self.logger = logging.getLogger(__name__)
        self._started = False
        
    async def process_frame(self, frame: Frame, direction=None):
        """Convert frames to ROS messages"""
        
        # Handle StartFrame
        if isinstance(frame, StartFrame):
            self._started = True
            await self.push_frame(frame, direction)
            return
        
        # Convert audio frames
        if isinstance(frame, AudioRawFrame):
            # Convert to ROS audio message
            audio_array = np.frombuffer(frame.audio, dtype=np.int16)
            
            message = {
                'data': audio_array.tolist(),
                'sample_rate': frame.sample_rate
            }
            
            await self.bridge.put_outbound_message('audio_out', message, 'audio_common_msgs/AudioData')
            
        # Convert text frames
        elif isinstance(frame, TextFrame):
            # Send as transcript
            message = {'data': frame.text}
            
            # Route based on type
            if hasattr(frame, 'response_type'):
                if frame.response_type == 'command':
                    await self.bridge.put_outbound_message('command_transcript', message, 'std_msgs/String')
                elif frame.response_type == 'scene':
                    await self.bridge.put_outbound_message('scene_description', message, 'std_msgs/String')
                else:
                    await self.bridge.put_outbound_message('llm_transcript', message, 'std_msgs/String')
            else:
                await self.bridge.put_outbound_message('llm_transcript', message, 'std_msgs/String')
                
        # Pass through
        await self.push_frame(frame, direction)


class GeminiLiveAgent:
    """Simplified Gemini Live Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.bridge_interface = None
        self.pipeline = None
        self.running = False
        
    async def initialize(self):
        """Initialize agent components"""
        
        # Connect to ROS bridge using config dict
        self.bridge_interface = WebSocketBridgeInterface(self.config)
        
        # Connect with retry
        success = await self.bridge_interface.connect_with_retry()
        if not success:
            raise RuntimeError("Failed to connect to ROS bridge")
        self.logger.info("Connected to ROS bridge")
        
        # Create simplified pipeline (minimal for testing)
        gemini_proc = GeminiLiveProcessor(
            api_key=self.config.get('api_key'),
            model=self.config.get('model', 'models/gemini-2.0-flash-exp')
        )
        
        # Create pipeline with just Gemini processor
        self.pipeline = Pipeline([
            gemini_proc
        ])
        
        self.logger.info("Pipeline created")
        
    async def run(self):
        """Run the agent"""
        self.running = True
        
        try:
            if not self.bridge_interface:
                await self.initialize()
                
            # Create pipeline task
            self.task = PipelineTask(
                self.pipeline,
                idle_timeout_secs=None  # Don't timeout
            )
            
            # Create and start pipeline runner first
            runner = PipelineRunner()
            runner_task = asyncio.create_task(runner.run(self.task))
            
            # Wait a moment for pipeline to be ready
            await asyncio.sleep(0.2)
            
            # Queue a single start frame to initialize pipeline
            start_frame = StartFrame()
            await self.task.queue_frame(start_frame)
            self.logger.info("Queued StartFrame to pipeline")
            
            # Now start message processor task
            message_task = asyncio.create_task(self._process_messages())
            
            # Wait for both tasks
            await asyncio.gather(runner_task, message_task)
                    
        except Exception as e:
            self.logger.error(f"Agent error: {e}", exc_info=True)
        finally:
            self.running = False
            await self.shutdown()
            
    async def _process_messages(self):
        """Process incoming messages from ROS and feed to pipeline"""
        self.logger.info("Starting message processor")
        
        while self.running:
            try:
                # Get message from bridge
                envelope = await self.bridge_interface.get_inbound_message(timeout=0.1)
                
                if envelope:
                    msg_type = envelope.ros_msg_type
                    
                    # Log received messages
                    if msg_type == 'by_your_command/AudioDataUtterance':
                        audio_data = envelope.raw_data.int16_data if hasattr(envelope.raw_data, 'int16_data') else []
                        if audio_data:
                            self.logger.info(f"Received audio data: {len(audio_data)} samples")
                            # Store for processing in Gemini processor
                            # We'll access this through the bridge in the processor
                            
                    elif msg_type == 'sensor_msgs/Image':
                        self.logger.info("Received image data")
                        
                    elif msg_type == 'std_msgs/String':
                        text = envelope.raw_data.data if hasattr(envelope.raw_data, 'data') else ''
                        self.logger.info(f"Received text: {text}")
                            
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Message processing error: {e}", exc_info=True)
            
    async def shutdown(self):
        """Shutdown agent"""
        self.logger.info("Shutting down agent")
        
        if self.bridge_interface:
            await self.bridge_interface.close()
            
        self.running = False


async def main():
    """Test entry point"""
    import yaml
    
    # Load config
    config_path = "/home/karim/ros2_ws/src/by_your_command/config/gemini_live_agent.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    agent_config = config.get('gemini_live_agent', {})
    
    # Add API key from environment
    agent_config['api_key'] = os.getenv('GEMINI_API_KEY', '')
    
    if not agent_config['api_key']:
        print("Error: GEMINI_API_KEY environment variable not set")
        return
        
    # Create and run agent
    agent = GeminiLiveAgent(agent_config)
    await agent.run()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())