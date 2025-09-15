# Gemini Live Agent Implementation - Product Requirements Document

> **Note: This document predates the topic renaming refactoring (2025-09-14). Some topic names mentioned here have been updated in the implementation. See `topic_renaming_refactoring_prd.md` for current naming.**


**Author**: Karim Virani  
**Version**: 2.0  
**Date**: September 2025  
**Status**: Implemented (without Pipecat)

## 1. Executive Summary

This document originally outlined requirements for integrating Google's Gemini Live API using Pipecat's pipeline architecture. The actual implementation was completed using a direct WebSocket approach similar to the OpenAI agent, without Pipecat dependencies. The system successfully enables multimodal human-robot interactions with real-time voice conversations and visual scene understanding, supporting dual-agent configurations for conversation and command extraction.

### Implementation Status
- ✅ Dual Gemini agent configuration (conversation + command extraction)
- ✅ Visual scene understanding with native bounding box format
- ✅ Smart voice-triggered frame forwarding (reduces latency from 500ms to ~50ms)
- ✅ User and assistant transcription support
- ✅ Text buffering for fragmented responses
- ❌ Pipecat integration (deemed unnecessary)
- ❌ Triple agent configuration (not yet implemented)

## 2. Problem Statement

### 2.1 Current Limitations
- **Single Modality**: Existing OpenAI Realtime agent handles only voice/audio
- **No Visual Understanding**: Robot cannot analyze or respond to visual scenes
- **Limited Flexibility**: Current architecture tightly coupled to OpenAI's specific protocol
- **Cost Constraints**: OpenAI's aggressive token pricing requires complex session cycling

### 2.2 Opportunities with Gemini Live
- **Native Multimodal**: Voice, text, and vision in a single unified model
- **Better Pricing Model**: No need for aggressive session cycling
- **Superior Turn Detection**: Advanced semantic understanding of conversation flow
- **Flexible Frame Rates**: Dynamic FPS from 0.1 to 60 for different scenarios

## 3. Actual Implementation

### 3.1 Architecture Decision - No Pipecat Required

The Gemini Live agent was successfully implemented using a direct approach similar to the OpenAI agent, without requiring Pipecat. This decision was made because:
- The existing WebSocket bridge architecture was sufficient
- Direct integration provided better control over the Gemini Live protocol
- Reduced dependencies and complexity
- Easier debugging and maintenance

### 3.2 Key Implementation Learnings

#### 3.2.1 Transcription Support
Contrary to initial assumptions, Gemini Live API **does** provide transcription capabilities:
- Enable with `input_audio_transcription: {}` in session config
- Enable output transcription with `output_audio_transcription: {}` for conversational agents
- Transcriptions arrive via `server_content.input_transcription` and `server_content.output_transcription`

#### 3.2.2 Response Modality Selection
- Command extraction agents should use `TEXT` modality only (no audio output)
- Conversational agents use `AUDIO` modality for natural speech
- Set by configuring `response_modalities: ["TEXT"]` or `["AUDIO"]` in session config

#### 3.2.3 Fragmented Text Responses
Command agents output text in fragments that must be buffered:
- Accumulate fragments in `text_buffer`
- Process complete response only on `turn_complete` signal
- Clean markdown wrappers if present (e.g., ` ```json...``` `)

#### 3.2.4 Native Bounding Box Format
Gemini prefers its own spatial understanding format:
- Accepts formats like `{"box_2d": [x1, y1, x2, y2], "label": "object"}`
- Also supports `{"label": "object", "box_3d": [x, y, z, ...]}`
- Don't force specific formats - let Gemini use its native understanding

### 3.3 Smart Frame Forwarding Solution

#### Problem: Frame Timing Race Condition
When users asked "what do you see?" immediately after robot movement, Gemini would describe the old position because:
- Fixed 2fps rate limiting caused up to 500ms staleness
- Voice prompts triggered before new frames arrived
- Images were sent with voice, but were from before movement

#### Solution: Hybrid Approach
Implemented smart voice-triggered frame forwarding with baseline frames:

1. **Baseline Frames**: Send at reduced rate (0.5 fps) to ensure agents always have recent frames
2. **Triggered Forwarding**: When voice/text detected, immediately forward cached frames
3. **Frame Caching**: Store all incoming frames without rate limiting
4. **Age Filtering**: Only forward frames less than 1 second old

Results:
- Frame latency reduced from 500ms to ~50ms
- API efficiency maintained (98% frame drop rate)
- Reliable vision responses even after robot movement

#### Future Enhancement: Audio Buffering at Utterance Start
To completely eliminate any remaining race conditions, an optional audio buffering mechanism was designed but not yet implemented:

```python
# In receive_coordinator.py (not yet implemented)
class ReceiveCoordinator:
    def __init__(self):
        self.audio_buffer = []
        self.waiting_for_frame = False
        
    async def handle_first_chunk(self, audio_chunk):
        # On first voice chunk (sequence=0)
        if self.video_enabled:
            self.waiting_for_frame = True
            self.audio_buffer.append(audio_chunk)
            # Wait up to 100ms for fresh frame
            await asyncio.sleep(0.1)
            
    async def on_frame_received(self):
        if self.waiting_for_frame:
            # Send buffered audio after frame arrives
            for chunk in self.audio_buffer:
                await self.session.send(chunk)
            self.audio_buffer.clear()
            self.waiting_for_frame = False
```

This would ensure the freshest possible frame is sent with voice queries, though current ~50ms latency is typically sufficient.

### 3.4 Actual Architecture

```python
# Conceptual Pipeline Structure
pipeline = Pipeline(
    ROSVoiceInput(),           # Subscribe to /prompt_voice
    ROSCameraInput(),          # Subscribe to /camera/image_raw
    GeminiLiveBridge(),        # Custom processor for Gemini Live API
    ResponseRouter(),          # Route to appropriate outputs
    ROSTranscriptOutput(),     # Publish to /response_text
    ROSAudioOutput(),          # Publish to /response_voice
    ROSCommandOutput()         # Publish to /response_cmd
)
```

### 3.5 Actual Implementation Components

1. **GeminiSessionManager**: Manages WebSocket connection to Gemini Live API
   - Handles session lifecycle and configuration
   - Manages audio/video streaming
   - Implements the unified `session.send()` API for all content types

2. **ReceiveCoordinator**: Coordinates response handling
   - Manages receive generators per turn
   - Handles transcription events
   - Buffers text fragments for command agents
   - Implements silence hack for Gemini VAD

3. **Prompt System**: Dynamic prompt loading with macro expansion
   - Supports recursive macro expansion
   - Separate prompts for conversation and command agents
   - Native Gemini visual descriptor macros

4. **Frame Forwarding**: Smart image frame delivery
   - Caches frames at bridge level
   - Triggers forwarding on voice/text detection
   - Hybrid baseline + triggered approach

### 3.3 Comparison with OpenAI Agent

| Component | OpenAI Agent | Gemini Live Agent |
|-----------|-------------|-------------------|
| Architecture | Direct WebSocket management | Pipecat pipeline framework |
| Session Management | Aggressive cycling for cost | Persistent connections |
| Modalities | Audio only | Audio + Vision |
| Turn Detection | VAD-based with manual triggers | Semantic turn understanding |
| Context Preservation | Text-based between sessions | Continuous multimodal context |
| Concurrency Model | Single asyncio loop | Pipecat async processors |

## 4. Functional Requirements

### 4.1 Core Capabilities

#### 4.1.1 Voice Interaction
- **Input**: Process 16kHz PCM audio from `/prompt_voice` topic
- **Output**: Generate 24kHz PCM audio responses
- **Transcription**: Real-time speech-to-text with speaker diarization
- **Natural Conversation**: Support interruptions and turn-taking

#### 4.1.2 Visual Scene Analysis
- **Input Sources**:
  - Camera frames from `/camera/image_raw`
  - Depth images from `/camera/depth/image_raw` (optional)
  - Point clouds from `/camera/points` (future)
  
- **Analysis Capabilities**:
  - Object detection and labeling
  - Spatial relationships ("left of", "behind", "near")
  - Scene description for navigation
  - Face detection and recognition (privacy-aware)
  - Gesture recognition
  
- **Output Formats**:
  - Structured JSON with bounding boxes
  - Natural language descriptions
  - Command extraction from visual context

#### 4.1.3 Multimodal Fusion
- Correlate voice commands with visual context
- Use visual grounding for ambiguous references ("that", "there", "the red one")
- Generate contextually aware responses combining audio and visual understanding

### 4.2 Deployment Configurations

#### 4.2.1 Single Gemini Visual Agent
```yaml
# Single agent focused on visual analysis
agent_type: "visual"
modalities: ["vision", "text"]
system_prompt: "You are a visual analysis system..."
```

#### 4.2.2 Mixed OpenAI + Gemini
```yaml
# OpenAI for conversation, Gemini for vision
openai_agents:
  - type: "conversation"
  - type: "command"
gemini_agents:
  - type: "visual"
```

#### 4.2.3 Triple Gemini Agents
```yaml
# Three specialized Gemini agents
agents:
  - id: "conversation"
    modalities: ["audio", "text"]
    prompt: "conversational_assistant"
  - id: "command"  
    modalities: ["audio", "text"]
    prompt: "command_extractor"
  - id: "visual"
    modalities: ["vision", "text", "audio"]
    prompt: "scene_analyzer"
```

### 4.3 Message Flow

#### 4.3.1 Input Processing
```
/prompt_voice → ROSVoiceInput → GeminiLiveBridge
                                        ↓
/camera/image_raw → ROSCameraInput →→→→↓
                                        ↓
                                 Gemini Live API
```

#### 4.3.2 Output Distribution
```
Gemini Live API
        ↓
ResponseRouter
        ├→ /response_voice (voice response)
        ├→ /response_text (conversation text)
        ├→ /response_cmd (extracted commands)
        └→ /scene_description (visual analysis)
```

## 5. Technical Requirements

### 5.1 Pipecat Pipeline Components

#### 5.1.1 Custom Processors

```python
class ROSVoiceInput(Processor):
    """Subscribe to voice chunks from ROS"""
    async def process(self, frame):
        # Convert AudioDataUtterance to Gemini format
        pass

class ROSCameraInput(Processor):
    """Subscribe to camera frames from ROS"""
    async def process(self, frame):
        # Convert sensor_msgs/Image to Gemini format
        pass

class GeminiLiveBridge(Processor):
    """Interface with Gemini Live API"""
    def __init__(self, session_manager):
        self.session = session_manager
        
    async def process(self, frame):
        # Send to Gemini, yield responses
        pass

class ResponseRouter(Processor):
    """Route responses to appropriate outputs"""
    async def process(self, frame):
        if frame.type == "audio":
            yield AudioFrame(frame.data)
        elif frame.type == "transcript":
            yield TextFrame(frame.text)
        elif frame.type == "command":
            yield CommandFrame(frame.command)
```

#### 5.1.2 ROS Integration Pattern

```python
class GeminiLiveNode(Node):
    def __init__(self):
        super().__init__('gemini_live_agent')
        
        # Create Pipecat pipeline
        self.pipeline = Pipeline(
            ROSVoiceInput(self),
            ROSCameraInput(self),
            GeminiLiveBridge(self.session),
            ResponseRouter(),
            ROSAudioOutput(self),
            ROSTranscriptOutput(self)
        )
        
        # ROS subscribers
        self.voice_sub = self.create_subscription(
            AudioDataUtterance,
            'prompt_voice',
            self.voice_callback,
            10
        )
        
        self.camera_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.camera_callback,
            10
        )
        
    async def run_pipeline(self):
        await self.pipeline.run()
```

### 5.2 Gemini Live API Requirements

#### 5.2.1 Audio Specifications
- **Input**: 16-bit PCM @ 16kHz (matches our VAD output)
- **Output**: 16-bit PCM @ 24kHz (will resample for speakers)
- **Format Declaration**: `"audio/pcm;rate=16000"`

#### 5.2.2 Video Specifications
- **Frame Rate**: Configurable 0.1-60 FPS
- **Resolution**: 360p/480p/720p based on bandwidth
- **Encoding**: JPEG or raw RGB
- **Optimization**: Adjust FPS based on scene dynamics

#### 5.2.3 Session Configuration
```json
{
  "model": "gemini-2.0-flash-exp",
  "modalities": ["audio", "video", "text"],
  "system_instruction": "...",
  "audio_config": {
    "input_sample_rate": 16000,
    "output_sample_rate": 24000
  },
  "video_config": {
    "fps": 1.0,  // Start low, adjust dynamically
    "resolution": "480p"
  }
}
```

### 5.3 ROS AI Bridge Enhancements

#### 5.3.1 Multi-Agent Support
```python
class EnhancedBridge:
    def __init__(self):
        self.agents = {}  # agent_id -> AgentInterface
        self.broadcast_topics = set()  # Topics sent to all agents
        
    async def register_agent(self, agent_id, agent_type):
        self.agents[agent_id] = AgentInterface(agent_id, agent_type)
        
    async def broadcast_message(self, envelope):
        """Send message to all registered agents"""
        for agent in self.agents.values():
            await agent.send(envelope)
```

#### 5.3.2 Camera Topic Support with Frame Throttling
```yaml
subscribed_topics:
  - topic: "prompt_voice"
    msg_type: "by_your_command/AudioDataUtterance"
    broadcast: true  # Send to all agents
  - topic: "camera/image_raw"
    msg_type: "sensor_msgs/Image"
    broadcast: false  # Only to visual agents
    throttle:
      enabled: true
      rate: 1.0  # Target FPS for AI agents
      strategy: "latest"  # latest|sample|average
  - topic: "camera/depth/image_raw"
    msg_type: "sensor_msgs/Image"
    broadcast: false
    throttle:
      enabled: true
      rate: 0.5  # Even lower rate for depth
```

#### 5.3.3 Frame Throttling Implementation
```python
class TopicThrottler:
    """
    Throttle high-rate topics before sending to AI agents.
    
    Allows other ROS consumers to receive full frame rate while
    reducing load and costs for AI processing.
    """
    
    def __init__(self, target_fps: float, strategy: str = "latest"):
        self.target_fps = target_fps
        self.strategy = strategy
        self.min_interval = 1.0 / target_fps
        self.last_sent_time = 0
        self.buffer = None
        
    def should_forward(self, envelope: MessageEnvelope) -> bool:
        """Determine if this frame should be forwarded to agents"""
        current_time = envelope.timestamp
        time_since_last = current_time - self.last_sent_time
        
        if self.strategy == "latest":
            # Always keep latest frame, send at throttled rate
            self.buffer = envelope
            if time_since_last >= self.min_interval:
                self.last_sent_time = current_time
                return True
                
        elif self.strategy == "sample":
            # Simple sampling - drop intermediate frames
            if time_since_last >= self.min_interval:
                self.last_sent_time = current_time
                return True
                
        elif self.strategy == "average":
            # Future: Could implement frame averaging
            pass
            
        return False
```

## 6. Non-Functional Requirements

### 6.1 Performance
- **Latency**: < 500ms for voice response (including network)
- **Frame Processing**: Maintain real-time processing at selected FPS
- **CPU Usage**: < 30% on target hardware (Jetson or equivalent)
- **Memory**: < 2GB RAM per agent

### 6.2 Reliability
- **Reconnection**: Automatic reconnection on network failures
- **Graceful Degradation**: Fall back to audio-only if vision unavailable
- **Error Recovery**: Continue operation if individual processors fail

### 6.3 Scalability
- **Multi-Agent**: Support up to 3 concurrent agents
- **Dynamic FPS**: Adjust frame rate based on available resources
- **Queue Management**: Prevent memory overflow with bounded queues

### 6.4 Security & Privacy
- **No Recording**: Process streams in real-time without persistent storage
- **Face Anonymization**: Optional blurring before transmission
- **Local Processing**: Option to use local models for sensitive data

## 7. Actual Configuration

### 7.1 Dual Agent Configuration (Currently Deployed)
```yaml
# Conversational Agent (gemini_conversational_agent.yaml)
gemini_live_agent:
  agent_id: "gemini_conversational"
  agent_type: "conversation"
  
  # Gemini API settings
  api_key: ""  # Via GEMINI_API_KEY env var
  model: "gemini-2.0-flash-exp"
  
  # Modalities
  modalities:
    - audio
    - vision
    - text
    
  # Audio settings
  audio:
    input_sample_rate: 16000
    output_sample_rate: 24000
    voice: "default"
    
  # Video settings
  video:
    enabled: true
    source_topic: "/camera/image_raw"
    fps: 1.0  # Initial FPS
    max_fps: 10.0
    min_fps: 0.1
    resolution: "480p"
    dynamic_fps: true
    
  # Prompts
  prompt_id: "visual_analyzer"
  system_prompt_override: ""
  
  # Session management
  session_timeout: 300.0  # 5 minutes
  reconnect_attempts: 5
  
  # Output topics
  outputs:
    audio: "response_voice"
    transcript: "response_text"
    commands: "response_cmd"
    scene: "scene_description"
```

### 7.2 Bridge Configuration with Smart Frame Forwarding
```yaml
# bridge_dual_agent.yaml
ros_ai_bridge:
  ros__parameters:
    # Video frame rate limiting (hybrid approach)
    max_video_fps: 0.5  # Baseline frames at 0.5 fps
    
    # Smart frame forwarding configuration
    frame_forwarding:
      enabled: true
      trigger_on_voice: true
      trigger_on_text: true
      continuous_nth_frame: 5
      max_frame_age_ms: 1000
      
    # Topics configuration
    subscribed_topics:
      - topic: "prompt_voice"
        msg_type: "by_your_command/AudioDataUtterance"
      - topic: "/grunt1/arm1/cam_live/color/image_raw/compressed"
        msg_type: "sensor_msgs/CompressedImage"
```

### 7.3 Launch File Configuration
```yaml
multi_agent:
  mode: "gemini_triple"  # single|mixed|gemini_triple
  
  agents:
    - id: "conversation"
      type: "gemini"
      config_file: "gemini_conversation.yaml"
      
    - id: "command"
      type: "gemini"
      config_file: "gemini_command.yaml"
      
    - id: "visual"
      type: "gemini"
      config_file: "gemini_visual.yaml"
      
  routing:
    voice_chunks: ["conversation", "command"]
    camera_frames: ["visual"]
    
  coordination:
    visual_to_conversation: true  # Route visual descriptions to conversation agent
    command_priority: "command"  # Which agent's commands take precedence
```

## 8. Development Phases (Actual Progress)

### Phase 1: Foundation ✅ COMPLETE
- [x] ~~Create basic Pipecat pipeline structure~~ Used direct WebSocket approach
- [x] Implement WebSocket connection to Gemini Live
- [x] Implement GeminiSessionManager
- [x] Test audio-only pipeline

### Phase 2: Visual Integration ✅ COMPLETE
- [x] Implement camera frame handling
- [x] Add visual processing to Gemini agents
- [x] Create scene description outputs (JSON bounding boxes)
- [x] **Implement smart frame forwarding in ROS AI Bridge**
  - [x] Add frame caching to bridge
  - [x] Implement voice-triggered forwarding
  - [x] Configure hybrid baseline + triggered approach
- [x] Test multimodal pipeline

### Phase 3: Multi-Agent Support ✅ COMPLETE
- [x] Enhance ROS AI Bridge for multiple agents
- [x] Implement dual-agent configuration (conversation + command)
- [x] Create launch files for dual agent setup
- [x] Test concurrent agent scenarios

### Phase 4: Optimization ✅ COMPLETE
- [x] Implement smart frame forwarding (reduces latency to ~50ms)
- [x] Add transcription support
- [x] Fix fragmented response handling
- [x] Complete integration testing
- [x] Verify frame forwarding reduces latency while maintaining API efficiency

## 9. Testing Strategy

### 9.1 Unit Tests
- Individual Pipecat processor tests
- Serialization/deserialization tests
- Mock Gemini API responses

### 9.2 Integration Tests
- Pipeline end-to-end tests
- ROS topic communication tests
- Multi-agent coordination tests

### 9.3 System Tests
- Real hardware testing with camera and microphone
- Network failure recovery tests
- Performance benchmarking
- Extended conversation tests

## 10. Success Criteria

1. **Functional**: System successfully processes voice and video simultaneously
2. **Performance**: Achieves < 500ms response latency
3. **Reliability**: Maintains operation for > 1 hour continuous use
4. **Flexibility**: Supports all three deployment configurations
5. **Integration**: Seamlessly works with existing OpenAI agents

## 11. Dependencies

### 11.1 Python Packages
```txt
pipecat>=0.0.39
google-generativeai>=0.8.0
websockets>=12.0
numpy>=1.24.0
opencv-python>=4.8.0
```

### 11.2 ROS2 Packages
- audio_common_msgs
- sensor_msgs
- std_msgs
- cv_bridge

### 11.3 Hardware Requirements
- Camera: USB or CSI camera with ROS2 driver
- Microphone: Already configured for OpenAI agent
- GPU: Recommended for image processing (optional)

## 12. Future Enhancements

### Near-term Optimizations
1. **Audio Buffering at Utterance Start**: Buffer initial audio chunks (100-200ms) while waiting for fresh frames to completely eliminate race conditions
2. **Triple Agent Configuration**: Add dedicated visual analysis agent alongside conversation and command agents
3. **Dynamic Frame Rate Adjustment**: Automatically adjust FPS based on scene complexity and movement

### Long-term Features
1. **Depth Integration**: Use depth data for 3D scene understanding
2. **SLAM Integration**: Correlate visual features with robot location
3. **Object Tracking**: Maintain persistent object identities across frames
4. **Gesture Control**: Implement gesture-based robot commands
5. **Multi-Camera**: Support multiple camera viewpoints
6. **Local Models**: Option to run Gemini models locally
7. **Point Cloud Processing**: Direct 3D point cloud analysis

## Appendix A: Pipecat Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Pipecat Pipeline                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐       │
│  │   ROS    │   │   ROS    │   │              │       │
│  │  Voice   │──▶│  Camera  │──▶│   Gemini     │       │
│  │  Input   │   │  Input   │   │   Live       │       │
│  └──────────┘   └──────────┘   │   Bridge     │       │
│                                 └──────┬───────┘       │
│                                        │                │
│                                        ▼                │
│                              ┌──────────────┐          │
│                              │   Response   │          │
│                              │    Router    │          │
│                              └──────┬───────┘          │
│                                     │                  │
│            ┌────────────────────────┼─────────┐        │
│            ▼                        ▼         ▼        │
│     ┌──────────┐           ┌──────────┐  ┌────────┐   │
│     │   ROS    │           │   ROS    │  │  ROS   │   │
│     │  Audio   │           │Transcript│  │Command │   │
│     │  Output  │           │  Output  │  │ Output │   │
│     └──────────┘           └──────────┘  └────────┘   │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## Appendix B: Message Format Examples

### B.1 Visual Analysis Output
```json
{
  "timestamp": 1736819200.123,
  "frame_id": 12345,
  "objects": [
    {
      "label": "person",
      "confidence": 0.95,
      "bbox": [100, 200, 300, 400],
      "attributes": {
        "pose": "standing",
        "facing": "towards_camera",
        "distance_m": 2.5
      }
    },
    {
      "label": "chair",
      "confidence": 0.87,
      "bbox": [400, 300, 200, 250],
      "spatial_relation": "left_of_person"
    }
  ],
  "scene_description": "A person is standing in front of a chair in what appears to be a living room",
  "navigation_hints": {
    "obstacles": ["chair on left"],
    "clear_path": "forward and right"
  }
}
```

### B.2 Command Extraction
```json
{
  "timestamp": 1736819200.456,
  "source": "multimodal",
  "command": "move",
  "parameters": {
    "target": "chair",
    "visual_reference": {
      "object_id": 1,
      "bbox": [400, 300, 200, 250]
    },
    "instruction": "go to the chair on your left"
  },
  "confidence": 0.92
}
```