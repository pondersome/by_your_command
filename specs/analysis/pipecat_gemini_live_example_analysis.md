# Pipecat Gemini Live Example Analysis

## Executive Summary

The Word Wrangler example demonstrates Pipecat 0.0.80's implementation of the GeminiMultimodalLiveLLMService for building real-time voice-based games. This analysis examines the pipeline architecture, integration patterns, and how these can be applied to our ROS2-based multimodal robot control system.

## Architecture Overview

### System Components

1. **Web-based Game (Simple Pipeline)**
   - Single Gemini Live instance playing the guesser role
   - RTVIProcessor for client-server communication
   - User Context Aggregator for conversation persistence
   - Daily WebRTC transport (not needed for our use case)

2. **Phone-based Game (Complex Parallel Pipeline)**
   - Dual AI architecture: Host AI + Player AI
   - Parallel pipeline branches with audio routing
   - Producer/Consumer pattern for inter-branch communication
   - Notifier-based synchronization between branches

## Key Implementation Patterns

### 1. Pipeline Construction (bot.py)

```python
pipeline = Pipeline([
    transport.input(),
    rtvi,
    stt_mute_filter,
    context_aggregator.user(),
    llm,
    transport.output(),
    context_aggregator.assistant(),
])
```

**Key Insights:**
- Linear pipeline for simple use cases
- RTVI processor enables client-server events
- Context aggregators track conversation history
- STT mute filter prevents interruptions during bot intro

### 2. GeminiMultimodalLiveLLMService Usage

```python
llm = GeminiMultimodalLiveLLMService(
    api_key=os.getenv("GOOGLE_API_KEY"),
    transcribe_user_audio=True,  # Built-in transcription
    system_instruction=system_instruction,
)
```

**Important Features:**
- Built-in speech-to-text via `transcribe_user_audio=True`
- No separate STT service required
- Direct audio input processing
- System instructions for behavior control

### 3. Parallel Pipeline Architecture (bot_phone_local.py)

```python
ParallelPipeline(
    # Host branch
    [
        consumer,  # Receives from player
        host_llm,
        game_state_tracker,
        host_tts,
        bot_stopped_speaking_detector,
    ],
    # Player branch
    [
        start_frame_gate,  # Waits for host
        player_llm,
        producer,  # Sends to host
    ],
)
```

**Architecture Patterns:**
- Producer/Consumer for cross-branch communication
- Notifier-based synchronization
- State tracking processors
- Frame filtering and transformation

### 4. User Context Aggregation

```python
context = OpenAILLMContext(messages)
context_aggregator = llm.create_context_aggregator(context)

# In pipeline:
context_aggregator.user(),  # Before LLM
llm,
context_aggregator.assistant(),  # After LLM
```

**Purpose:**
- Maintains conversation history
- Enables multi-turn interactions
- Compatible with OpenAI context format
- Could persist across sessions

## Critical Discoveries

### 1. No Video Support in Example

The example uses only audio modalities:
- `GeminiMultimodalModalities.TEXT` for text-only host
- Voice input/output for player
- **No vision/video pipeline demonstrated**

### 2. Frame Processing Model

Pipecat 0.0.80 uses typed frames:
- `StartFrame` - Pipeline initialization
- `AudioRawFrame` - Raw audio data
- `TextFrame` - Text messages
- `ImageRawFrame` - Image data (not used in example)
- `TTSAudioRawFrame` - TTS output
- `LLMTextFrame` - LLM text output

### 3. Synchronization Mechanisms

```python
# Notifier pattern for event coordination
bot_speaking_notifier = EventNotifier()
new_word_notifier = EventNotifier()

# Processors that use notifiers
BotStoppedSpeakingNotifier(bot_speaking_notifier)
StartFrameGate(bot_speaking_notifier)
GameStateTracker(new_word_notifier)
```

## Application to Our Requirements (Updated)

### 1. Multi-Agent Architecture (Following OpenAI Pattern)

Based on our successful OpenAI dual-agent implementation, we'll use **separate processes** for each agent type:

```python
# Launch file pattern - multiple independent agents
gemini_conversation = ExecuteProcess(
    cmd=['gemini_live_agent', '--config', 'gemini_conversation.yaml'])
    
gemini_command = ExecuteProcess(
    cmd=['gemini_live_agent', '--config', 'gemini_command.yaml'])

gemini_scene = ExecuteProcess(  # Future enhancement
    cmd=['gemini_live_agent', '--config', 'gemini_scene.yaml'])
```

**Rationale:**
- Proven pattern from OpenAI implementation
- Complete isolation between agents
- Independent failure/restart capability
- Simpler debugging with separate logs
- Easy to scale (add/remove agents)

### 2. Pipeline Structure Per Agent

Each agent will have its own simple pipeline:

```python
# Conversation agent pipeline
pipeline = Pipeline([
    ros_bridge_input(),      # WebSocket from ROS
    frame_throttler,         # For video frames (1 fps)
    gemini_conversation_llm, # With multimodal support
    ros_bridge_output(),     # Back to ROS
])

# Command agent pipeline (simpler, no video)
pipeline = Pipeline([
    ros_bridge_input(),     # Audio only
    gemini_command_llm,     # Text-only output
    command_parser,         # Extract JSON commands
    ros_bridge_output(),    # Command messages to ROS
])
```

### 3. Context Management Strategy

**Keep existing approach:** Our current context reinjection system is more sophisticated than Pipecat's aggregators:

```python
# Skip Pipecat aggregators, use our existing system
pipeline = Pipeline([
    ros_bridge_input(),
    gemini_llm,  # Direct to LLM, no aggregator
    ros_bridge_output(),
])

# Inject context manually when needed
if self.prepared_context:
    await task.queue_frames([
        TextFrame(text=self.prepared_context.to_prompt())
    ])
```

**Rationale:**
- Pipecat aggregators are session-only
- We need cross-session persistence
- We need dynamic system prompt changes
- Our system already handles context blending

### 4. VAD Configuration

**Disable Pipecat VAD** since ROS already handles it:

```python
# No double VAD processing
class GeminiLiveAgent:
    def create_transport(self):
        return WebSocketTransport(
            params=WebSocketParams(
                vad_analyzer=None,  # Disabled - ROS handles VAD
                audio_in_enabled=True,
            )
        )
```

### 5. Frame Conversion Layer

```python
# ROS to Pipecat converter
class ROSFrameConverter:
    def convert_to_pipecat(self, msg_envelope):
        if msg_envelope.ros_msg_type == "by_your_command/AudioDataUtterance":
            return AudioRawFrame(
                audio=bytes(msg_envelope.raw_data.int16_data),
                sample_rate=16000,
                num_channels=1
            )
        elif msg_envelope.ros_msg_type == "sensor_msgs/Image":
            return ImageRawFrame(
                image=self.cv_bridge.imgmsg_to_cv2(msg_envelope.raw_data),
                size=(msg.width, msg.height),
                format="BGR"
            )
        elif msg_envelope.ros_msg_type == "std_msgs/String":
            return TextFrame(text=msg_envelope.raw_data.data)
    
    def convert_from_pipecat(self, frame):
        if isinstance(frame, AudioRawFrame):
            return {
                'topic': 'audio_out',
                'msg_type': 'audio_common_msgs/AudioData',
                'data': {'data': frame.audio}
            }
        elif isinstance(frame, TextFrame):
            return {
                'topic': 'llm_transcript',
                'msg_type': 'std_msgs/String', 
                'data': {'data': frame.text}
            }
```

### 6. Video Support Implementation

Since the example lacks video, custom implementation required:

```python
class VideoFrameThrottler(FrameProcessor):
    """Throttle video frames to control costs"""
    def __init__(self, fps=1.0, dynamic=True):
        self.min_interval = 1.0 / fps
        self.last_frame_time = 0
        self.dynamic = dynamic
        
    async def process_frame(self, frame: Frame, direction):
        if isinstance(frame, ImageRawFrame):
            now = time.time()
            if now - self.last_frame_time >= self.min_interval:
                await self.push_frame(frame, direction)
                self.last_frame_time = now
                
                # Dynamic FPS adjustment based on scene
                if self.dynamic:
                    self.adjust_fps_for_scene(frame)
        else:
            # Pass non-image frames through
            await self.push_frame(frame, direction)
```

### 7. Configuration Files (Following OpenAI Pattern)

```yaml
# gemini_conversation_agent.yaml
gemini_live_agent:
  agent_id: "gemini_conversation"
  bridge_connection:
    host: "localhost"
    port: 8765
  model: "gemini-2.0-flash-exp"
  prompt_id: "barney_conversational_gemini"
  modalities: ["audio", "text", "image"]
  video_fps: 1.0
  audio_out_topic: "audio_out"
  transcript_topic: "llm_transcript"

# gemini_command_agent.yaml
gemini_live_agent:
  agent_id: "gemini_command"
  bridge_connection:
    host: "localhost"
    port: 8765
  model: "gemini-2.0-flash-exp"
  prompt_id: "barney_command_extractor_gemini"
  modalities: ["audio", "text"]  # No video for commands
  audio_out_topic: ""  # No audio output
  transcript_topic: "command_transcript"
```

## Key Learnings

### 1. Pipecat Abstracts the API Complexity
- No need to implement WebSocket protocols
- No manual audio streaming
- Built-in VAD and turn detection
- Automatic reconnection handling

### 2. Frame-Based Architecture Benefits
- Clear data flow
- Type safety with frame classes
- Easy to add processors
- Composable pipeline stages

### 3. Missing Pieces for Our Use Case
- Video frame support (needs custom implementation)
- ROS bridge frame conversion
- Command extraction from multimodal context
- Scene description generation

### 4. Deployment Considerations
- The example uses subprocess spawning (not needed for us)
- We'll run as a ROS node directly
- No need for FastAPI server or Daily rooms
- Direct WebSocket connection to our bridge

## Recommended Implementation Path (Revised)

### Phase 1: Single Audio-Only Agent
1. Create basic `gemini_live_agent.py` with Pipecat pipeline
2. Connect to ROS bridge via existing WebSocketBridgeInterface
3. Process audio only (voice commands) with single agent
4. Test conversation and command understanding

### Phase 2: Dual Agent Setup (Following OpenAI)
1. Split into conversation and command agents
2. Create separate YAML configs for each agent
3. Update launch file to start both processes
4. Test simultaneous operation with shared bridge

### Phase 3: Add Video Support
1. Implement ImageRawFrame conversion from sensor_msgs/Image
2. Add VideoFrameThrottler processor (1 fps default)
3. Enable video only for conversation agent
4. Test multimodal understanding

### Phase 4: Production Features
1. Port existing context management system
2. Add scene description agent (third process)
3. Implement dynamic FPS adjustment
4. Performance optimization and monitoring

## Critical Implementation Notes

### 1. Pipecat Version Compatibility
- Example uses 0.0.80
- Our current version: 0.0.80 ✓
- GeminiMultimodalLiveLLMService requires `pipecat-ai[google]`

### 2. Dependencies
```python
# Required packages
pipecat-ai[google,silero]  # Core + Google + VAD
google-genai  # For direct API access if needed
```

### 3. Environment Variables
```bash
GEMINI_API_KEY=your_key_here
# No DAILY_API_KEY needed
# No room URLs needed
```

### 4. Testing Strategy
1. Start with example's simple pipeline
2. Replace Daily transport with ROS bridge
3. Add video frames incrementally
4. Test command extraction separately

## Conclusion (Updated with Architecture Decisions)

The Word Wrangler example provides excellent patterns for:
- Pipeline construction and frame processing
- GeminiMultimodalLiveLLMService usage
- LLM abstraction benefits
- State management within a single agent

However, based on our OpenAI experience, we'll diverge from the example in key areas:

### What We'll Use from the Example:
1. **Simple pipeline pattern** from bot.py (not parallel pipelines)
2. **GeminiMultimodalLiveLLMService** for API abstraction
3. **Frame-based architecture** for data flow
4. **Basic frame processors** as templates

### What We'll Do Differently:
1. **Multiple agent processes** instead of single process with branches
2. **Keep our context management** instead of Pipecat aggregators
3. **Disable Pipecat VAD** (ROS already handles it)
4. **Custom video frame handling** (not in example)
5. **WebSocket bridge** instead of Daily transport

### Architecture Summary:
- **Multi-agent**: Separate processes per agent type (conversation, command, scene)
- **Configuration**: Individual YAML files per agent (like OpenAI)
- **Launch**: Single launch file orchestrating multiple agents
- **Bridge**: Shared ROS AI Bridge with WebSocket server
- **Pipeline**: Simple linear pipelines per agent
- **Context**: Our existing cross-session context system
- **VAD**: Handled by ROS, disabled in Pipecat

The key insight remains: Pipecat abstracts the Gemini Live API complexity, but we'll integrate it using our proven multi-agent patterns rather than adopting all of Pipecat's architectural choices. This gives us the best of both worlds - Pipecat's API simplification with our battle-tested system architecture.