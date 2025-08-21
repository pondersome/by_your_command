# PRD: Gemini Live Direct Agent

**Created:** 2025-08-15  
**Status:** Draft  
**Scope:** Direct Gemini Live API implementation without Pipecat

## Executive Summary

Create a `gemini_live_direct` agent that implements Google's Gemini Live API directly, bypassing Pipecat to achieve better integration with our existing ROS infrastructure and multi-agent architecture.

## Background & Motivation

The Pipecat-based implementation revealed fundamental architectural mismatches:
- Pipecat expects to own the entire audio pipeline (VAD, chunking, STT)
- We already have external VAD and context management in ROS
- Pipecat's opinionated design conflicts with our frame-based processing
- Direct API provides full control over timing, interruptions, and session management

## API References and Examples

  1. "Get started with Live API | Gemini API | Google AI for Developers"
    - https://ai.google.dev/gemini-api/docs/live
  2. "Live API reference | Generative AI on Vertex AI | Google Cloud"
    - https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-live
  3. "Live API | Generative AI on Vertex AI | Google Cloud"
    - https://cloud.google.com/vertex-ai/generative-ai/docs/live-api
  4. "Live API capabilities guide | Gemini API | Google AI for Developers"
    - https://ai.google.dev/gemini-api/docs/live-guide
  5. "How to Use the Google DeepMind Gemini 2.0 Flash Live API for Real-Time Conversations" (Medium article by azhar)
    - https://medium.com/ai-insights-cobet/how-to-use-the-google-deepmind-gemini-2-0-flash-live-api-for-real-time-conversations-89369f1b9bcb
  6. "Gemini Image Understanding"
    - https://ai.google.dev/gemini-api/docs/image-understanding

The most relevant documentation appears to be from the official Google AI for Developers site (links 1 and 4) which show the proper Python implementation for the Live API.

## Model Selection

- Native audio: This option provides the most natural and realistic-sounding speech and better multilingual performance. It also enables advanced features like affective (emotion-aware) dialogue, proactive audio (where the model can decide to ignore or respond to certain inputs), and "thinking". Native audio is supported by the following native audio models:
  - gemini-2.5-flash-preview-native-audio-dialog
  - gemini-2.5-flash-exp-native-audio-thinking-dialog
- Half-cascade audio: This option uses a cascaded model architecture (native audio input and text-to-speech output). It offers better performance and reliability in production environments, especially with tool use. These models do not support proactive audio. Half-cascaded audio is supported by the following models:
  - gemini-live-2.5-flash-preview
  - gemini-2.0-flash-live-001

There is no documented variation in how the models support video input.

## Key Requirements

### 1. Core Architecture
- **NO ALTERATION OF ROS COMPONENTS**: Use existing ROS AI Bridge for audio/text/image I/O. Any setup for other ROS components should be done identically to the `oai_realtime` agent. So config and launch files should be the same for ROS components including the ros_ai_bridge.
- **Direct WebSocket Connection**: Use native python Gemini Live API via WebSocket
- **Frame-based Processing**: Similar to `oai_realtime` but adapted for Gemini protocol
- **Multi-Agent Support**: Separate conversation and command extraction processes
- **ROS Integration**: WebSocket bridge to ROS AI Bridge for audio/text/image I/O

### 2. Connection & Session Management
Per Gemini Live documentation:
- **10-minute connection limit**: Auto-reconnect before limit
- **15-minute session limit** (text/voice): Track session duration  
- **2-minute limit with video**: Special handling for video sessions (this should be the default session time limit)
- **Graceful reconnection**: Preserve context across connection drops
- **Content reinjection**: Rebuild conversation state after reconnection

### 3. User Experience Features
- **User Interruptions**: Support cutting off assistant mid-response
- **Proactive Audio**: Model decides when it's being addressed (no wake words) - this only applies to the native audio models
- **Low Latency**: Direct streaming without intermediate abstraction layers
- **Audio Echo Prevention**: Proper source tracking to avoid feedback loops

### 4. Multimodal Support
- **Audio**: PCM 16-bit mono 16kHz streaming
- **Text**: Bidirectional text messaging
- **Images**: Base64-encoded image frames from ROS
- **Video**: Optional frame-by-frame processing with specified frame rates

## Technical Architecture

### Component Structure
```
agents/gemini_live/
├── gemini_live_direct_agent.py      # Main agent class
├── gemini_session_manager.py        # Connection/session management
├── gemini_serializer.py             # Protocol serialization
├── gemini_websocket_client.py       # Low-level WebSocket handling
├── main.py                          # Entry point with arg parsing
└── __init__.py
```

### Key Components

#### 1. GeminiLiveDirectAgent
Main agent class following `oai_realtime` pattern:
- WebSocket connection to Gemini Live API
- Integration with ROS AI Bridge via WebSocketBridgeInterface
- Frame processing for audio/text/image data
- Session state management and reconnection logic

#### 2. GeminiSessionManager
Handles Gemini-specific session constraints:
- Track connection duration (2 min limit)
- Track session duration (15 min audio/text, 2 min video)
- Implement proactive reconnection with exponential backoff
- Context preservation and reinjection after reconnection
- Different from OpenAI's pricing-optimized SessionManager

#### 3. GeminiSerializer
Protocol adaptation layer:
- Convert ROS messages to Gemini Live format
- Handle `bidiGenerateContent` request/response serialization
- Audio format conversion (ROS AudioData → PCM chunks)
- Image/video frame encoding for multimodal sessions

#### 4. GeminiWebSocketClient
Low-level WebSocket management:
- Async WebSocket connection to `generativelanguage.googleapis.com`
- Authentication with Google API key
- Message framing and protocol handling
- Connection health monitoring and recovery

### Integration Points

#### ROS AI Bridge Integration
- **Inbound**: Audio chunks, text messages, image frames
- **Outbound**: Audio responses, transcripts, command detection
- **Zero-copy**: Direct frame passing without unnecessary serialization

#### Prompt Management
- Reuse existing `PromptLoader` system
- Support for Gemini-specific prompt formats
- Runtime prompt switching capability

#### Context Management
- Port `ConversationContext` from `oai_realtime`
- Adapt for Gemini's conversation threading
- Context summarization for reconnection scenarios

## Implementation Plan - LIKELY OUTDATED

### Phase 1: Core Infrastructure
1. Create basic `GeminiLiveDirectAgent` class
2. Implement `GeminiWebSocketClient` for API connection
3. Port essential components from `oai_realtime`:
   - WebSocketBridgeInterface integration
   - Basic message routing
   - Configuration loading

### Phase 2: Protocol Implementation
1. Create `GeminiSerializer` for message format conversion
2. Implement bidirectional audio streaming
3. Add text message handling
4. Basic conversation flow without advanced features

### Phase 3: Session Management
1. Create `GeminiSessionManager` with time limit tracking
2. Implement connection lifecycle management
3. Add interruption support and proper state handling
4. Context preservation across reconnections

### Phase 4: Multimodal & Production
1. Add image/video frame processing
2. Implement proactive audio configuration
3. Error handling and recovery mechanisms
4. Testing and configuration files

## API Protocol Details

### Gemini Live WebSocket Endpoint
```
wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService/BidiGenerateContent
```

### Message Format
Based on `bidiGenerateContent` protocol:
```json
{
  "setup": {
    "model": "models/gemini-2.0-flash-live-001",
    "generationConfig": {...},
    "systemInstruction": {...}
  },
  "clientContent": {
    "realtimeInput": {
      "mediaChunks": [...],
      "turnComplete": boolean
    }
  }
}
```

### Key Configuration
- **Model**: `models/gemini-2.0-flash-live-001`
- **Audio Format**: PCM 16-bit mono 16kHz
- **Response Modalities**: Audio and text
- **Tools**: Support for function calling

## Configuration Files

### Conversation Agent Config
```yaml
# config/gemini_direct_conversation.yaml
model_config:
  model: "models/gemini-2.0-flash-live-001"
  response_modalities: ["AUDIO", "TEXT"]
  speech_config:
    voice_config:
      prebuilt_voice_config:
        voice_name: "Puck"

session_config:
  connection_timeout: 600  # 10 minutes
  session_timeout: 900     # 15 minutes
  video_timeout: 120       # 2 minutes
  
prompt_config:
  prompt_id: "barney_conversational_gemini"
```

### Launch Configuration
```python
# launch/gemini_direct_single.launch.py
def generate_launch_description():
    return LaunchDescription([
        Node(
            package='by_your_command',
            executable='gemini_live_direct_agent',
            parameters=[{'config': 'gemini_direct_conversation.yaml'}],
            arguments=['--agent-type', 'conversation']
        )
    ])
```

## Success Criteria

### Performance Metrics
- **Audio Latency**: < 500ms end-to-end (ROS → Gemini → ROS)
- **Connection Stability**: Handle 10+ minute conversations reliably
- **Interruption Response**: < 200ms to stop assistant and clear audio
- **Reconnection Time**: < 2 seconds with context preservation

### Functional Requirements
- ✅ Zero audio echo/feedback issues
- ✅ Successful user interruptions during assistant speech
- ✅ Proper context preservation across reconnections
- ✅ Multimodal support (audio + text + images)
- ✅ Integration with existing ROS AI Bridge
- ✅ Support for both conversation and command extraction agents

### Quality Metrics
- **Code Maintainability**: Clear separation of concerns, reusable components
- **Debugging**: Comprehensive logging and optional debug interface
- **Configuration**: YAML-based config with runtime parameter updates
- **Error Handling**: Graceful degradation and recovery from failures

## Risk Mitigation

### Technical Risks
1. **API Rate Limits**: Implement exponential backoff and connection pooling
2. **Audio Format Issues**: Comprehensive testing of PCM conversion pipeline
3. **Session Management**: Proactive monitoring to prevent hard disconnections
4. **Context Loss**: Robust serialization and reinjection mechanisms

### Integration Risks
1. **ROS Compatibility**: Thorough testing with existing VAD and audio systems
2. **Multi-Agent Communication**: Clear message routing and namespace isolation
3. **Configuration Drift**: Version-controlled configs with validation

## Gemini Live API Usage Guide - Empirical Findings

### Overview
This section documents empirical findings from extensive testing of the Gemini Live API, revealing undocumented behaviors, edge cases, and practical usage patterns not covered in the official documentation.

### Model Names and Connection

#### Model Name Format
- **MUST use `models/` prefix**: The model name requires the full path prefix
  - ✅ Correct: `models/gemini-2.0-flash-live-001`
  - ❌ Wrong: `gemini-2.0-flash-live-001`
- The API will connect but hang indefinitely without the prefix

#### Tested Working Models
- `models/gemini-2.0-flash-live-001` - Half-cascade, production-ready
- `gemini-live-2.5-flash-preview` - Works but may have different quotas
- `gemini-2.5-flash-preview-native-audio-dialog` - Native audio with voice selection

### Audio Input/Output

#### Audio Input Format
```python
await session.send_realtime_input(
    audio=types.Blob(
        data=audio_bytes,  # Raw PCM bytes
        mime_type="audio/pcm;rate=16000"
    )
)
```
- **Format**: 16-bit PCM, mono, 16kHz
- **No explicit end-of-turn needed**: Gemini auto-detects speech end
- **WARNING**: Do NOT send `turn_complete` with audio - causes "invalid argument" errors

#### Audio Output Characteristics
- **Format**: 24kHz PCM16 mono (1.5x input sample rate)
- **Chunk sizes**: Typically 9600-11520 bytes per response
- **Streaming pattern**: Audio arrives in many small chunks
  - Short response (~2s): 8-10 chunks
  - Detailed description (~22s): 90-100 chunks
- **Access via**: `response.data` contains raw PCM bytes

#### Audio Completion Signals
Gemini sends specific metadata after audio completes:
1. First: `response.server_content.generation_complete = True`
2. Second: `response.server_content.turn_complete = True` (includes usage_metadata)

This is more reliable than `StopAsyncIteration` which may come much later.

### Text Input/Output

#### Sending Text
```python
await session.send_client_content(
    turns=types.Content(
        parts=[types.Part(text="Your message here")]
    ),
    turn_complete=True  # Required for text
)
```
- **Note**: `send_client_content()` takes keyword arguments, not positional
- **Must include**: `turn_complete=True` for text (unlike audio)

#### Receiving Text
- Access via `response.text`
- May arrive in multiple chunks for longer responses
- Text responses trigger the same completion signals as audio

### Image Input

#### Sending Images with Prompts
```python
content = types.Content(
    parts=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/png'  # or 'image/jpeg'
        ),
        types.Part(text="Describe this image")
    ]
)
await session.send_client_content(turns=content, turn_complete=True)
```

#### Image Support Notes
- **Supported formats**: PNG, JPEG, WEBP, HEIC, HEIF
- **Works with both modalities**: Can request TEXT or AUDIO descriptions
- **No special configuration needed**: Same models support images
- **Response quality**: Detailed and accurate scene descriptions

### Object Detection

#### Requesting Object Detection
```python
prompt = """Detect all objects in this image and return a JSON array.
For each object provide:
- "label": the object name
- "box_2d": [ymin, xmin, ymax, xmax] normalized to 0-1000
- "confidence": detection confidence from 0.0 to 1.0

Return ONLY the JSON array, no other text."""

content = types.Content(
    parts=[
        types.Part.from_bytes(data=image_bytes, mime_type='image/png'),
        types.Part(text=prompt)
    ]
)
```

#### Coordinate System
- **Origin**: (0,0) is at **top-left corner** (verified through testing)
- **Format**: `[ymin, xmin, ymax, xmax]` (y-first, then x)
- **Scale**: Normalized to 0-1000 (not 0-1 as in some systems)
- **Conversion to pixels**:
  ```python
  x1_px = int(box[1] * image_width / 1000)
  y1_px = int(box[0] * image_height / 1000)
  x2_px = int(box[3] * image_width / 1000)
  y2_px = int(box[2] * image_height / 1000)
  ```

#### JSON Response Handling
- **Warning**: Gemini sometimes returns malformed JSON with syntax errors
- **Solution**: Use robust parsing with regex fallback
- **Confidence scores**: May return 0.0 if not explicitly requested in prompt

#### Bridge Output Format
Recommended format for ROS bridge consumption:
```json
{
  "label": "object_name",
  "confidence": 0.95,
  "bbox_normalized": [ymin, xmin, ymax, xmax],  // 0-1000 scale
  "bbox_pixels": {
    "x": x_pixels,
    "y": y_pixels,
    "width": width_pixels,
    "height": height_pixels,
    "center_x": center_x,
    "center_y": center_y
  }
}
```

### Video Streaming

#### Inline Video Frames
- **Both formats work**: Raw bytes or base64-decoded bytes
- **Same as images**: Use `Part.from_bytes()` for each frame
- **Frame rate control**: Implement throttling on sender side

#### Detection vs Description
Different prompts for different agent roles:
- **Command Agent**: Request structured JSON with specific objects
  ```python
  "List objects with locations as JSON: [{'label': 'name', 'box_2d': [...]}]"
  ```
- **Conversational Agent**: Request natural language
  ```python
  "What's happening in this scene?"
  ```

### Response Stream Patterns

#### Typical Response Flow
1. Content chunks (audio/text)
2. `generation_complete=True` signal
3. `turn_complete=True` with usage metadata
4. (Optional) Additional empty responses before `StopAsyncIteration`

#### Mixed Response Warnings
When requesting specific modalities, you may see warnings:
- `"Warning: there are non-text parts in the response: ['inline_data']"`
- `"Warning: there are non-data parts in the response: ['text']"`

These are normal and indicate the response contains multiple content types.

### Session Management

#### Connection Lifecycle
```python
async with client.aio.live.connect(model=model, config=config) as session:
    # Session is immediately ready - no need to wait for "session.created"
    # Unlike OpenAI, Gemini sessions are ready immediately after connection
```

#### Session Configuration
```python
config = {
    "response_modalities": ["AUDIO"],  # or ["TEXT"] or ["AUDIO", "TEXT"]
    "system_instruction": "Your system prompt here"
}
```
- Simplified config structure works better than nested generation_config
- Voice selection appears to be automatic for the model

### Common Pitfalls and Solutions

#### Issue: Connection Hangs
- **Cause**: Missing `models/` prefix in model name
- **Solution**: Always use full model path

#### Issue: "Invalid argument" Error with Audio
- **Cause**: Sending `turn_complete=True` after audio input
- **Solution**: Only use `turn_complete` with text, not audio

#### Issue: Incomplete Audio Responses
- **Cause**: Stopping reception too early
- **Solution**: Continue receiving until `turn_complete=True` signal

#### Issue: No Responses Received
- **Cause**: Using deprecated `send()` method
- **Solution**: Use `send_client_content()` for text, `send_realtime_input()` for audio

### Performance Considerations

#### Audio Streaming Latency
- First audio chunk typically arrives within 200-500ms
- Complete responses require patience - don't timeout too quickly
- 22-second description arrived in 97 chunks over ~25 seconds real-time

#### Token Usage
- Available in `response.usage_metadata` after `turn_complete`
- Includes breakdown by modality (TEXT, AUDIO, IMAGE tokens)

### API Method Reference

#### Available Session Methods
- `send_client_content(turns=..., turn_complete=True)` - Send text/images
- `send_realtime_input(audio=...)` - Send audio
- `send_tool_response(...)` - Send tool/function responses
- `receive()` - Async generator for responses
- `close()` - Close session
- `start_stream(stream=..., mime_type=...)` - For continuous streams (requires parameters)

#### Deprecated Methods
- `send()` - Deprecated, will be removed Q3 2025
- Use specific methods above instead

### Testing Recommendations

1. **Start with text**: Verify connection and responses work
2. **Test audio in isolation**: Audio I/O has different patterns than text
3. **Be patient with audio**: Full responses take many chunks
4. **Monitor completion signals**: More reliable than timeouts
5. **Handle warnings gracefully**: Mixed modality warnings are normal

## Future Enhancements

### Phase 2 Features (Future)
- **Video Streaming**: Full video pipeline with frame throttling
- **Advanced Tools**: MCP server integration for extended capabilities
- **Performance Optimization**: Connection pooling and audio buffering
- **Monitoring**: Comprehensive metrics and health checks

### Common Module Refactoring
- Extract shared utilities to `agents/common/`
- Unified WebSocket bridge interface
- Shared session management patterns
- Common testing utilities

## Dependencies

### Core Dependencies
- `google-ai-generativelanguage`: Official Gemini SDK
- `websockets`: Async WebSocket client
- `pydantic`: Configuration validation
- `PyYAML`: Configuration file parsing

### ROS Dependencies  
- `rclpy`: ROS2 Python client
- `std_msgs`: Standard message types
- `audio_common_msgs`: Audio data messages
- `sensor_msgs`: Image message types

### Development Dependencies
- `pytest`: Unit testing framework
- `pytest-asyncio`: Async test support
- `black`: Code formatting
- `mypy`: Type checking

---

**Next Steps**: Begin Phase 1 implementation with core infrastructure and basic WebSocket connectivity to Gemini Live API.