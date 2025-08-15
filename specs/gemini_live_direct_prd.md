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

## Key Requirements

### 1. Core Architecture
- **Direct WebSocket Connection**: Use native Gemini Live API via WebSocket
- **Frame-based Processing**: Similar to `oai_realtime` but adapted for Gemini protocol
- **Multi-Agent Support**: Separate conversation and command extraction processes
- **ROS Integration**: WebSocket bridge to ROS AI Bridge for audio/text/image I/O

### 2. Connection & Session Management
Per Gemini Live documentation:
- **10-minute connection limit**: Auto-reconnect before limit
- **15-minute session limit** (text/voice): Track session duration  
- **2-minute limit with video**: Special handling for video sessions
- **Graceful reconnection**: Preserve context across connection drops
- **Content reinjection**: Rebuild conversation state after reconnection

### 3. User Experience Features
- **User Interruptions**: Support cutting off assistant mid-response
- **Proactive Audio**: Model decides when it's being addressed (no wake words)
- **Low Latency**: Direct streaming without intermediate abstraction layers
- **Audio Echo Prevention**: Proper source tracking to avoid feedback loops

### 4. Multimodal Support
- **Audio**: PCM 16-bit mono 16kHz streaming
- **Text**: Bidirectional text messaging
- **Images**: Base64-encoded image frames from ROS
- **Video**: Optional frame-by-frame processing with time limits

## Technical Architecture

### Component Structure
```
agents/gemini_live_direct/
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
- Track connection duration (10 min limit)
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

## Implementation Plan

### Phase 1: Core Infrastructure (Day 1)
1. Create basic `GeminiLiveDirectAgent` class
2. Implement `GeminiWebSocketClient` for API connection
3. Port essential components from `oai_realtime`:
   - WebSocketBridgeInterface integration
   - Basic message routing
   - Configuration loading

### Phase 2: Protocol Implementation (Day 2)
1. Create `GeminiSerializer` for message format conversion
2. Implement bidirectional audio streaming
3. Add text message handling
4. Basic conversation flow without advanced features

### Phase 3: Session Management (Day 3)
1. Create `GeminiSessionManager` with time limit tracking
2. Implement connection lifecycle management
3. Add interruption support and proper state handling
4. Context preservation across reconnections

### Phase 4: Multimodal & Production (Day 4)
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