# Multi-Agent Architecture PRD

**Package**: by_your_command  
**Subsystem**: agents  
**Version**: 1.0  
**Last Updated**: July 2025

## Overview

The multi-agent architecture provides a scalable framework for integrating multiple LLM providers with ROS2 robotic systems. It serves as the intelligent decision-making layer that processes speech, vision, and sensor data to generate natural language responses and robot commands through a unified async/ROS bridge interface.

## Core Objectives

### Primary Goals
1. **Multi-Provider Support**: Enable seamless integration of multiple LLM APIs (OpenAI, Google, Anthropic, etc.)
2. **Async/ROS Bridge**: Decouple asyncio-based LLM sessions from ROS2's callback-based concurrency
3. **Session Management**: Intelligent lifecycle management with cost optimization and failover
4. **Unified Interface**: Consistent API across different LLM providers for interchangeable usage
5. **Orchestration Ready**: Foundation for LangGraph-based multi-agent workflows

### Performance Requirements
- **Session Latency**: < 500ms for new LLM connections
- **Response Latency**: < 200ms from LLM response to ROS2 action
- **Concurrent Sessions**: Support 2-3 active LLM providers simultaneously
- **Memory Efficiency**: < 200MB per active agent session
- **Cost Optimization**: Automatic session cycling and context management

## System Architecture

### Directory Structure
```
agents/
├── __init__.py                    # Multi-provider imports and version management
├── graph.py                       # LangGraph orchestration (placeholder)
├── oai_realtime/                  # OpenAI Realtime API agent (active)
│   ├── __init__.py
│   ├── oai_realtime_agent.py     # Main agent implementation
│   ├── session_manager.py        # Session lifecycle management  
│   ├── pause_detector.py         # Speech boundary detection
│   ├── serializers.py            # Message format handling
│   ├── context.py                # Conversation memory
│   ├── main.py                   # Standalone executable
│   └── oai_realtime.py          # Core API client
├── gemini_live/                   # Google Gemini Live agent (placeholder)
│   └── __init__.py               # Future: GeminiLiveAgent
└── tools/                         # Shared utilities
    └── command_scraper.py        # ROS command processing
```

### Data Flow Architecture
```
ROS2 Topics → [ros_ai_bridge] → [Agent Queues] → [LLM APIs] → [Response Processing] → ROS2 Actions
     ↑                 ↓              ↓             ↓              ↓                    ↓
Audio Capture    Message Queue   Session Mgmt   WebSocket     Command Parse      /cmd_vel
Camera Feed      BEST_EFFORT     Context Mgmt   Streaming     Tool Calling       /response_voice
Sensor Data      Async Bridge    Cost Control   Real-time     Action Planning    Custom Actions
/conversation_id  Conversation    Conv Monitor   API Calls     Response Format    /conversation_id
```

## Provider Implementations

### OpenAI Realtime Agent (Active)

**Status**: Fully implemented and operational

**Key Features**:
- WebSocket streaming with bidirectional audio
- Built-in VAD and turn detection
- Function calling for ROS command execution
- Intelligent session cycling with cost controls
- Context compression and memory management

**Models Supported**:
- `gpt-4o-realtime-preview` (primary)
- `gpt-4o-audio-preview` (fallback)

**Performance Characteristics**:
- Session spin-up: ~300ms
- Audio latency: < 100ms end-to-end
- Context window: 128K tokens with intelligent compression
- Cost optimization: Cached pricing, automatic session cycling

### Google Gemini Live Agent (Planned)

**Status**: Placeholder implementation ready

**Planned Features**:
- Native multimodal input (audio + 1024x1024 video)
- Affective dialog capabilities
- Real-time streaming with low latency
- Integration with Google Cloud services

**Target Models**:
- `gemini-live-2.5-flash-preview`
- `gemini-2.5-flash-preview-native-audio-dialog`

**Expected Performance**:
- Session spin-up: < 400ms
- Multimodal latency: < 150ms
- Video processing: 1024x1024 JPEG at 5-30Hz

### LangGraph Orchestration (Planned)

**Status**: Architecture placeholder ready

**Planned Capabilities**:
- Multi-agent workflow coordination
- Long-term planning and goal management
- Cross-provider session routing
- Complex task decomposition
- Memory persistence across sessions

## Technical Architecture

### Async/ROS Concurrency Bridge

**Challenge**: ROS2 uses callback-based concurrency while LLM APIs require asyncio event loops

**Solution**: `ros_ai_bridge` provides zero-copy message queuing between paradigms
- ROS2 callbacks append to thread-safe queues
- Asyncio agents consume from queues without blocking
- Responses flow back through similar queue mechanism
- BEST_EFFORT QoS for real-time streaming compatibility

### Session Management Strategy

**Lifecycle States**:
1. **IDLE**: No active session, minimal resource usage
2. **CONNECTING**: Establishing WebSocket/API connection
3. **ACTIVE**: Processing conversations with context retention
4. **PAUSED**: Temporary suspension with context preservation
5. **CYCLING**: Intelligent session refresh for cost/memory optimization
6. **ERROR**: Failure state with automatic recovery attempts

**Cost Optimization**:
- Session cycling at configurable token/cost/time limits
- Context compression using LLM summarization
- Cached pricing models with real-time cost tracking
- Intelligent pause detection to reduce API usage

### Provider Abstraction Layer

**Common Interface**:
```python
class BaseAgent:
    async def initialize() -> None
    async def process_audio_chunk(chunk: AudioData) -> None
    async def process_image(image: Image) -> None
    async def get_response() -> AgentResponse
    async def execute_command(command: str) -> ActionResult
    async def stop() -> None
```

**Provider-Specific Extensions**:
- OpenAI: Real-time streaming, function calling
- Gemini: Multimodal input, affective responses
- Future providers: Custom capabilities as needed

## Integration Points

### Upstream Dependencies
- **ros_ai_bridge**: Message transport and queue management
- **voice_detection**: Speech chunk generation and VAD
- **audio_common**: Audio capture and output
- **camera**: Visual input streams (future)
- **conversation_id topic**: System-wide conversation boundary coordination
  - **Topic Name**: `conversation_id` (relative topic - respects namespace/prefix)
  - **Message Type**: `std_msgs/String`
  - **Format**: `conv_YYYYMMDD_HHMMSS_microseconds` (e.g., `conv_20250804_143052_123456`)
  - **Publishers**: All agents publish on conversation timeout or reset
  - **Subscribers**: All agents subscribe to detect external conversation changes
  - **Behavior**: Any ID change triggers immediate context reset across all agents
  - **With Namespaces**: e.g., `/robot1/voice/conversation_id` for multi-robot scenarios

### Downstream Interfaces
- **ROS2 Actions**: Robot movement and manipulation commands
- **TTS Systems**: Speech synthesis for agent responses
- **Logging**: Structured logging with LangSmith integration
- **Monitoring**: Performance metrics and cost tracking

### Configuration Management

**Provider Configuration**:
```yaml
agents:
  openai_realtime:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4o-realtime-preview"
    voice: "alloy"
    session_max_tokens: 50000
    session_max_cost: 5.00
    pause_timeout: 10.0
    
  gemini_live:  # Future
    api_key: "${GOOGLE_API_KEY}"
    model: "gemini-live-2.5-flash-preview"
    video_resolution: "1024x1024"
    affective_mode: true
```

## Conversation Coordination

### Multi-Agent Conversation Synchronization

The conversation_id topic enables synchronized conversation boundaries across multiple agents:

#### Conversation Lifecycle
1. **Shared Conversation Context**: All agents operate within the same conversation boundary
2. **Synchronized Resets**: When any agent detects timeout or reset, all agents transition together
3. **Unified User Experience**: Users perceive a single coherent conversation across agents

#### Implementation Pattern
```python
class BaseAgent:
    def __init__(self):
        # Subscribe to conversation_id for external changes (relative topic)
        self.conversation_id_sub = self.create_subscription(
            String, 'conversation_id', self.handle_conversation_change
        )
        
        # Publish conversation_id on timeout or reset (relative topic)
        self.conversation_id_pub = self.create_publisher(
            String, 'conversation_id'
        )
        
    async def handle_conversation_change(self, msg):
        """Handle external conversation ID changes"""
        if msg.data != self.current_conversation_id:
            # Reset context and session
            await self.reset_conversation_context()
            self.current_conversation_id = msg.data
```

#### Coordination Scenarios
1. **Timeout Coordination**: When conversational agent times out, command agent also resets
2. **Manual Reset**: Operator can publish new conversation_id to reset all agents
3. **User Switching**: Future voice/face recognition triggers coordinated reset
4. **Error Recovery**: Critical errors can trigger system-wide conversation reset

#### Benefits
- **Consistent State**: All agents maintain synchronized conversation boundaries
- **Clean Handoffs**: Context resets prevent cross-conversation contamination
- **Debugging**: Conversation IDs in logs enable easy correlation across agents
- **Metrics**: System-wide conversation tracking and analytics

## Quality Assurance

### Testing Strategy
1. **Unit Testing**: Individual agent components and session management
2. **Integration Testing**: Multi-provider failover and orchestration
3. **Performance Testing**: Latency, throughput, and resource consumption
4. **Cost Testing**: Billing accuracy and optimization effectiveness
5. **Reliability Testing**: Network failures, API limits, session recovery

### Monitoring and Observability
- **LangSmith Integration**: Session tracing and performance analytics
- **Structured Logging**: JSON logs with correlation IDs
- **Metrics Collection**: Response times, token usage, error rates
- **Cost Tracking**: Real-time spending with alerts and limits

### Error Handling and Recovery
- **Graceful Degradation**: Fallback to simpler models or providers
- **Automatic Retry**: Exponential backoff with circuit breakers
- **Session Recovery**: Context preservation across connection failures
- **Provider Switching**: Seamless failover between OpenAI and Gemini

## Future Enhancements

### ROS AI Bridge Improvements
1. **Topic Aliasing System**: 
   - Allow agents to subscribe to logical topic names (e.g., `camera/image_raw`)
   - Bridge maintains alias mapping in configuration (e.g., `camera/image_raw` → `/grunt1/arm1/cam_live/color/image_raw`)
   - Agents remain hardware-agnostic and portable across different robot configurations
   - Benefits: Easier multi-robot deployment, simplified agent development, centralized topic management

### Planned Provider Integrations
1. **Anthropic Claude**: Constitutional AI and safety-focused responses
2. **Cohere Command**: Specialized for command and control applications
3. **Local Models**: Whisper.cpp, Llama.cpp for edge deployment
4. **Custom Providers**: Organization-specific fine-tuned models

### Advanced Orchestration Features
1. **Multi-Agent Conversations**: Collaborative problem solving
2. **Specialized Agents**: Task-specific LLMs (navigation, manipulation, conversation)
3. **Memory Persistence**: Long-term context across robot power cycles
4. **Learning Integration**: Feedback loops for continuous improvement

### Performance Optimizations
1. **Model Caching**: Local model deployment for reduced latency
2. **Request Batching**: Optimize API usage for multiple concurrent requests
3. **Predictive Pre-loading**: Anticipate agent needs based on context
4. **Hardware Acceleration**: GPU utilization for local model inference

## Deployment Considerations

### Resource Requirements
- **CPU**: 2-4 cores recommended for multiple concurrent agents
- **Memory**: 512MB-1GB RAM depending on active provider count
- **Network**: Stable internet connection for cloud APIs (5+ Mbps recommended)
- **Storage**: 100MB for agent code + variable for context/cache storage

### Configuration Tuning
- **Development**: Single provider (OpenAI) with verbose logging
- **Production**: Multi-provider with failover and cost controls
- **Edge Deployment**: Local models with cloud fallback
- **Research**: Full observability with LangSmith integration

### Security Considerations
- **API Key Management**: Environment variables and secure storage
- **Data Privacy**: Audio/video data handling and retention policies
- **Network Security**: TLS/WSS encryption for all API communications
- **Access Control**: Role-based permissions for agent configuration

## Success Criteria

### Technical Metrics
- **Multi-Provider Readiness**: 2+ providers implementable within 4 weeks
- **Session Reliability**: 99.5% successful session establishment
- **Response Latency**: 95th percentile < 300ms end-to-end
- **Cost Efficiency**: 30% reduction through intelligent session management

### User Experience Metrics
- **Natural Conversations**: Seamless provider switching without user awareness
- **Command Accuracy**: 95%+ correct interpretation of robot commands
- **Context Continuity**: Maintain conversation coherence across session cycles
- **Robustness**: Graceful handling of network failures and API limits

### Development Metrics
- **Code Reusability**: 80%+ shared components across providers
- **Testing Coverage**: 90%+ code coverage for agent implementations
- **Documentation Quality**: Complete API documentation and integration guides
- **Deployment Simplicity**: Single-command deployment with default configurations

## Architecture Benefits

### Scalability
- **Horizontal**: Add new providers without affecting existing implementations
- **Vertical**: Support multiple concurrent sessions per provider
- **Resource**: Intelligent resource allocation based on workload

### Maintainability
- **Separation of Concerns**: Provider-specific code isolated in subdirectories
- **Common Patterns**: Shared interfaces and utilities reduce duplication
- **Testing**: Isolated testing of individual providers and orchestration

### Extensibility
- **Plugin Architecture**: Easy addition of new LLM providers
- **Custom Tools**: ROS command integration extensible to new domains
- **Configuration**: Runtime provider selection and parameter tuning

This multi-agent architecture positions the by_your_command package as a comprehensive platform for LLM-powered robotics, with the flexibility to adapt to new providers and use cases while maintaining high performance and reliability standards.