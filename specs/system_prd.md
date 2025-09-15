# By Your Command System PRD

> **Note: This document predates the topic renaming refactoring (2025-09-14). Some topic names mentioned here have been updated in the implementation. See `topic_renaming_refactoring_prd.md` for current naming.**


## 1. Executive Summary

By Your Command is a comprehensive ROS 2 package enabling real-time, multimodal human-robot interaction through voice, camera, and video streams. The system provides a complete pipeline from sensor input through LLM integration, enabling robots to engage in natural conversations, understand commands, and respond with voice synthesis.

## 2. System Architecture Overview

### 2.1 Core Architecture Principles

- **Distributed Processing**: Separation of ROS 2 callback-based concurrency (sensors/actuators) from asyncio-based concurrency (AI/WebSocket APIs)
- **Modular Design**: Loosely coupled nodes communicating through standardized ROS 2 topics
- **Real-time Processing**: Low-latency pipeline optimized for conversational interaction
- **Cost Optimization**: Intelligent session management to minimize API usage
- **Fault Tolerance**: Automatic reconnection and state recovery mechanisms

### 2.2 System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Audio Capture  │    │  Camera Capture  │    │ Video Streams   │
└────────┬────────┘    └────────┬─────────┘    └────────┬────────┘
         │                      │                        │
         ▼                      ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Echo Suppressor │    │ Image Processing │    │ Video Processing│
└────────┬────────┘    └────────┬─────────┘    └────────┬────────┘
         │/audio_filtered       │/camera/image_raw      │/video/*
         ▼                      ▼                        ▼
┌─────────────────┐            │                        │
│   Silero VAD    │            │                        │
└────────┬────────┘            │                        │
         │/prompt_voice         │                        │
         ▼                      ▼                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                        ROS AI Bridge                              │
│  - WebSocket Server (port 8765)                                  │
│  - Message Queuing & Routing                                     │
│  - Protocol Translation (ROS ↔ WebSocket)                        │
│  - Supports up to 10 concurrent agent connections                │
│                                                                  │
│  Subscribed Topics (ROS → Agents):                               │
│    • /prompt_voice (AudioDataUtterance)                          │
│    • /prompt_text (String)                                        │
│    • /camera/image_raw (Image)                                   │
│                                                                  │
│  Published Topics (Agents → ROS):                                │
│    • /response_voice (AudioData)                                      │
│    • /response_text (String)                                    │
│    • /cmd_vel (Twist)                                           │
│    • /response_cmd (String) [dual-agent mode]              │
│    • /command_detected (Bool) [dual-agent mode]                  │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              │ WebSocket Protocol
                              │ ┌─────────────────────┐
                              │ │ Outbound Messages:  │
                              │ │ • voice_chunks      │
                              │ │ • text_input        │
                              │ │ • camera_frames     │
                              ▼ └─────────────────────┘
                    ┌─────────────────────┐
                    │    LLM Agents       │
                    │                     │
                    │ • OpenAI Realtime   │
                    │ • Command Extractor │
                    │ • Gemini Live      │
                    │ • Claude            │
                    └─────────┬───────────┘
                              │
                              │ ┌─────────────────────┐
                              │ │ Inbound Messages:   │
                              │ │ • audio_out         │
                              │ │ • transcript        │
                              │ │ • robot_cmds       │
                              │ │ • command_detected  │
                              ▲ └─────────────────────┘
                              │
                              │ WebSocket Protocol
                              │
┌─────────────────────────────┴────────────────────────────────────┐
│                        ROS AI Bridge                              │
│                 (Same instance as above)                          │
└─────────────────────────────┬────────────────────────────────────┘
                              │
         ┌────────────────────┼───────────┬────────────────┐
         ▼                    ▼           ▼                ▼
┌─────────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐
│  Audio Player   │  │Robot Commands│  │LLM Transcript│  │Command Trans.│
│ (/response_voice)    │  │ (/cmd_vel)   │  │(/llm_trans.) │  │(/cmd_trans.) │
└────────┬────────┘  └──────────────┘  └─────────────┘  └──────┬───────┘
         │                                                       │
         ▼                                                       ▼
┌─────────────────┐                                    ┌─────────────────┐
│/assistant_speaking│                                  │/command_detected│
└─────────────────┘                                    └─────────────────┘
```

### 2.3 Dual-Agent Architecture

The system supports running multiple specialized agents concurrently, enabling sophisticated behaviors through separation of concerns:

```
                           /prompt_voice
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                        ROS AI Bridge                              │
│                                                                  │
│  Broadcasting to all subscribed agents:                          │
│    • Each agent receives same /prompt_voice                      │
│    • Agents process independently                                │
│    • No inter-agent communication required                       │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             │ WebSocket Broadcast
                ┌────────────┴────────────┐
                ▼                         ▼
┌────────────────────────┐    ┌────────────────────────┐
│  Conversational Agent  │    │  Command Extractor     │
│                        │    │                        │
│ Config:                │    │ Config:                │
│ • agent_id:            │    │ • agent_id:            │
│   "openai_realtime"    │    │   "openai_command"     │
│ • Friendly dialogue    │    │ • Command detection    │
│ • Natural responses    │    │ • No audio output      │
│ • Full voice output    │    │ • Minimal responses    │
└───────────┬────────────┘    └────────────┬───────────┘
            │                               │
            ▼                               ▼
     OpenAI RT API                   OpenAI RT API
            │                               │
            ▼                               ▼
     (Responses)                     (Commands only)
            │                               │
            │ WebSocket                     │ WebSocket
            ▼                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                        ROS AI Bridge                              │
│                                                                  │
│  Publishing agent-specific outputs:                              │
│    Conversational:              Command Extractor:               │
│    • /response_voice                 • /response_cmd            │
│    • /response_text            • /command_detected              │
│    • /assistant_speaking        • /cmd_vel                       │
└──────────────────────────────────────────────────────────────────┘
```

**Key Benefits:**
- **Parallel Processing**: Both agents process same input simultaneously
- **Specialized Behavior**: Each agent optimized for its specific task
- **No Conflicts**: Different output topics prevent interference
- **Cost Optimization**: Command extractor can use cheaper/faster models
- **Extensibility**: Easy to add more specialized agents

## 3. Functional Requirements

### 3.1 Voice Interaction

#### 3.1.1 Input Processing
- **Real-time Voice Activity Detection**: < 50ms latency using Silero VAD
- **Utterance Chunking**: Configurable chunking with pre-roll support
- **Echo Suppression**: Prevent feedback loops in open-mic scenarios
- **Multi-format Support**: Handle various audio encodings (int16, float32, etc.)

#### 3.1.2 Output Generation
- **Text-to-Speech**: Support multiple voice models (OpenAI voices, Google voices)
- **Audio Playback**: Direct speaker output with configurable sample rates
- **Conversation State**: Track assistant speaking status for echo control

### 3.2 Visual Processing

#### 3.2.1 Camera Integration
- **Image Capture**: Subscribe to standard ROS 2 camera topics
- **Frame Processing**: Batch processing for efficiency
- **Multimodal Context**: Combine visual and audio inputs for LLM

#### 3.2.2 Video Streams
- **Stream Management**: Handle multiple concurrent video sources
- **Frame Extraction**: Intelligent keyframe selection
- **Bandwidth Optimization**: Adaptive quality based on network conditions

### 3.3 LLM Integration

#### 3.3.1 Supported Providers
- **OpenAI Realtime API**: WebSocket-based streaming (fully implemented)
- **Google Gemini Live**: Low-latency multimodal (planned)
- **Anthropic Claude**: Via API bridge (planned)

#### 3.3.2 Session Management
- **Connection Lifecycle**: Automatic establishment and recovery
- **Cost Optimization**: Session cycling on conversation pauses
- **Context Preservation**: Maintain conversation continuity across sessions

### 3.4 Robot Control

#### 3.4.1 Command Processing
- **Natural Language Understanding**: Parse voice commands to robot actions
- **Action Execution**: Translate to ROS 2 actions/services
- **Safety Constraints**: Validate commands before execution

#### 3.4.2 Feedback Mechanisms
- **Status Reporting**: Voice feedback on command execution
- **Error Handling**: Graceful degradation with user notification

## 4. Non-Functional Requirements

### 4.1 Performance

- **Voice Detection Latency**: < 50ms
- **Speech-to-Text**: Real-time streaming
- **Response Generation**: 1-2 seconds for voice response
- **Audio Playback Latency**: < 100ms from API to speakers
- **System Resource Usage**: < 2GB RAM, < 20% CPU on modern hardware

### 4.2 Reliability

- **Uptime**: 99.9% for core voice pipeline
- **Fault Recovery**: Automatic reconnection within 5 seconds
- **Data Integrity**: No loss of utterances or commands
- **Graceful Degradation**: Continue operation with reduced functionality

### 4.3 Security

- **API Key Management**: Secure storage, never in logs or repos
- **Network Security**: TLS/WSS for all external connections
- **Access Control**: Configurable permissions for robot commands
- **Privacy**: Local processing where possible, minimal data retention

### 4.4 Scalability

- **Concurrent Sessions**: Support multiple robots/users
- **Modular Extension**: Easy addition of new LLM providers
- **Resource Scaling**: Adaptive quality based on available resources

## 5. Technical Specifications

### 5.1 Audio Pipeline

#### 5.1.1 Input Specifications
- **Sample Rate**: 16kHz (standard), configurable
- **Channels**: Mono (default), stereo supported
- **Format**: PCM int16 (primary), float32 supported
- **Chunk Size**: 100 frames (configurable)

#### 5.1.2 Output Specifications
- **Sample Rate**: 24kHz (OpenAI), 16kHz (others)
- **Channels**: Mono
- **Format**: PCM int16
- **Buffering**: Queue-based with overflow protection

### 5.2 Message Protocols

#### 5.2.1 ROS 2 Messages
- **AudioDataUtterance**: Enhanced audio with utterance metadata
- **AudioDataUtteranceStamped**: Timestamped version
- **Standard Messages**: Bool, String, Image, etc.

#### 5.2.2 WebSocket Protocol
- **MessageEnvelope**: Standardized wrapper for all messages
- **JSON Payloads**: Configuration and control messages
- **Binary Payloads**: Audio and video data

### 5.3 Configuration Management

- **YAML Configuration**: Node parameters and settings
- **Environment Variables**: API keys and runtime options
- **Launch Parameters**: Dynamic configuration at startup
- **Named Prompts**: Context-aware system prompts

## 6. Development Guidelines

### 6.1 Code Organization

- **Package Structure**: Logical separation by functionality
- **Naming Conventions**: Descriptive, consistent naming
- **Documentation**: Inline comments and README files
- **Testing**: Unit and integration tests required

### 6.2 Best Practices

- **Logging**: Descriptive, leveled logging throughout
- **Error Handling**: Graceful failures with recovery
- **Resource Management**: Proper cleanup and disposal
- **Performance**: Profile and optimize critical paths

### 6.3 Contribution Process

- **Code Review**: All changes require review
- **Testing**: Automated tests must pass
- **Documentation**: Update relevant docs
- **Versioning**: Semantic versioning for releases

## 7. Deployment Considerations

### 7.1 Hardware Requirements

- **Minimum**: 4GB RAM, 2 CPU cores, USB microphone
- **Recommended**: 8GB RAM, 4 CPU cores, quality audio interface
- **GPU**: Optional, beneficial for local model inference

### 7.2 Software Dependencies

- **ROS 2**: Humble or newer
- **Python**: 3.8+
- **Audio**: PortAudio, PyAudio
- **AI/ML**: Various based on chosen providers

### 7.3 Network Requirements

- **Bandwidth**: 1 Mbps minimum for voice
- **Latency**: < 100ms to API endpoints
- **Reliability**: Stable connection for streaming

## 8. Future Roadmap

### 8.1 Short Term (1-3 months)

- Complete Gemini Live integration
- Add Claude API support
- Implement visual command processing
- Enhance error recovery mechanisms

### 8.2 Medium Term (3-6 months)

- Local model support (Whisper, LLaMA)
- Multi-language support
- Advanced robot control primitives
- Performance optimization

### 8.3 Long Term (6-12 months)

- Fully autonomous conversation management
- Multi-robot coordination
- Advanced multimodal understanding
- Edge deployment optimization

## 9. Success Metrics

### 9.1 Technical Metrics

- **Response Time**: < 2s for 95% of interactions
- **Recognition Accuracy**: > 95% for clear speech
- **Uptime**: > 99.9% for core services
- **API Costs**: < $0.10 per conversation hour

### 9.2 User Experience Metrics

- **Conversation Success Rate**: > 90%
- **User Satisfaction**: > 4.5/5 rating
- **Task Completion**: > 85% success rate
- **Natural Interaction**: Minimal user training required

## 10. Risk Mitigation

### 10.1 Technical Risks

- **API Availability**: Multiple provider fallback
- **Network Failures**: Local caching and retry
- **Resource Constraints**: Adaptive quality settings
- **Security Breaches**: Regular audits and updates

### 10.2 Operational Risks

- **Cost Overruns**: Usage monitoring and limits
- **Scaling Issues**: Load testing and optimization
- **User Adoption**: Intuitive design and documentation
- **Maintenance Burden**: Automated testing and deployment

## 11. Conclusion

By Your Command represents a comprehensive solution for multimodal human-robot interaction, combining state-of-the-art AI capabilities with robust robotics infrastructure. The system's modular architecture, real-time performance, and extensibility make it suitable for a wide range of robotics applications, from research platforms to production deployments.

The successful implementation of Phase 1 (voice interaction with OpenAI Realtime API) demonstrates the viability of the architecture and provides a solid foundation for future enhancements. The roadmap outlined above will expand capabilities while maintaining the core principles of reliability, performance, and user experience that define the system.