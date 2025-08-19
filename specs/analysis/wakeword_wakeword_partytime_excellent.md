# Wake Word Attention Management Analysis
## Solving the "Always-On" Problem for Public Space Deployments

**Author**: Karim Virani 
**Date**: August 2025  
**Status**: Analysis Complete - Future Implementation Project

## Executive Summary

OpenAI's Realtime API lacks the ability to detect if conversation is directed at the robot, causing inappropriate responses in public spaces. This analysis explores multiple approaches to implement sleep/wake functionality, ultimately recommending OpenWakeWord as a comprehensive solution that combines VAD and wake word detection at the ROS level.

## Problem Statement

### Core Issue
OpenAI's Realtime API tries to respond to everything it hears, making it unsuitable for public deployments where:
- Background conversations trigger unwanted responses
- Robot appears to eavesdrop on private conversations  
- Continuous API usage incurs unnecessary costs
- User has no control over robot's attention state

### User Experience Requirements
- Natural sleep commands: "go to sleep", "be quiet", "stop talking"
- Clear wake mechanisms: "Hey [robotname]", "wake up"
- Visual/audio feedback of attention state
- Minimal latency when transitioning between states
- Graceful handling of public vs. private spaces

## Approaches Considered

### Approach 1: Sleep/Wake State Management (Initial Suggestion)
**Implementation**: Agent-level state machine with prompt switching

**Mechanism**:
- Sleep commands switch to minimal wake-word-only prompt
- OpenAI processes all audio but with restrictive instructions
- Wake words restore normal conversational prompt

**Pros**:
- Simple state machine implementation
- Preserves conversation context during sleep
- No API reconnection overhead

**Cons**:
- Still processes all audio (cost implications)
- OpenAI might still try to be helpful despite restrictive prompt
- Relies on prompt engineering for behavior control

**Assessment**: Workable but not optimal due to continued API usage during sleep

### Approach 2: Session Pause/Resume
**Implementation**: Complete API disconnection during sleep

**Mechanism**:
- Disconnect from OpenAI when sleeping
- Local VAD continues running to detect wake words
- Reconnect to OpenAI only when awakened

**Pros**:
- Zero API costs during sleep
- Guaranteed no responses during sleep
- Privacy-preserving (no audio sent to cloud)

**Cons**:
- Requires separate wake word detection system
- Slight delay on wake (session creation time)
- More complex session lifecycle management

**Assessment**: Cost-effective but requires additional infrastructure

### Approach 3: Selective Audio Forwarding
**Implementation**: Gate audio at bridge/agent level

**Mechanism**:
- Add gating logic in bridge or agent
- During sleep, audio only forwarded if local wake word detected
- Lightweight local model for pre-filtering

**Pros**:
- Reduces API calls during sleep
- Quick response on wake
- Can use existing VAD infrastructure

**Cons**:
- Still need local wake word detection
- More complex audio routing logic

**Assessment**: Good compromise between complexity and effectiveness

### Approach 4: Attention/Addressee Detection
**Implementation**: Multi-stage processing with attention detection

**Mechanism**:
- Preliminary "attention detector" agent
- Use cheaper/faster model to determine if speech is directed at robot
- Forward to main agent only if attention detected
- Could use cues: gaze, proximity, voice patterns, specific phrases

**Pros**:
- More natural interaction (no explicit wake words when active)
- Could work with multiple robots
- Scales for public environments

**Cons**:
- Complex multi-stage processing
- Need to define attention heuristics
- Higher computational overhead

**Assessment**: Sophisticated but may be over-engineered for current needs

### Approach 5: Hybrid with Proactive Audio Features
**Implementation**: Leverage Gemini Live's proactive capabilities

**Mechanism**:
- Use Gemini Live as attention detector when available
- OpenAI agent only activated when Gemini determines user addressing robot
- Fallback to wake word when only OpenAI available

**Pros**:
- Leverages platform capabilities
- More natural when Gemini available
- Graceful degradation

**Cons**:
- Dependency on multiple services
- Complexity in service coordination
- Gemini Live still experimental

**Assessment**: Interesting for future but adds complexity

### Approach 6: Context-Aware Filtering
**Implementation**: Environmental awareness with adaptive behavior

**Mechanism**:
- Monitor conversation patterns to auto-sleep
- Environmental detection (multiple voices, TV audio)
- Adaptive timeouts based on context

**Pros**:
- Adaptive to environment
- No explicit commands needed
- Smart power/cost management

**Cons**:
- Complex heuristics needed
- May miss legitimate interactions
- Difficult to tune reliably

**Assessment**: Promising long-term but requires significant development

## Key Architectural Insight

### The ROS-Level Gating Revelation
During analysis, we realized that if a local wake word engine is needed anyway, it's more elegant to control audio flow at the ROS level rather than managing complex state machines in the agent.

**Before (Complex)**:
```
Microphone → VAD → voice_chunks → Bridge → Agent → [Complex State Management]
```

**After (Elegant)**:
```
Microphone → VAD → [Wake Word Gate] → voice_chunks → Bridge → Agent
                           ↑
                    Sleep/Wake Control
```

### Benefits of ROS-Level Approach
1. **Clean Separation**: ROS handles wake/sleep, agent handles conversation only
2. **Cost Optimization**: Zero API calls during sleep (audio never reaches agent)  
3. **Simplicity**: Agent doesn't need state management logic
4. **Instant Wake**: No session creation delay
5. **Privacy**: Local processing during sleep

## OpenWakeWord Technical Analysis

### Discovery: Perfect Fit for Requirements
OpenWakeWord investigation revealed it's an ideal solution because:

1. **Already Includes Silero VAD**: No need for separate VAD implementation
2. **Proven Performance**: 15-20 models run simultaneously on RPi3
3. **Multiple Wake Words**: Can handle "Hey Barney", "Barney", "Wake up" simultaneously
4. **Built-in Features**: Speex noise suppression, configurable thresholds
5. **Custom Training**: Can create robot-specific wake words

### Technical Specifications

**Performance Characteristics**:
- Frame size: 80ms chunks (1280 samples at 16kHz)
- Latency: Real-time processing with minimal delay
- Accuracy: <5% false-reject, <0.5/hour false-accept rates
- Resource usage: Lightweight enough for edge deployment

**Audio Format Compatibility**:
- Input: 16kHz 16-bit PCM (matches current system)
- Processing: Melspectrogram → embeddings → classification
- VAD integration: Silero VAD built-in with configurable threshold

**Model Architecture**:
- Shared feature extraction backbone
- Individual classification heads per wake word
- Pre-trained on synthetic speech (high robustness)

### Advantages Over Alternatives

**vs. Separate VAD + Wake Word**:
- Single dependency instead of two
- Shared computation between functions
- Consistent audio pipeline
- Easier configuration management

**vs. Porcupine/Other Commercial**:
- Open source with Apache 2.0 license
- Custom training capabilities
- No licensing costs
- Active community support

**vs. Prompt-Based Solutions**:
- Hardware-level gating (no API costs)
- Guaranteed silence during sleep
- Immediate wake response
- Privacy preservation

## Recommended Architecture

### Multi-Tier Sleep System
```
ACTIVE (Normal conversation)
   ↓ (30s timeout OR sleep command)
LIGHT_SLEEP (Wake word only, local processing)
   ↓ (120s timeout)  
DEEP_SLEEP (OpenWakeWord only, no API)
   ↑ (Wake word detected)
ACTIVE (Resume conversation)
```

### Component Integration
```python
# Replaces silero_vad_node
class OpenWakeWordGateNode:
    def __init__(self):
        self.model = Model(
            wakeword_models=["hey_barney", "barney"],
            vad_threshold=0.5
        )
        self.state = GateState.ACTIVE
        
    def process_audio(self, frame):
        predictions = self.model.predict(frame)
        
        if self.state == GateState.ACTIVE:
            if predictions['vad'] > self.vad_threshold:
                self.forward_audio(frame)  # To bridge
                
        elif self.state == GateState.DEEP_SLEEP:
            for wake_word in ['hey_barney', 'barney']:
                if predictions[wake_word] > 0.5:
                    self.transition_to_active()
                    self.forward_audio(frame)
```

### State Transition Logic

**Timeout Management**:
- Active → Light Sleep: 30s no interaction
- Light Sleep → Deep Sleep: 120s no wake words
- Any State → Active: Wake word detected

**Command Detection**:
- Monitor `/llm_transcript` for sleep commands
- Immediate state transition on explicit commands
- Visual/audio feedback for state changes

### Configuration Schema
```yaml
openwakeword_gate:
  ros__parameters:
    # Wake word models
    wake_words:
      - model: "custom/hey_barney"
        threshold: 0.6
        phrases: ["hey barney", "barney", "wake up barney"]
    
    # Sleep triggers  
    sleep_commands: ["go to sleep", "be quiet", "stop talking", "shut up"]
    
    # VAD settings (replacing Silero)
    vad_threshold: 0.5
    enable_speex_ns: true
    
    # State management
    active_timeout: 30.0      # seconds
    deep_sleep_timeout: 120.0
    
    # Audio processing
    frame_size_ms: 80
    sample_rate: 16000
```

## Implementation Roadmap

### Phase 1: Foundation
**Goal**: Replace Silero VAD with OpenWakeWord VAD
- Install and configure OpenWakeWord
- Create new `openwakeword_gate_node.py`
- Test VAD functionality and performance
- Validate audio format compatibility

**Deliverables**:
- Working OpenWakeWord VAD node
- Performance benchmarks vs. Silero
- Integration tests with existing pipeline

### Phase 2: Wake Word Integration  
**Goal**: Add wake word detection and basic state management
- Implement pre-trained wake word models ("hey jarvis" initially)
- Add state machine (ACTIVE/LIGHT_SLEEP/DEEP_SLEEP)
- Implement audio gating based on state
- Add timeout-based state transitions

**Deliverables**:
- Multi-state audio gating system
- Wake word detection with configurable thresholds
- State persistence and recovery

### Phase 3: Custom Training
**Goal**: Train robot-specific wake words
- Use OpenWakeWord's Colab notebook for "Hey Barney" training
- Collect/generate training data for robot name
- Test and optimize custom model performance
- Add multiple wake phrase variations

**Deliverables**:
- Custom "Hey Barney" wake word model
- Training documentation and process
- Performance validation in target environments

### Phase 4: Advanced Features
**Goal**: Polish and production readiness
- Add visual/audio state indicators
- Implement command detection from transcripts
- Add environmental adaptation (public space detection)
- Performance optimization and tuning

**Deliverables**:
- Production-ready wake word system
- Complete documentation and configuration guides
- Performance metrics and tuning recommendations

## Expected Benefits

### User Experience
- Natural sleep/wake commands without remembering specific phrases
- Immediate response to wake words (no session creation delay)
- Clear indication of robot's attention state
- Appropriate behavior in public vs. private spaces

### Technical Performance
- Zero API costs during sleep periods
- Reduced false activations in noisy environments
- Consolidated VAD and wake word processing
- Simplified agent architecture

### System Integration  
- Cleaner separation of concerns (ROS vs. agent responsibilities)
- Easier testing and debugging of attention states
- Reusable architecture for future agents (Gemini, etc.)
- Standard configuration patterns

## Risk Assessment

### Low Risk
- OpenWakeWord is mature and actively maintained
- Silero VAD integration is proven
- Audio format compatibility is direct
- Fallback to current system is straightforward

### Medium Risk  
- Custom wake word training quality depends on data generation
- Need to tune thresholds for specific deployment environments
- Frame size changes may affect existing buffering logic

### Mitigation Strategies
- Start with pre-trained models before custom training
- Implement comprehensive configuration system for easy tuning
- Maintain backward compatibility during transition
- Extensive testing in target deployment environments

## Future Enhancements

### Short Term (3-6 months)
- Multi-robot wake word coordination
- Adaptive threshold tuning based on environment
- Integration with visual attention cues (if camera available)

### Long Term (6-12 months)
- Voice print recognition for user-specific responses
- Continuous learning for improved wake word accuracy
- Integration with Gemini Live's proactive audio features
- Advanced environmental context detection

## Conclusion

The OpenWakeWord solution represents an elegant architecture that solves the "always-on" problem while enhancing system capabilities. By moving attention management to the ROS level and leveraging OpenWakeWord's combined VAD and wake word detection, we achieve:

1. **Cost Efficiency**: Zero API usage during sleep
2. **User Control**: Natural sleep/wake commands  
3. **Privacy**: Local processing of sensitive audio
4. **Performance**: Immediate wake response
5. **Simplicity**: Cleaner agent architecture

This approach transforms a significant limitation of the OpenAI Realtime API into a feature that makes the robot more socially appropriate and user-friendly in all environments.

The phased implementation plan provides a clear path from current Silero VAD to a comprehensive attention management system, with each phase delivering incremental value while building toward the complete solution.