# Voice Detection System PRD

**Package**: by_your_command  
**Subsystem**: voice_detection  
**Version**: 2.0  
**Last Updated**: July 2025

## Overview

The voice detection system provides intelligent voice activity detection (VAD) and speech chunk extraction for the ByYourCommand robotic interaction package. It serves as a critical preprocessing layer that filters continuous audio streams to extract only speech-containing segments, optimizing downstream processing for transcription and LLM interaction.

## Core Objectives

### Primary Goals
1. **Selective Audio Processing**: Stream only speech chunks to prevent unnecessary processing of silence and non-human audio
2. **Speech Completeness**: Ensure no speech is lost at utterance boundaries through intelligent pre-roll buffering
3. **Distributed Architecture Support**: Enable flexible deployment across robot networks based on resource availability
4. **Real-time Performance**: Maintain low-latency speech detection suitable for interactive robotic applications

### Performance Requirements
- **Latency**: <100ms from speech start to detection
- **Accuracy**: Minimize false positives/negatives in speech detection
- **Resource Efficiency**: Continuous operation suitable for embedded/edge deployment
- **Network Optimization**: Reduce audio data transmission by ~80% through speech-only filtering

## System Architecture

### Node Structure
```
Audio Input → [silero_vad_node] → Speech Chunks → [downstream processors]
                     ↓
              [speech_chunk_recorder] (testing/validation)
```

### Core Components

#### 1. silero_vad_node
**Purpose**: Primary VAD processing and speech chunk extraction

**Inputs**:
- `/audio` (AudioStamped): Continuous audio stream from audio_capturer_node

**Outputs**:
- `/speech_chunks` (AudioData): Extracted speech segments with pre-roll
- `/voice_activity` (Bool): Binary speech state for system coordination

**Key Features**:
- Frame-based processing with rolling buffer
- Dual timeout system for utterance boundary detection  
- Configurable chunking for long utterances
- Pre-roll buffer to capture speech beginnings

#### 2. speech_chunk_recorder  
**Purpose**: Testing and validation node for speech chunk quality

**Inputs**:
- `/speech_chunks` (AudioData): Speech segments from VAD node
- `/voice_activity` (Bool): Utterance boundary markers

**Outputs**:
- WAV files: Reconstructed audio for validation testing

**Key Features**:
- Real-time audio reassembly
- Timestamped file output
- Quality validation through playback testing

## Technical Architecture

### Frame-Based Processing Model
- **Frame Size**: Full incoming audio chunks (512 samples @ 16kHz ≈ 32ms)
- **Processing Unit**: Maintains Silero VAD minimum input requirements (>31.25 sample ratio)
- **Buffer Management**: Separate dedicated buffers to eliminate circular buffer corruption

### VAD-Based Speech Detection
1. **VAD-Level Timeout** (`min_silence_duration_ms`): Internal Silero parameter for sentence breaks (250ms default)
2. **State Transition Detection**: Speech boundaries detected via VAD iterator state changes rather than timeout polling

### Buffering Strategy
```
[Circular Buffer] → [Pre-roll Extraction] → [Direct Frame Accumulation] → [Chunk Publication]
     ↑                        ↑                         ↑                        ↑
   History                Copy Frames              Dedicated Buffers          Output
```

**Buffer Architecture**:
- `frame_buffer`: Circular buffer (deque) for VAD processing and pre-roll history
- `utterance_buffer`: Direct accumulation for full utterance mode 
- `chunking_buffer`: Direct accumulation for streaming chunk mode

### Chunking Modes

#### Full Utterance Mode (`utterance_chunk_frames = 0`)
- Accumulates complete utterances
- Publishes single chunk per utterance
- Optimal for short-to-medium speech segments

#### Streaming Chunk Mode (`utterance_chunk_frames > 0`)  
- Publishes interim chunks during long utterances
- Enables low-latency processing of extended speech
- Maintains pre-roll on first chunk only

## Configuration Parameters

### Core VAD Parameters
```yaml
silero_vad_node:
  ros__parameters:
    sample_rate: 16000                    # Audio sampling rate
    threshold: 0.5                        # VAD sensitivity (0.0-1.0)
    min_silence_duration_ms: 250          # Internal VAD silence threshold
```

### Buffer Management
```yaml
    max_buffer_frames: 250                # Circular buffer depth for VAD/pre-roll (~8 seconds)
    pre_roll_frames: 15                   # Pre-speech buffer (~0.5 seconds)
```

### Chunking Control
```yaml
    utterance_chunk_frames: 10            # Interim chunk size (0 = full utterance)
                                          # 10 frames = ~0.32s streaming chunks
```

## Quality Assurance

### Testing Strategy
1. **Unit Testing**: VAD accuracy with known speech/silence samples
2. **Integration Testing**: End-to-end audio pipeline validation
3. **Real-time Testing**: Live microphone input with WAV output verification
4. **Performance Testing**: Latency and resource consumption measurement

### Validation Metrics
- **Detection Accuracy**: Speech/silence classification accuracy
- **Boundary Precision**: Utterance start/end timing accuracy  
- **Audio Quality**: Reconstructed speech clarity and completeness
- **System Latency**: Time from speech to chunk availability

### Debug and Monitoring
- Configurable logging levels with throttled verbose output
- Real-time buffer state visualization
- Performance metrics collection (processing time, queue depths)
- Audio artifact detection through playback validation

## Integration Points

### Upstream Dependencies
- **audio_common**: AudioStamped message input from audio_capturer_node
- **silero-vad**: PyPI package for VAD model and iterator

### Downstream Interfaces
- **ROS AI Bridge**: Speech chunk transport to external agents
- **Whisper Transcription**: Speech-to-text processing
- **LLM Interaction**: Natural language processing pipeline

### QoS Profile
```yaml
QoS:
  history: KEEP_LAST
  depth: 10
  reliability: BEST_EFFORT  # Prioritize recent audio over guaranteed delivery
```

## Deployment Considerations

### Resource Requirements
- **CPU**: Continuous VAD processing (~10-20% on modern CPU)
- **Memory**: Rolling buffer + model weights (~100MB typical)
- **Network**: Reduced bandwidth through speech-only transmission

### Configuration Tuning
- **Sensitive Environments**: Lower threshold, shorter silence duration
- **Noisy Environments**: Higher threshold, longer silence duration  
- **Long-form Speech**: Enable streaming chunks, adjust timeout values
- **Low-latency Applications**: Minimize buffer depths, reduce timeouts

## Architecture Lessons Learned

### Critical Bug Fixes (July 2025)
1. **Circular Buffer Corruption**: Absolute indexing with circular buffers caused data loss when buffer wrapped around
   - **Root Cause**: Storing `utterance_start_buffer_idx` as absolute index with circular deque
   - **Solution**: Eliminate absolute indexing, copy frames directly into dedicated buffers
   - **Impact**: Fixed missing audio data and state corruption on long utterances

2. **Race Conditions**: Timeout-based file closing in speech_chunk_recorder caused empty files
   - **Root Cause**: Files closed before final chunks processed
   - **Solution**: Event-driven closing on final chunk receipt
   - **Anti-pattern**: Increasing timeout duration (masks underlying issue)

3. **Log Flooding**: VAD speech activity messages created excessive output
   - **Solution**: Log on state changes (true↔false) + periodic 10-second intervals

### Design Principles
- **No Absolute Indexing**: Use direct frame copying with circular buffers
- **Event-Driven Processing**: Avoid timeout-based state management where possible  
- **Separate Buffer Concerns**: Dedicated buffers for different processing modes
- **Streaming-First Architecture**: BEST_EFFORT QoS with increased depth for real-time audio

## Future Enhancements

### Planned Features
1. **Multi-speaker Detection**: Speaker identification and separation
2. **Adaptive Thresholding**: Dynamic VAD sensitivity based on environment
3. **Advanced Chunking**: Semantic boundary detection for chunk breaks
4. **Performance Optimization**: GPU acceleration for VAD processing

### Integration Roadmap
1. **LangSmith Instrumentation**: Observability and performance monitoring
2. **Multi-modal Integration**: Coordination with visual attention systems
3. **Context Awareness**: Integration with robot state for environmental adaptation
4. **Edge Optimization**: Quantized models for embedded deployment

## Success Criteria

### Technical Metrics
- 95%+ speech detection accuracy in typical environments  
- <100ms average detection latency
- 80%+ reduction in audio data transmission
- <5% CPU utilization on target hardware

### User Experience Metrics
- Natural conversation flow without speech cutoffs
- Minimal false activations from environmental noise
- Reliable operation across diverse acoustic environments
- Seamless integration with downstream processing systems