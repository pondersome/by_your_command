# Voice Detection System PRD

**Author**: Karim Virani  
**Package**: by_your_command  
**Subsystem**: voice_detection  
**Version**: 2.1  
**Last Updated**: July 2025

## Overview

The voice detection system provides intelligent voice activity detection (VAD) and voice chunk extraction for the ByYourCommand robotic interaction package. This is entirely made possible by the excellent SileroVAD package. Here we extend its iterator capabilities to serve as a critical preprocessing layer that filters continuous audio streams to extract only voice-containing frames and assemble them into voice chunks with utterance metadata, optimizing downstream processing for transcription and LLM interaction.

**Related Documentation**: See [Utterance Enhancement Summary](utterance_enhancement_summary.md) for detailed implementation of utterance tracking features.

## Core Objectives

### Primary Goals
1. **Selective Audio Processing**: Stream only voice chunks to prevent unnecessary processing of silence and non-human audio
2. **Voice Completeness**: Ensure no voice is lost at utterance boundaries through intelligent pre-roll buffering
3. **Utterance Continuity**: Provide robust utterance tracking with unique IDs and boundary detection for downstream processing
4. **Distributed Architecture Support**: Enable flexible deployment across robot networks based on resource availability
5. **Real-time Performance**: Maintain low-latency voice detection suitable for interactive robotic applications

### Performance Requirements
- **Latency**: <100ms from voice start to detection
- **Accuracy**: Minimize false positives/negatives in voice detection
- **Resource Efficiency**: Continuous operation suitable for embedded/edge deployment
- **Network Optimization**: Reduce audio data transmission by ~80% through voice-only filtering

## System Architecture

### Node Structure
```
Audio Input → [silero_vad_node] → Voice Chunks → [downstream processors]
                     ↓
              [voice_chunk_recorder] (testing/validation)
```

### Core Components

#### 1. silero_vad_node
**Purpose**: Primary VAD processing and voice chunk extraction

**Inputs**:
- `/audio` (AudioStamped): Continuous audio stream from audio_capturer_node

**Outputs**:
- `/voice_chunks` (by_your_command/AudioDataUtterance): Enhanced voice segments with utterance metadata
- `/voice_activity` (Bool): Binary voice state for system coordination

**Key Features**:
- Frame-based processing with rolling buffer
- Utterance ID stamping using first frame timestamp
- One-frame delay end-of-utterance detection for precise boundaries
- Configurable chunking for long utterances with sequence numbering
- Pre-roll buffer to capture speech beginnings

#### 2. voice_chunk_recorder  
**Purpose**: Testing and validation node for voice chunk quality

**Inputs**:
- `/voice_chunks` (by_your_command/AudioDataUtterance): Enhanced voice segments with metadata
- `/voice_activity` (Bool): Utterance boundary markers

**Outputs**:
- WAV files: Reconstructed audio for validation testing

**Key Features**:
- Utterance-aware file naming with timestamp and ID
- Automatic file closing on end-of-utterance detection
- Chunk sequence tracking for debugging
- Quality validation through playback testing

## Technical Architecture

### Utterance Enhancement Features

The voice detection system implements comprehensive utterance tracking to provide downstream processors with rich metadata about voice chunk relationships and boundaries.

#### Utterance ID Stamping
- **Timestamp-based IDs**: Each utterance receives a unique 64-bit ID derived from the first frame's timestamp (nanoseconds since epoch)
- **Temporal Correlation**: Utterance IDs preserve timing relationship with original audio capture
- **Cross-chunk Continuity**: All chunks within an utterance share the same ID for easy grouping

#### End-of-Utterance Detection
- **One-frame Delay**: End-of-utterance detection uses a one-frame delay mechanism for precise boundary marking
- **Explicit Marking**: Final chunks are explicitly marked with `is_utterance_end=true` flag
- **Reliable Boundaries**: Eliminates guesswork in downstream processors about utterance completion

#### Chunk Sequencing
- **Sequential Numbering**: Each chunk within an utterance receives a sequence number (0-based)
- **Gap Detection**: Downstream processors can detect dropped chunks using sequence numbers
- **Ordering Guarantee**: Sequence numbers enable proper chunk reassembly despite network timing variations

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
2. **Integration Testing**: End-to-end audio pipeline validation with utterance metadata
3. **Real-time Testing**: Live microphone input with WAV output verification
4. **Performance Testing**: Latency and resource consumption measurement
5. **Utterance Testing**: Verification of ID uniqueness, boundary detection, and sequence integrity

### Validation Metrics
- **Detection Accuracy**: Speech/silence classification accuracy
- **Boundary Precision**: Utterance start/end timing accuracy  
- **Audio Quality**: Reconstructed speech clarity and completeness
- **System Latency**: Time from voice to chunk availability
- **Utterance Integrity**: ID uniqueness, sequence correctness, and boundary marking accuracy
- **Metadata Consistency**: Verification of chunk sequencing and end-of-utterance flags

### Debug and Monitoring
- Configurable logging levels with throttled verbose output
- Real-time buffer state visualization
- Performance metrics collection (processing time, queue depths)
- Audio artifact detection through playback validation
- Utterance metadata logging (ID, sequence, boundary events)

### Test Utilities
- **test_utterance_chunks**: Interactive test listener for utterance metadata validation
- **test_recorder_integration**: Synthetic utterance generator for recorder testing
- **Utterance-aware WAV naming**: Files named with utterance ID and timestamp for debugging

## Integration Points

### Upstream Dependencies
- **audio_common**: AudioStamped message input from audio_capturer_node
- **silero-vad**: PyPI package for VAD model and iterator

### Downstream Interfaces
- **ROS AI Bridge**: Voice chunk transport to external agents
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
- **Long-form Voice**: Enable streaming chunks, adjust timeout values
- **Low-latency Applications**: Minimize buffer depths, reduce timeouts

## Architecture Lessons Learned

### Major Enhancements (July 2025)
1. **Utterance Enhancement Implementation**: Added comprehensive utterance tracking with ID stamping, end-detection, and chunk sequencing
   - **Implementation**: One-frame delay boundary detection with timestamp-based unique IDs
   - **Benefits**: Enables precise utterance reconstruction and downstream processing optimization
   - **Documentation**: See [Utterance Enhancement Summary](utterance_enhancement_summary.md) for complete technical details

### Critical Bug Fixes (July 2025)
1. **Circular Buffer Corruption**: Absolute indexing with circular buffers caused data loss when buffer wrapped around
   - **Root Cause**: Storing `utterance_start_buffer_idx` as absolute index with circular deque
   - **Solution**: Eliminate absolute indexing, copy frames directly into dedicated buffers
   - **Impact**: Fixed missing audio data and state corruption on long utterances

2. **Race Conditions**: Timeout-based file closing in voice_chunk_recorder caused empty files
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

### Completed Enhancements (v2.1)
1. **Utterance ID Tracking**: Timestamp-based unique identifiers for utterance correlation
2. **End-of-Utterance Detection**: One-frame delay boundary marking for precise utterance completion
3. **Chunk Sequencing**: Sequential numbering within utterances for gap detection and ordering
4. **Enhanced Message Types**: AudioDataUtterance with rich metadata support

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
- 100% utterance ID uniqueness across system operation
- <1-frame accuracy for end-of-utterance detection

### User Experience Metrics
- Natural conversation flow without speech cutoffs
- Minimal false activations from environmental noise
- Reliable operation across diverse acoustic environments
- Seamless integration with downstream processing systems
- Consistent utterance boundary detection for improved transcription accuracy
- Robust chunk reassembly in downstream processors through sequence metadata

## Implementation Status (v2.1)

### ✅ Completed Features
- ✅ Core VAD processing with Silero integration
- ✅ Utterance ID stamping using timestamp-based unique identifiers
- ✅ One-frame delay end-of-utterance detection for precise boundaries
- ✅ Chunk sequence numbering within utterances
- ✅ Enhanced AudioDataUtterance message type with rich metadata
- ✅ Utterance-aware voice chunk recorder with automatic file management
- ✅ Comprehensive test utilities for validation and integration testing
- ✅ Speech→voice terminology standardization across the codebase

### 📋 Current Architecture
- **Message Types**: by_your_command/AudioDataUtterance, by_your_command/AudioDataUtteranceStamped
- **Topics**: /voice_chunks (enhanced metadata), /voice_activity (status)  
- **Nodes**: silero_vad_node (enhanced), voice_chunk_recorder (utterance-aware)
- **Test Utilities**: test_utterance_chunks, test_recorder_integration
- **Documentation**: Complete PRD with utterance enhancement summary

The voice detection system has successfully evolved from basic audio chunking to a comprehensive utterance tracking system that provides downstream processors with rich metadata for optimal processing and reconstruction.