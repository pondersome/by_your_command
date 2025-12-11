---
Author: Karim Virani  
Package: by_your_command  
Subsystem: voice_detection  
Version: 2.2  
Updated: August 2025
---
# Voice Detection System PRD

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
- `/prompt_voice` (by_your_command/AudioDataUtterance): Enhanced voice segments with utterance metadata
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
- `/prompt_voice` (by_your_command/AudioDataUtterance): Enhanced voice segments with metadata
- `/voice_activity` (Bool): Utterance boundary markers

**Outputs**:
- WAV files: Reconstructed audio for validation testing

**Key Features**:
- Utterance-aware file naming with timestamp and ID
- Automatic file closing on end-of-utterance detection
- Chunk sequence tracking for debugging
- Quality validation through playback testing

### Wake-Up Mechanisms

The voice detection system includes multiple wake-up mechanisms for reactivating the VAD when muted, providing flexible control options for different interaction scenarios.

#### 1. Adaptive Clap Detection
**Purpose**: Enable hands-free wake-up through acoustic pattern recognition

**Architecture**:
- **AdaptiveClapDetector Class**: Continuously monitors background noise and detects sharp acoustic transients
- **Double-Clap Pattern**: Requires two distinct claps within a timing window to prevent false positives
- **Adaptive Thresholding**: Dynamically adjusts to ambient noise levels for consistent detection across environments

**Key Design Decisions**:
- **Background RMS Tracking**: Uses exponential moving average (α=0.02) for gradual adaptation
- **Contamination Prevention**: Excludes loud sounds (>2x background) from background calculation to prevent clap sounds from raising threshold
- **Multi-Stage Validation**: Combines configurable spike ratio with hardcoded transient characteristics
- **State Machine**: Tracks first clap timing and validates second clap within acceptable gap window

**Detection Algorithm**:
1. Monitor audio RMS against adaptive background threshold
2. Initial spike detection when RMS exceeds `background_rms * spike_ratio` (configurable)
3. Validate transient characteristics (ALL must pass):
   - **Attack/Decay Ratio**: Start RMS > End RMS × 1.8 (sharp attack) - hardcoded
   - **Peak Detection**: Peak > Average RMS × 2.5 (impulsive spike) - hardcoded
   - **Frequency Content**: Zero-crossing rate > 0.2 (high frequency) - hardcoded
4. Track timing between claps for double-clap pattern recognition
5. Reset state if gap exceeds maximum threshold

**Fixed (v2.2)**: Previously, a hardcoded loudness check (15x background) invalidated the configurable `clap_spike_ratio`. This has been removed, making the parameter actually effective for tuning sensitivity.

#### 2. Text-Based Wake Commands
**Purpose**: Provide programmatic wake-up via text messages

**Supported Commands**:
- "wake", "awaken", "wake up", "wakeup" (case-insensitive, partial matching)
- Publishes to `/voice_active` topic for system-wide activation
- Includes feedback loop prevention with `self_triggered_wake` flag

**Use Cases**:
- Remote wake-up via network messages
- Integration with other system components
- Fallback when acoustic wake-up is unavailable

#### 3. Voice Active Control
**Purpose**: Direct mute/unmute control via ROS topic

**Interface**:
- Subscribe to `/voice_active` (Bool) for external control
- Publish to `/voice_active` (Bool) for wake events
- Bidirectional control enables system-wide coordination

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

### Clap Detection Parameters
```yaml
    clap_detection_enabled: true          # Enable adaptive clap detection
    clap_spike_ratio: 4.0                 # Clap must be Nx louder than background
    clap_min_gap_ms: 300                  # Minimum gap between claps (ms)
    clap_max_gap_ms: 800                  # Maximum gap between claps (ms)
```

## Environment Tuning Guide

### Clap Detection Tuning

The clap detection system uses adaptive thresholding to work across different environments, but fine-tuning can improve reliability:

#### Room Acoustics Considerations

**Quiet Environments (Office/Bedroom)**:
- Lower `clap_spike_ratio` to 3.0-3.5 for more sensitive detection
- Tighter gap window: `min_gap_ms: 250`, `max_gap_ms: 700`
- Background adapts quickly to minimal ambient noise

**Noisy Environments (Kitchen/Living Room)**:
- Higher `clap_spike_ratio` to 4.5-5.0 to reduce false positives
- Standard gap window: `min_gap_ms: 300`, `max_gap_ms: 800`
- Background adaptation filters out consistent noise (fans, appliances)

**Large/Echoey Rooms**:
- Keep standard `clap_spike_ratio` (4.0)
- Wider gap window: `min_gap_ms: 350`, `max_gap_ms: 1000`
- Accounts for echo delay between claps

**Small/Dampened Rooms**:
- Standard `clap_spike_ratio` (4.0)
- Tighter gap window: `min_gap_ms: 200`, `max_gap_ms: 600`
- Quick response due to minimal acoustic delay

#### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Missing first clap | Spike threshold too high | Decrease `clap_spike_ratio` by 0.5 |
| False positives from speech | Ratio too low | Increase `clap_spike_ratio` by 0.5 |
| Second clap not detected | Gap timing too restrictive | Increase `max_gap_ms` by 100-200 |
| Random noise triggers | Background contamination | Ensure robot is stationary during calibration |
| Location-dependent sensitivity | Different acoustic zones | Create per-location config profiles |
| Claps still missed after tuning | Hardcoded thresholds too strict | See limitations below |

#### Current Limitations

The clap detection system has several **hardcoded thresholds** that cannot be tuned via configuration:
- **Attack/Decay ratio (1.8x)** - may be too strict for soft claps
- **Peak-to-average ratio (2.5x)** - may miss claps with less pronounced peaks  
- **Zero-crossing rate (>0.2)** - may exclude lower-frequency claps

~~**Fixed in v2.2**: Previously, a hardcoded 15x loudness threshold invalidated the configurable parameter. This has been removed.~~

**Impact**: These remaining hardcoded values may still cause detection issues in certain scenarios. Consider exposing them as parameters if problems persist after tuning `clap_spike_ratio`.

#### Calibration Procedure

1. **Initial Setup**: Place robot in typical operating position
2. **Background Calibration**: Allow 5-10 seconds of ambient noise monitoring
3. **Test Claps**: Perform test claps from various distances (1m, 2m, 3m)
4. **Adjust Sensitivity**: 
   - If claps missed at desired range: decrease `clap_spike_ratio`
   - If false triggers occur: increase `clap_spike_ratio`
5. **Validate Timing**: Test natural clapping rhythm and adjust gap parameters
6. **Environment Testing**: Test with typical background noise (TV, conversation)

### VAD Tuning

**Speech Detection Sensitivity**:
- Lower `threshold` (0.3-0.4): More sensitive, may pick up breathing
- Standard `threshold` (0.5): Balanced for normal conversation
- Higher `threshold` (0.6-0.7): Less sensitive, requires louder speech

**Utterance Boundaries**:
- Short `min_silence_duration_ms` (150-200): Quick response, may fragment sentences
- Standard `min_silence_duration_ms` (250): Natural sentence boundaries
- Long `min_silence_duration_ms` (300-500): Groups multiple sentences together

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

### Subscribed Topics
- `/audio` (AudioStamped): Raw audio input stream
- `/voice_active` (Bool): Remote mute/unmute control
- `/prompt_text` (String): Text-based wake commands

### Published Topics
- `/prompt_voice` (AudioDataUtterance): Enhanced voice segments with metadata
- `/voice_activity` (Bool): Binary voice detection state
- `/voice_active` (Bool): Wake signals from clap/text detection

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

### Major Enhancements (August 2025)
1. **Adaptive Clap Detection**: Implemented hands-free wake-up mechanism with double-clap pattern recognition
   - **Architecture**: AdaptiveClapDetector class with background RMS tracking and spike detection
   - **Key Decision**: Double-clap pattern to prevent false positives from single transients
   - **Adaptive Design**: Dynamic threshold adjustment based on ambient noise levels
   - **State Machine**: Tracks timing between claps for pattern validation
   - **Bug Fix**: Removed hardcoded 15x loudness check that invalidated configurable spike_ratio

2. **Text-Based Wake Commands**: Added programmatic wake-up via prompt_text topic
   - **Commands**: "wake", "awaken", "wake up" with case-insensitive partial matching
   - **Integration**: Publishes to voice_active for system-wide coordination
   - **Safety**: Self-triggered wake flag prevents feedback loops

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
- **Avoid Redundant Thresholds**: Don't combine configurable and hardcoded thresholds for the same check (antipattern from clap detection)

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
5. **Configurable Clap Detection Thresholds**: Expose hardcoded parameters
   - Attack/decay ratio parameter
   - Peak-to-average ratio parameter
   - Loudness multiplier parameter
   - Zero-crossing rate threshold
   - Background update rate (alpha) parameter

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

## Implementation Status (v2.2)

### ✅ Completed Features
- ✅ Core VAD processing with Silero integration
- ✅ Utterance ID stamping using timestamp-based unique identifiers
- ✅ One-frame delay end-of-utterance detection for precise boundaries
- ✅ Chunk sequence numbering within utterances
- ✅ Enhanced AudioDataUtterance message type with rich metadata
- ✅ Utterance-aware voice chunk recorder with automatic file management
- ✅ Comprehensive test utilities for validation and integration testing
- ✅ Speech→voice terminology standardization across the codebase
- ✅ **Adaptive clap detection with double-clap pattern recognition**
- ✅ **Text-based wake commands via prompt_text topic**
- ✅ **Remote mute/unmute control via voice_active topic**
- ✅ **Background noise adaptation for consistent detection**

### 📋 Current Architecture
- **Message Types**: by_your_command/AudioDataUtterance, by_your_command/AudioDataUtteranceStamped
- **Topics**: 
  - Input: /audio, /voice_active, /prompt_text
  - Output: /prompt_voice, /voice_activity, /voice_active
- **Nodes**: silero_vad_node (with wake-up mechanisms), voice_chunk_recorder (utterance-aware)
- **Classes**: AdaptiveClapDetector (background tracking and spike detection)
- **Test Utilities**: test_utterance_chunks, test_recorder_integration, test_clap_detection
- **Documentation**: Complete PRD with wake-up mechanisms and environment tuning guide

The voice detection system now provides comprehensive voice processing with multiple wake-up mechanisms, making it suitable for hands-free robot interaction across diverse environments. The adaptive clap detection and text-based wake commands ensure reliable reactivation when muted, while maintaining the core VAD functionality for high-quality voice extraction.
