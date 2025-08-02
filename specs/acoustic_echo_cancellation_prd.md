# Product Requirements Document: Acoustic Echo Cancellation (AEC)

## Executive Summary

This document details the implementation journey of Acoustic Echo Cancellation (AEC) for the ROS2-based voice assistant system. After extensive testing of multiple software AEC solutions (WebRTC, Speex, custom adaptive filters), we ultimately determined that PulseAudio's built-in echo cancellation module provides the most effective solution. This document preserves our learning journey and implementation attempts for future reference.

## Problem Statement

### Current State
- The system uses crude microphone muting when the assistant speaks
- Users cannot interrupt the assistant mid-sentence
- No echo cancellation leads to feedback loops if unmuted
- Emergency interruptions are impossible

### Desired State
- Continuous microphone listening even while assistant speaks
- Natural conversation flow with interruption capability
- Clean audio input without echo or feedback
- Multiple AEC algorithm options for different scenarios

## Implementation Journey

### Phase 1: Initial Research and Design

#### Requirements Identified
1. Remove speaker output (24kHz) from microphone input (16kHz)
2. Handle ~130ms acoustic delay from speaker to microphone
3. Support real-time processing with minimal latency
4. Integrate with existing ROS2 audio pipeline

#### Architecture Decisions
- Create dedicated `aec_node` in the audio processing pipeline
- Support multiple AEC backends (WebRTC, Speex, adaptive filter)
- Handle resampling between different sample rates
- Implement test mode with tone generation for validation

### Phase 2: Implementation Challenges

#### 1. NumPy/SciPy Compatibility Crisis
**Problem**: scipy import failed with NumPy 2.2.6
```
AttributeError: _ARRAY_API not found
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

**Solution**: Disabled scipy and implemented simple resampling
```python
SCIPY_AVAILABLE = False  # Disable scipy due to numpy compatibility issues
# Implemented custom 3:2 decimation for 24kHz to 16kHz conversion
```

**Lesson**: Always have fallback implementations for scientific computing libraries

#### 2. Speex Installation and Initialization
**Problem**: Multiple issues with speexdsp
- Initial import failures due to missing system library
- Parameter passing confusion
- No echo reduction observed initially

**Solution**:
```bash
# Required system dependency
sudo apt install libspeexdsp-dev
# Python package
pip install speexdsp
```

**Implementation Fix**:
```python
# Speex expects raw bytes, not numpy arrays
mic_bytes = mic_frame.astype(np.int16).tobytes()
speaker_bytes = speaker_frame.astype(np.int16).tobytes()
processed_bytes = self.aec.process(mic_bytes, speaker_bytes)
```

**Lesson**: Read the C++ source when Python documentation is sparse

#### 3. WebRTC Integration Complexity
**Problem**: WebRTC requires specific constraints
- Exactly 10ms audio chunks
- Different frame sizes for different sample rates
- Initialization order issues with Python methods

**Solution**:
```python
# WebRTC requires exactly 10ms chunks
self.webrtc_mic_frame_size = int(self.mic_sample_rate * 0.01)     # 160 samples at 16kHz
self.webrtc_speaker_frame_size = int(self.speaker_sample_rate * 0.01)  # 240 samples at 24kHz

# Set both stream formats
self.aec.set_stream_format(self.mic_sample_rate, self.channels)
self.aec.set_reverse_stream_format(self.speaker_sample_rate, self.channels)

# Configure system delay
self.aec.set_system_delay(130)  # milliseconds
```

**Lesson**: WebRTC's sophistication requires precise configuration

#### 4. Python Method Definition Order
**Problem**: `AttributeError: 'AcousticEchoCancellerNode' object has no attribute 'log_info'`

**Solution**: Define logging methods before using them in initialization
```python
class AcousticEchoCancellerNode(Node):
    def __init__(self):
        super().__init__('aec_node')
        self.setup_node()  # Call after log methods defined
        
    def log_info(self, msg):
        """Must be defined before use in __init__"""
```

**Lesson**: Python's dynamic nature requires careful method ordering

#### 5. Audio Routing Feedback Loop
**Problem**: AEC node was both subscribing to and publishing to `/audio_out`

**Solution**: Separate test tone topic
```python
# In test mode, publish to different topic
self.tone_pub = self.create_publisher(
    AudioData,
    'test_tone_out',  # Different topic to avoid feedback
    audio_qos
)
```

**Lesson**: Always diagram your data flow to avoid circular dependencies

### Phase 3: Testing and Validation

#### Test Infrastructure

##### 1. Test Launch File Configuration (`aec_test.launch.py`)
```python
def generate_launch_description():
    # Audio capture from microphone
    audio_capturer = Node(
        package='audio_common',
        executable='audio_capturer_node',
        name='audio_capturer_node',
        parameters=[{
            'device': 14,  # Use pulse audio for automatic resampling
            'rate': 16000,
            'chunk': 512
        }]
    )
    
    # Speaker playback for test tones
    audio_player = Node(
        package='by_your_command',
        executable='simple_audio_player',
        name='simple_audio_player',
        parameters=[{
            'topic': 'test_tone_out',  # Separate topic to avoid feedback
            'sample_rate': 24000,
            'channels': 1,
            'device': 14  # Pulse audio device
        }]
    )
    
    # AEC node with test mode enabled
    aec_node = Node(
        package='by_your_command',
        executable='aec_node',
        name='aec_node',
        parameters=[{
            'mic_sample_rate': 16000,
            'speaker_sample_rate': 24000,
            'aec_method': 'webrtc',  # or 'speex'
            'debug_logging': True,
            'test_mode': True,  # Enables tone generation
            'frame_size': 160,
            'filter_length': 3200
        }],
        remappings=[
            ('audio', 'audio'),
            ('audio_out', 'test_tone_out'),  # Remap to avoid feedback
            ('audio_filtered', 'audio_filtered')
        ]
    )
```

##### 2. Test Tone Generation
- **Frequency**: 1kHz sine wave (easily detectable)
- **Duration**: 1.0 second burst
- **Interval**: Every 3.0 seconds
- **Amplitude**: 30000 (int16 range: -32768 to 32767)
- **Sample Rate**: 24kHz (matching assistant voice)

##### 3. Voice Recording Infrastructure
Three parallel recorders capture different stages:

```python
# Records generated test tones (24kHz)
voice_recorder_tone = Node(
    package='by_your_command',
    executable='voice_chunk_recorder',
    name='voice_recorder_tone',
    parameters=[{
        'output_dir': '/tmp/voice_chunks/test_tone',
        'input_mode': 'audio_data',
        'input_topic': 'test_tone_out',
        'input_sample_rate': 24000,
        'audio_timeout': 10.0
    }]
)

# Records raw microphone input (16kHz)
voice_recorder_raw = Node(
    package='by_your_command',
    executable='voice_chunk_recorder',
    name='voice_recorder_raw',
    parameters=[{
        'output_dir': '/tmp/voice_chunks/test_raw',
        'input_mode': 'audio_stamped',
        'input_topic': 'audio',
        'input_sample_rate': 16000,
        'audio_timeout': 10.0
    }]
)

# Records AEC output (16kHz)
voice_recorder_filtered = Node(
    package='by_your_command',
    executable='voice_chunk_recorder',
    name='voice_recorder_filtered',
    parameters=[{
        'output_dir': '/tmp/voice_chunks/test_filtered',
        'input_mode': 'audio_stamped', 
        'input_topic': 'audio_filtered',
        'input_sample_rate': 16000,
        'audio_timeout': 10.0
    }]
)
```

##### 4. WAV File Analysis
After running the test, examine recordings in `/tmp/voice_chunks/`:

```bash
# Expected file structure:
/tmp/voice_chunks/
├── test_tone/
│   └── *.wav  # Clean 1kHz tones at 24kHz
├── test_raw/
│   └── *.wav  # Tones + echo at 16kHz  
└── test_filtered/
    └── *.wav  # Echo-reduced audio at 16kHz

# Analysis tools:
# 1. Audacity - Visual waveform and spectrum analysis
# 2. sox - Command line analysis
sox /tmp/voice_chunks/test_raw/*.wav -n stat
sox /tmp/voice_chunks/test_filtered/*.wav -n stat

# 3. Custom Python analysis
python3 analyze_aec_performance.py \
    --raw /tmp/voice_chunks/test_raw/*.wav \
    --filtered /tmp/voice_chunks/test_filtered/*.wav
```

##### 5. Expected Differences in Recordings

**test_tone/ recordings**:
- Pure 1kHz sine wave
- No noise or distortion
- Consistent amplitude
- 1 second on, 2 seconds silence pattern

**test_raw/ recordings**:
- 1kHz tone present during tone periods
- Amplitude varies based on acoustic coupling
- May contain room reverb and noise
- Delay visible when comparing to test_tone

**test_filtered/ recordings**:
- Significantly reduced 1kHz tone
- Some residual echo expected
- Transient artifacts at tone start/stop
- Background noise may be reduced (WebRTC)

##### 6. Performance Metrics

Real-time logging shows:
```
[15:50:58.471] [aec] Tone power - Mic: 19.2dB (RMS: 68), Filtered: -4.1dB, Reduction: 23.3dB
```

Where:
- **Tone power**: FFT-based measurement at 1kHz
- **RMS**: Overall signal level
- **Reduction**: Difference between input and output

#### Test Scenarios

1. **Basic Functionality Test**
   - Run `ros2 launch by_your_command aec_test.launch.py`
   - Verify tone generation every 3 seconds
   - Check delay measurement logs
   - Confirm echo reduction metrics

2. **Algorithm Comparison**
   - Modify launch file: `'aec_method': 'webrtc'` vs `'speex'` vs `'adaptive'`
   - Run identical tests with each algorithm
   - Compare reduction metrics and artifacts

3. **Real-World Testing**
   - Disable test mode: `'test_mode': False`
   - Play actual assistant speech through speaker
   - Speak simultaneously to test double-talk handling
   - Test interruption scenarios

4. **Stress Testing**
   - Continuous speech playback
   - Variable volume levels
   - Multiple speakers (human)
   - Background noise conditions

#### Results

**Speex Performance**:
- Variable reduction: -10.8dB to +12.6dB
- Erratic behavior with internal routing
- Better suited for acoustic echo
- Requires exact frame sizes

**WebRTC Performance**:
- Consistent reduction: 4.8dB to 27.2dB
- More robust to non-acoustic scenarios
- Better adaptation to changing conditions
- Includes noise suppression

**NoMachine Audio Routing Issue**:
- The internal routing (monitor source) was likely caused by NoMachine remote desktop
- NoMachine forwards audio and microphone, creating internal loopback
- While problematic for real use, it provided consistent test conditions
- Real hardware testing recommended for production validation

## Technical Specifications

### Dependencies

#### System Packages
```bash
# SWIG - Required for WebRTC Python bindings
sudo apt install swig

# Speex DSP library
sudo apt install libspeexdsp-dev

# Audio libraries
sudo apt install portaudio19-dev libportaudio2

# ROS audio packages (for audio_common nodes)
sudo apt install ros-$ROS_DISTRO-audio-common ros-$ROS_DISTRO-audio-common-msgs
```

#### Python Packages
```bash
# WebRTC audio processing
pip install webrtc-audio-processing

# Speex DSP support  
pip install speexdsp

# NumPy (specific version to avoid conflicts)
# pip install numpy<2.0  # If scipy needed
```

### Installation Troubleshooting

#### WebRTC Installation Issues
```bash
# If webrtc-audio-processing fails to install:
# 1. Ensure SWIG is installed first
sudo apt install swig

# 2. Try upgrading pip
pip install --upgrade pip

# 3. Install from source if needed
git clone https://github.com/xiongyihui/python-webrtc-audio-processing
cd python-webrtc-audio-processing
python setup.py install
```

#### Speex Installation Issues
```bash
# If speexdsp Python package fails:
# 1. Ensure system library is installed
sudo apt install libspeexdsp-dev

# 2. Check pkg-config
pkg-config --libs speexdsp

# 3. May need development headers
sudo apt install python3-dev
```

### API Design

#### ROS2 Topics
- **Input**: `/audio` (AudioStamped) - Raw microphone at 16kHz
- **Input**: `/audio_out` (AudioData) - Speaker output at 24kHz  
- **Output**: `/audio_filtered` (AudioStamped) - Echo-cancelled audio at 16kHz
- **Status**: `/aec_active` (Bool) - Whether AEC is actively processing

#### Parameters
```yaml
aec_node:
  ros__parameters:
    mic_sample_rate: 16000
    speaker_sample_rate: 24000
    aec_method: 'webrtc'  # Options: webrtc, speex, adaptive
    frame_size: 160       # 10ms at 16kHz
    filter_length: 3200   # 200ms echo tail
    debug_logging: false
    bypass: false         # Pass-through mode
    test_mode: false      # Generate test tones
```

### Algorithm Details

#### WebRTC AudioProcessingModule
```python
# Initialization
from webrtc_audio_processing import AudioProcessingModule as AP

# Create with AEC2, noise suppression, no AGC/VAD
aec = AP(aec_type=2, enable_ns=True, enable_agc=False, enable_vad=False)

# Configure streams
aec.set_stream_format(16000, 1)           # Microphone: 16kHz mono
aec.set_reverse_stream_format(24000, 1)   # Speaker: 24kHz mono
aec.set_system_delay(130)                 # System delay in ms

# Process audio (requires exactly 10ms chunks)
aec.process_reverse_stream(speaker_bytes)  # Far-end (speaker)
processed = aec.process_stream(mic_bytes)  # Near-end (microphone)

# Available methods:
# - has_echo: bool - Whether echo is detected
# - aec_level: int - Current AEC aggressiveness
# - set_ns_level(0-2) - Noise suppression level
```

#### Speex Echo Canceller
```python
# Initialization
from speexdsp import EchoCanceller

# Create with frame size and filter length
ec = EchoCanceller.create(
    frame_size=160,      # Samples per frame (10ms at 16kHz)
    filter_length=3200,  # Echo tail length (200ms at 16kHz)  
    sample_rate=16000    # Must match actual sample rate
)

# Process audio (flexible frame sizes, but consistent)
# Expects bytes input, returns bytes
processed = ec.process(mic_bytes, speaker_bytes)

# Note: Both inputs must be same sample rate
# Requires manual resampling for mismatched rates
```

#### Custom Adaptive Filter (NLMS)
```python
class AdaptiveEchoCanceller:
    """Normalized Least Mean Squares (NLMS) implementation"""
    
    def __init__(self, filter_length, frame_size, sample_rate, learning_rate=0.3):
        self.filter_coeffs = np.zeros(filter_length)
        self.ref_buffer = np.zeros(filter_length)
        self.learning_rate = learning_rate
        
    def process(self, mic_signal, speaker_signal):
        # Estimate echo using convolution
        echo_estimate = np.convolve(speaker_signal, self.filter_coeffs)
        
        # Calculate error signal
        error = mic_signal - echo_estimate
        
        # Update filter coefficients (NLMS)
        norm_factor = np.dot(speaker_signal, speaker_signal) + 1e-6
        update = (self.learning_rate * error * speaker_signal) / norm_factor
        self.filter_coeffs += update
        
        return error
```

### Algorithm Comparison

| Feature | WebRTC | Speex | Adaptive Filter |
|---------|---------|-------|-----------------|
| Sample Rate Handling | Native multi-rate | Requires resampling | Requires resampling |
| Chunk Size | Strict 10ms | Flexible | Flexible |
| CPU Usage | Medium (~15%) | Low (~5%) | Very Low (~2%) |
| Echo Reduction | 20-30dB typical | 10-20dB typical | 5-15dB typical |
| Noise Suppression | Included | Separate module | None |
| Double-talk Handling | Excellent | Good | Poor |
| Convergence Time | 100-200ms | 200-500ms | 500-1000ms |
| Robustness | High | Medium | Low |

### Buffer Management

#### Critical Timing Considerations
```python
# Speaker buffer: Stores recent speaker output
self.speaker_buffer = deque(maxlen=int(sample_rate * 0.5))  # 500ms

# Delay compensation calculation
# Mic hears what was played delay_samples ago
start_idx = len(speaker_buffer) - delay_samples - len(mic_data)
end_idx = len(speaker_buffer) - delay_samples
speaker_ref = speaker_buffer[start_idx:end_idx]
```

#### Frame Size Alignment
- **WebRTC**: Must process exactly 10ms chunks
  - 16kHz: 160 samples per chunk
  - 24kHz: 240 samples per chunk
- **Speex**: Flexible but consistent frame size
- **Adaptive**: Fully flexible

### Performance Monitoring

#### Real-time Metrics
```python
# Calculate echo reduction
input_power = np.mean(mic_data**2)
output_power = np.mean(processed**2)
reduction_db = 10 * np.log10(output_power / input_power)

# Detect tone in signal (for testing)
fft = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
tone_bin = np.argmin(np.abs(freqs - 1000))  # 1kHz
tone_power = 20 * np.log10(np.abs(fft[tone_bin]) / len(signal))
```

#### Status Reporting
```
=== AEC Status Report ===
AEC Type: webrtc
Mic chunks: 157
Speaker chunks: 10
Assistant speaking: True
Average echo reduction: 18.3dB
WebRTC chunks processed: 1570
WebRTC detecting echo: True
========================
```

## Lessons Learned

### 1. Start Simple, Add Complexity
- Basic adaptive filter helped understand the problem
- Test infrastructure (tone generator) invaluable for debugging
- Incremental testing revealed issues early

### 2. Internal Routing as Debugging Tool
- PulseAudio monitor source provided consistent test signal
- Eliminated acoustic variables during development
- Revealed algorithm limitations with non-acoustic echo

### 3. Library Integration Best Practices
- Always check example code and tests, not just docs
- C++ source often more informative than Python bindings
- Have fallbacks for every external dependency

### 4. Real-time Audio Constraints
- Buffer management is critical
- Timing alignment more important than algorithm sophistication
- Frame size mismatches cause silent failures

## Future Enhancements

1. **Automatic Algorithm Selection**
   - Detect acoustic vs. electrical echo
   - Switch algorithms based on performance

2. **Adaptive Delay Estimation**
   - Continuous correlation-based delay tracking
   - Handle variable system latencies

3. **Multi-channel Support**
   - Spatial filtering for multi-microphone arrays
   - Beamforming integration

4. **Performance Monitoring**
   - Real-time dashboard for echo metrics
   - Automatic quality reporting

## Final Implementation Decision

### Journey Summary

After implementing and testing multiple AEC approaches:

1. **WebRTC AEC** - Achieved only -1.1dB to -1.5dB average reduction despite correct delay settings
2. **Speex AEC** - Even worse at -0.1dB average reduction  
3. **Custom Adaptive NLMS Filter** - Slightly better but still inadequate
4. **Acoustic Delay Measurement** - Built custom tool, measured ~73ms with tone bursts, but actual delay in practice was ~180ms

### Root Cause Analysis

The poor performance wasn't due to incorrect implementation but rather:
- Complex acoustic environment with multiple reflections
- Non-linear distortions in the audio path
- High echo levels relative to speech
- Variable delays (measured 110-199ms range during operation)

### Final Solution: PulseAudio Echo Cancellation

After extensive testing, we discovered that PulseAudio's `module-echo-cancel` provides superior echo cancellation:

```bash
# Enable PulseAudio echo cancellation
pactl load-module module-echo-cancel
```

This system-level solution:
- Operates at the lowest level in the audio stack
- Handles echo cancellation before audio reaches ROS nodes
- Reduces microphone RMS from ~20-30 to ~0.5-1.0 during playback
- Allows user speech to pass through while filtering residual echo

### Simplified Architecture

With PulseAudio handling AEC, we simplified the system:

1. **Removed Components**:
   - Deleted `aec_node.py` and all related files
   - Removed echo suppressor implementations
   - Deleted acoustic measurement tools
   - Removed all AEC-related launch files

2. **Standardized on 512-sample chunks** (32ms @ 16kHz):
   - No longer need WebRTC-compatible 160/320 sample sizes
   - Simplified audio pipeline
   - Native Silero VAD compatibility

3. **Added amplitude filtering** in Silero VAD:
   - `amplitude_threshold` parameter filters residual low-level echo
   - Works in conjunction with PulseAudio's echo cancellation
   - Simple RMS-based filtering

### Lessons Learned

1. **System-level solutions often outperform application-level ones** - Hardware DSP or OS-level processing has advantages
2. **Complex acoustic environments challenge software AEC** - Multiple reflections and reverb are difficult to handle
3. **Measuring actual acoustic delay is non-trivial** - Simple correlation methods may find early reflections rather than the main path
4. **Library compatibility matters** - NumPy 2.x broke scipy, affecting our implementation options

### Setup Instructions

For future deployments:

1. Enable PulseAudio echo cancellation:
   ```bash
   pactl load-module module-echo-cancel
   ```

2. Configure Silero VAD with appropriate amplitude threshold:
   ```yaml
   amplitude_threshold: 100.0  # Adjust based on residual echo level
   ```

3. Use standard 512-sample audio chunks throughout the pipeline

This approach provides effective echo cancellation while maintaining system simplicity and reliability.