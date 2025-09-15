# Utterance ID and End-of-Utterance Detection Implementation

## Overview

This implementation adds utterance tracking and end-of-utterance detection to the ROS2 audio processing pipeline. The solution addresses the need for better utterance continuity tracking across audio chunks while maintaining backward compatibility.

## Key Features

### 1. Utterance ID Stamping
- **Implementation**: Uses timestamp from the first audio frame in each new utterance as the utterance ID
- **Format**: 64-bit unsigned integer (nanoseconds since epoch)
- **Benefits**: 
  - Provides temporal ordering of utterances
  - Enables correlation with original audio timing
  - Implicit signal of short-term voice continuity

### 2. End-of-Utterance Detection
- **Implementation**: One-frame delay mechanism to detect utterance boundaries
- **Process**: 
  1. When VAD detects speech end, set pending flag
  2. On next audio frame, mark previous chunk as end-of-utterance
  3. Publish the marked chunk
- **Benefits**: Gives downstream consumers clear utterance boundary information

### 3. Enhanced Message Types

#### AudioDataUtterance.msg
```
# Enhanced AudioData with utterance metadata
float32[] float32_data
int32[] int32_data
int16[] int16_data
int8[] int8_data 
uint8[] uint8_data

# Utterance metadata
uint64 utterance_id      # Timestamp (nanoseconds) of first frame in utterance
bool is_utterance_end    # True if this is the last chunk in the utterance
uint32 chunk_sequence    # Sequential chunk number within this utterance (0-based)
```

#### AudioDataUtteranceStamped.msg
```
# Timestamped version for header compatibility
std_msgs/Header header
by_your_command/AudioDataUtterance audio_data_utterance
```

## Implementation Details

### Modified Files

1. **Message Definitions**:
   - `by_your_command/msg/AudioDataUtterance.msg` (new)
   - `by_your_command/msg/AudioDataUtteranceStamped.msg` (new)
   - `by_your_command/CMakeLists.txt` (updated)
   - `by_your_command/package.xml` (updated)

2. **Core Logic**:
   - `by_your_command/voice_detection/silero_vad_node.py` (enhanced)
   - `by_your_command/voice_detection/voice_chunk_recorder.py` (updated)

3. **Testing**:
   - `by_your_command/tests/test_utterance_chunks.py` (new)
   - `by_your_command/tests/test_recorder_integration.py` (new)

### Key Changes in Silero VAD Node

1. **Utterance Tracking State**:
   ```python
   self.current_utterance_id = None
   self.utterance_start_timestamp = None
   self.pending_utterance_end = False
   self.last_chunk_data = None
   ```

2. **Utterance Start Detection**:
   ```python
   if speech_activity and not self.in_utterance:
       self.current_utterance_id = int(msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec)
   ```

3. **End-of-Utterance with One-Frame Delay**:
   ```python
   if self.pending_utterance_end and self.last_chunk_data is not None:
       self.last_chunk_data.is_utterance_end = True
       self.chunk_pub.publish(self.last_chunk_data)
   ```

### Key Changes in Voice Chunk Recorder

1. **Message Type Update**:
   ```python
   from by_your_command.msg import AudioDataUtterance
   self.create_subscription(AudioDataUtterance, 'prompt_voice', self.chunk_callback, qos_profile=qos)
   ```

2. **Utterance-Aware File Management**:
   ```python
   # Create filename with utterance ID and timestamp
   utterance_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(msg.utterance_id / 1e9))
   fname = os.path.join(self.output_dir, f"utterance_{msg.utterance_id}_{utterance_time}.wav")
   ```

3. **Automatic File Closing on End Detection**:
   ```python
   if msg.is_utterance_end:
       self._close_current_file("End of utterance detected")
   ```

**Benefits**:
- **Precise file boundaries**: Files automatically close when utterance ends
- **Meaningful filenames**: Include both utterance ID and human-readable timestamp  
- **Better reliability**: Reduces reliance on timeout-based file closing
- **Chunk tracking**: Logs chunk sequence numbers for debugging

## Edge Case Handling

### 1. Audio Stream Stops on Voice Frame
- **Problem**: Stream ends while VAD still indicates voice activity
- **Solution**: `__del__` method publishes pending end-of-utterance chunk during cleanup

### 2. Dropped Chunks at Consumer
- **Problem**: Consumer misses chunk marked as utterance end
- **Solution**: Consumer can detect missing utterance end by:
  - Monitoring for new utterance_id without seeing previous end
  - Implementing timeout-based utterance closing
  - Using sequence numbers to detect gaps

### 3. Network Timing Variations
- **Benefit**: Utterance ID is tightly coupled with audio data timestamp, not network transmission time
- **Robustness**: Sequence numbers allow reconstruction of proper chunk ordering

## Backward Compatibility

### Option 1: Direct Migration
- Change topic type from `AudioData` to `AudioDataUtterance`
- Consumers access audio data same way: `msg.int16_data`, etc.
- Additional metadata available: `msg.utterance_id`, `msg.is_utterance_end`

### Option 2: Stamped Version
- Use `AudioDataUtteranceStamped` for header-based workflows
- Access pattern: `msg.audio_data_utterance.int16_data`
- Header available: `msg.header`

## Testing

### Test Script Usage

#### Basic Functionality Test
```bash
# Terminal 1: Run the enhanced VAD node
ros2 run by_your_command silero_vad_node

# Terminal 2: Run the test listener
ros2 run by_your_command test_utterance_chunks

# Terminal 3: Provide audio input (e.g., from microphone)
ros2 run audio_common audio_capturer_node
```

#### Recorder Integration Test
```bash
# Terminal 1: Run the updated recorder (specify output directory)
ros2 run by_your_command voice_chunk_recorder --ros-args -p output_dir:=/tmp/test_recordings

# Terminal 2: Send test utterances with metadata
ros2 run by_your_command test_recorder_integration
```

### Expected Output
```
🆕 New utterance started: ID=1642089123456789000
📊 CHUNK | Utterance: 1642089123456789000 | Chunk: 0 | Samples: 1600 | Duration: 0.10s
📊 CHUNK | Utterance: 1642089123456789000 | Chunk: 1 | Samples: 1600 | Duration: 0.10s
📊 END | Utterance: 1642089123456789000 | Chunk: 2 | Samples: 800 | Duration: 0.05s
✅ Utterance 1642089123456789000 completed
```

## Benefits for Downstream Consumers

1. **Optimization Opportunities**:
   - Buffer entire utterances for batch processing
   - Implement streaming algorithms with clear boundaries
   - Reduce computational overhead by processing complete utterances

2. **Robustness**:
   - Detect and handle dropped chunks
   - Implement proper utterance timeout handling
   - Maintain state consistency across network issues

3. **Analytics**:
   - Track utterance statistics (duration, chunk count)
   - Implement quality metrics per utterance
   - Enable utterance-level debugging and monitoring

## Future Enhancements

1. **Quality Metrics**: Add SNR, energy levels per chunk
2. **Confidence Scores**: Include VAD confidence in metadata
3. **Chunking Strategies**: Adaptive chunk sizes based on content
4. **Compression**: Optional audio compression for bandwidth optimization