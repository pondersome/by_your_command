# By Your Command

by_your_command is a ROS 2 package for multimodal human-robot interaction supporting voice, camera, and video streams. It provides a complete pipeline from audio capture through LLM integration for real-time conversational robotics.

## Installation

Install ROS 2 dependencies:

```bash
# audio_common (publisher node & msg definitions)
sudo apt install ros-$ROS_DISTRO-audio-common ros-$ROS_DISTRO-audio-common-msgs

# PortAudio (for audio capture)
sudo apt install portaudio19-dev libportaudio2

# FFmpeg (openai-whisper needs ffmpeg to load audio files)
sudo apt install ffmpeg
```

Install Python dependencies:

```bash
cd setup
chmod +x setup.sh
./setup.sh
```

Build and source the package:

```bash
# From your ROS 2 workspace root (always use --symlink-install for faster config changes)
colcon build --packages-select by_your_command --symlink-install
source install/setup.bash
```

## Configuration

Edit `config/config.yaml` to add your API keys and model paths.

## Usage

Launch all nodes:

```bash
# Basic voice detection pipeline
ros2 launch by_your_command byc.launch.py

# OpenAI Realtime API integration
ros2 launch by_your_command oai_realtime.launch.py

# Individual nodes
ros2 run by_your_command silero_vad_node
ros2 run by_your_command voice_chunk_recorder
ros2 run by_your_command interaction_node

# Test utilities
ros2 run by_your_command test_utterance_chunks
ros2 run by_your_command test_recorder_integration
```

## Architecture

### Core Pipeline
```
Audio Capture → VAD → Voice Chunks → Bridge → Agents → LLM APIs
     ↓              ↓      (with utterance metadata)   ↓        ↓
Camera Feed → Image Processing → Bridge → Agents → Multimodal LLMs
```

### Package Structure
- `voice_detection/`: Silero VAD for voice activity detection and voice chunk recording
- `msg/`: Custom ROS message definitions (AudioDataUtterance, AudioDataUtteranceStamped)
- `ros_ai_bridge/`: Minimal data transport layer between ROS2 and async agents
- `agents/`: LLM integration agents with asyncio concurrency
  - `graph.py`: Agent orchestration and workflow management
  - `oai_realtime/`: OpenAI Realtime API integration
  - `tools/`: Command processing and ROS action tools
- `interactions/`: Legacy Whisper → LLM interaction (being replaced by agents)
- `tests/`: Test utilities and integration tests
- `config/`: Configuration files for API keys, prompts, and parameters
- `specs/`: Technical specifications and PRDs for complex components
- `bringup/`: Launch files and system orchestration
- `setup/`: Installation scripts and dependencies
- `devrules/`: Development guidelines and coding standards

## Nodes

### silero_vad_node
A node that performs voice activity detection using the Silero VAD model and publishes enhanced voice chunks with utterance metadata.

**Subscribed Topics**:
- `/audio` (audio_common_msgs/AudioStamped): Input audio stream

**Published Topics**:
- `/voice_activity` (std_msgs/Bool): Voice activity detection status
- `/voice_chunks` (by_your_command/AudioDataUtterance): Voice chunks with utterance metadata

**Parameters**:
- `sample_rate` (int, default 16000): Audio sampling rate in Hz
- `max_buffer_frames` (int, default 250): Maximum circular buffer size in frames
- `pre_roll_frames` (int, default 15): Frames to include before voice activity
- `utterance_chunk_frames` (int, default 100): Frames per chunk (0 = full utterance mode)
- `threshold` (float, default 0.5): VAD sensitivity threshold
- `min_silence_duration_ms` (int, default 200): Silence duration to end utterance

**Features**:
- Utterance ID stamping using first frame timestamp
- One-frame delay end-of-utterance detection
- Configurable chunking with pre-roll support

### voice_chunk_recorder
A node that subscribes to enhanced voice chunks and writes them to WAV files with utterance-aware naming.

**Subscribed Topics**:
- `/voice_chunks` (by_your_command/AudioDataUtterance): Enhanced voice chunks with metadata
- `/voice_activity` (std_msgs/Bool): Voice activity status (for debugging)

**Parameters**:
- `output_dir` (string, default "/tmp"): Directory for output WAV files
- `sample_rate` (int, default 16000): Audio sampling rate
- `close_timeout_sec` (float, default 2.0): Timeout for file closing

**Features**:
- Utterance-aware file naming: `utterance_{id}_{timestamp}.wav`
- Automatic file closing on end-of-utterance detection
- Chunk sequence logging for debugging

### ros_ai_bridge
A minimal data transport bridge that handles message queuing between ROS2's callback-based concurrency and agents using asyncio-based concurrency.
**Topics**:
- Subscribes: `/voice_chunks`, `/camera/image_raw`
- Publishes: `/audio_out`, `/cmd_vel`
**Parameters**:
- `max_queue_size` (int, default 100): Maximum queue size before dropping messages
- `subscribed_topics` (list): Topics to bridge from ROS to agents
- `published_topics` (list): Topics to publish from agents to ROS

### interaction_node (Legacy)
A node that transcribes voice chunks using Whisper and processes commands with an LLM via OpenAI.

**Subscribed Topics**:
- `/voice_chunks` (by_your_command/AudioDataUtterance): Voice chunks for transcription

**Parameters**:
- `openai_api_key` (string): OpenAI API key

**Status**: Being replaced by agent-based architecture

## Message Types

### AudioDataUtterance
Enhanced audio message with utterance metadata for voice chunk processing.

**Fields**:
- `float32[] float32_data` - Audio data in various formats
- `int32[] int32_data`
- `int16[] int16_data` 
- `int8[] int8_data`
- `uint8[] uint8_data`
- `uint64 utterance_id` - Timestamp (nanoseconds) of first frame in utterance
- `bool is_utterance_end` - True if this is the last chunk in the utterance
- `uint32 chunk_sequence` - Sequential chunk number within utterance (0-based)

### AudioDataUtteranceStamped
Timestamped version of AudioDataUtterance for header compatibility.

**Fields**:
- `std_msgs/Header header` - Standard ROS header with timestamp
- `by_your_command/AudioDataUtterance audio_data_utterance` - The audio data with metadata

## Testing Utilities

### test_utterance_chunks
Test listener that demonstrates enhanced voice chunk processing with utterance metadata.

**Usage**:
```bash
ros2 run by_your_command test_utterance_chunks
```

### test_recorder_integration
Integration test that generates synthetic voice chunks with proper utterance metadata for testing the voice chunk recorder.

**Usage**:
```bash
# Terminal 1: Start recorder with test directory
ros2 run by_your_command voice_chunk_recorder --ros-args -p output_dir:=/tmp/test_recordings

# Terminal 2: Generate test utterances
ros2 run by_your_command test_recorder_integration
```

## LLM Integration

### Supported APIs
- **OpenAI Realtime API**: WebSocket-based streaming with built-in VAD and turn detection
  - Models: `gpt-4o-realtime-preview`, `gpt-4o-audio-preview`
  - Features: Bidirectional audio streaming, cached pricing optimization
- **Google Gemini Live API - TBD**: Low-latency multimodal conversations
  - Models: `gemini-live-2.5-flash-preview`, `gemini-2.5-flash-preview-native-audio-dialog`
  - Features: Video support (1024x1024 JPEG), affective dialog capabilities

### Agent Architecture
The system uses an agent-based approach for LLM integration:
- **Separation of Concerns**: ROS bridge handles only data transport; agents handle LLM sessions
- **Asyncio Concurrency**: Agents run in asyncio event loops for optimal WebSocket performance  
- **Session Management**: Intelligent session lifecycle with cost optimization and timeout handling
- **Provider Flexibility**: Pluggable architecture supports multiple LLM providers

### Performance Targets
- **Audio Latency**: < 100ms from speech to LLM processing
- **Command Latency**: < 200ms from LLM response to ROS2 action
- **Session Spin-up**: < 500ms for new LLM connections
- **Throughput**: 50Hz audio chunks, 5-30Hz video streams

