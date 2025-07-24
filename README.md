# ByYourCommand (byc_)

ByYourCommand is a ROS 2 package for multimodal human-robot interaction supporting voice, camera, and video streams. It provides a complete pipeline from audio capture through LLM integration for real-time conversational robotics.

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
# From your ROS 2 workspace root
colcon build --packages-select by_your_command
source install/setup.bash
```

## Configuration

Edit `config/config.yaml` to add your API keys and model paths.

## Usage

Launch all nodes:

```bash
ros2 launch by_your_command bringup.launch.py
```

## Architecture

### Core Pipeline
```
Audio Capture → VAD → Speech Chunks → Bridge → Agents → LLM APIs
     ↓              ↓                      ↓        ↓
Camera Feed → Image Processing → Bridge → Agents → Multimodal LLMs
```

### Package Structure
- `voice_detection/`: Silero VAD for voice activity detection and speech chunk recording
- `ros_ai_bridge/`: Minimal data transport layer between ROS2 and async agents
- `agents/`: LLM integration agents with asyncio concurrency
  - `graph.py`: Agent orchestration and workflow management
  - `oai_realtime.py`: OpenAI Realtime API integration
  - `tools/`: Command processing and ROS action tools
- `interactions/`: Legacy Whisper → LLM interaction (being replaced by agents)
- `config/`: Configuration files for API keys, prompts, and parameters
- `specs/`: Technical specifications and PRDs for complex components
- `bringup/`: Launch files and system orchestration
- `setup/`: Installation scripts and dependencies
- `devrules/`: Development guidelines and coding standards

## Nodes

### silero_vad_node
A node that performs voice activity detection using the Silero VAD model.
**Parameters**:
- `sample_rate` (int, default 16000): Audio sampling rate in Hz.
- `frame_duration_ms` (int, default 20): Frame duration in milliseconds.
- `max_buffer_seconds` (int, default 5): Seconds of audio buffer for pre-roll.
- `pre_roll_ms` (int, default 300): Milliseconds to rewind before VAD trigger.
- `utterance_timeout_sec` (float, default 1.0): Silence duration to end an utterance.
- `utterance_chunk_sec` (float, default 2.0): Duration for sub-utterance chunking.

### speech_chunk_recorder
A test/debug node that subscribes to VAD output and writes speech chunks to WAV files.
**Parameters**:
- `output_dir` (string, default "/tmp"): Directory for output WAV files.
- `sample_rate` (int, default 16000): Audio sampling rate in Hz.

### ros_ai_bridge
A minimal data transport bridge that handles message queuing between ROS2's callback-based concurrency and agents using asyncio-based concurrency.
**Topics**:
- Subscribes: `/speech_chunks`, `/camera/image_raw`
- Publishes: `/audio_out`, `/cmd_vel`
**Parameters**:
- `max_queue_size` (int, default 100): Maximum queue size before dropping messages
- `subscribed_topics` (list): Topics to bridge from ROS to agents
- `published_topics` (list): Topics to publish from agents to ROS

### interaction_node (Legacy)
A node that transcribes speech chunks using Whisper and processes commands with an LLM via OpenAI.
**Parameters**:
- `openai_api_key` (string): OpenAI API key.
**Status**: Being replaced by agent-based architecture

## LLM Integration

### Supported APIs
- **OpenAI Realtime API**: WebSocket-based streaming with built-in VAD and turn detection
  - Models: `gpt-4o-realtime-preview`, `gpt-4o-audio-preview`
  - Features: Bidirectional audio streaming, cached pricing optimization
- **Google Gemini Live API**: Low-latency multimodal conversations
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

