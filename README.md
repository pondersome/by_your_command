# By Your Command

by_your_command is a ROS 2 package for multimodal human-robot interaction supporting voice, camera, and video streams. It provides a complete pipeline from audio capture through LLM integration for real-time conversational robotics.

## Warning: This project is in rapid development and is not ready for production use. Waaaay pre-alpha.

## Key Features

- **Voice Activity Detection**: Real-time speech detection using Silero VAD
- **OpenAI Realtime API Integration**: Full bidirectional voice conversations with GPT-4
- **Echo Suppression**: Prevents feedback loops in open-mic scenarios
- **Distributed Architecture**: WebSocket-based agent deployment for flexibility
- **Cost-Optimized Sessions**: Intelligent session cycling to manage API costs
- **Multi-Agent Support**: Dual-agent mode for simultaneous conversation and command extraction
- **Multiple Providers**: Support for multiple LLM providers (OpenAI, Gemini Live)
- **Command Processing**: Automatic robot command extraction and routing
- **Namespace Support**: Full ROS2 namespace and prefix flexibility for multi-robot deployments
- **Recursive Macro System**: Configurable prompts with nested macro expansion

## Quick Start

1. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

2. Launch the complete system:
```bash
ros2 launch by_your_command oai_realtime.launch.py
```

3. Speak naturally - the robot will respond with voice!

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

# Additional dependency for audio playback
pip3 install pyaudio
```

Build and source the package:

```bash
# From your ROS 2 workspace root (always use --symlink-install for faster config changes)
colcon build --packages-select by_your_command --symlink-install
source install/setup.bash
```

## Configuration

### Prompt Macros
The system supports recursive macro expansion in prompts, allowing for modular and maintainable prompt engineering:

```yaml
# Define macros in config/prompts.yaml
macros:
  robot_name: "Barney"
  arm_presets: "bumper, tenhut, lookup, lookout, reach"
  compound_commands: |
    {{arm_presets}} combined with @{{bearing_presets}}

# Use in prompts
system_prompts:
  my_agent:
    system_prompt: |
      You are {{robot_name}}, a helpful robot.
      You can move to: {{arm_presets}}
```

**Features**:
- Recursive expansion up to 10 levels deep
- Circular reference detection
- Shared macros across multiple agents
- Dynamic prompt composition

### API Keys
Set your OpenAI API key either in `config/oai_realtime_agent.yaml` or as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
export GEMINI_API_KEY="your-api-key-here"
```

### VAD Settings
Edit `config/config.yaml` to tune voice detection parameters.

### Acoustic Echo Cancellation (AEC)

For the best voice interaction experience, echo cancellation is critical to prevent the robot from hearing its own voice. We recommend a three-tier approach:

#### 1. Hardware Echo Cancellation (Recommended)
The best solution is to use hardware with built-in echo cancellation:
- **Headsets**: Any headset naturally prevents echo by physical separation
- **Smart Speakers/Conference Systems**: Many USB speakerphones have DSP-based AEC
- **Far-field Microphone Arrays**: Devices like ReSpeaker or Matrix Voice include AEC

#### 2. PulseAudio Echo Cancellation Module
When hardware AEC isn't available, use PulseAudio's software echo cancellation:

```bash
# Load the echo cancellation module
pactl load-module module-echo-cancel aec_method=webrtc source_name=echo_cancelled_source sink_name=echo_cancelled_sink

# Make it the default source
pactl set-default-source echo_cancelled_source

# To make this permanent, add to /etc/pulse/default.pa:
load-module module-echo-cancel aec_method=webrtc source_name=echo_cancelled_source sink_name=echo_cancelled_sink
set-default-source echo_cancelled_source
```

The PulseAudio module provides:
- WebRTC-based echo cancellation (same technology used in video calls)
- Automatic gain control and noise suppression
- Works with any standard audio hardware
- No additional latency in the ROS pipeline

#### 3. Echo Suppressor Node (Fallback)
As a last resort, we provide a simple time-based echo suppressor that mutes the microphone while the assistant is speaking:

```bash
# The echo_suppressor node is included in launches but can be run standalone:
ros2 run by_your_command echo_suppressor
```

This approach:
- Prevents feedback loops but doesn't allow interruption
- Has zero computational overhead
- Works in any environment
- Should only be used when options 1 and 2 aren't available

**Note**: The custom AEC node has been removed in favor of these more robust solutions. Our testing showed that PulseAudio's echo cancellation module provides superior performance with less complexity.

## Usage

Launch all nodes:

```bash
# OpenAI Realtime API integration (RECOMMENDED)
ros2 launch by_your_command oai_realtime.launch.py

# Dual-agent mode: Conversation + Command extraction
ros2 launch by_your_command oai_dual_agent.launch.py

# Enable voice recording for debugging
ros2 launch by_your_command oai_realtime.launch.py enable_voice_recorder:=true

# Save raw microphone input (post echo suppression) for AEC debugging
ros2 launch by_your_command oai_realtime.launch.py save_mic:=true

# Basic voice detection pipeline (without LLM)
ros2 launch by_your_command byc.launch.py

# Individual nodes
ros2 run by_your_command silero_vad_node
ros2 run by_your_command voice_chunk_recorder
ros2 run by_your_command simple_audio_player
ros2 run by_your_command echo_suppressor
ros2 run by_your_command command_processor

# Bridge and agents
ros2 run by_your_command ros_ai_bridge
ros2 run by_your_command oai_realtime_agent

# Test utilities
ros2 run by_your_command test_utterance_chunks
ros2 run by_your_command test_recorder_integration
ros2 run by_your_command test_command_processor
ros2 run by_your_command publish_command "lookup"
```

## Architecture

### Core Pipeline

#### Voice Input Flow
```
Microphone → audio_capturer → echo_suppressor → /audio_filtered → 
silero_vad → /voice_chunks → ROS Bridge → WebSocket → 
OpenAI Agent → OpenAI Realtime API
```

#### Voice Output Flow
```
OpenAI API → response.audio.delta → OpenAI Agent → WebSocket → 
ROS Bridge → /audio_out → simple_audio_player → Speakers
         ↓                                              ↓
         └──────────→ /llm_transcript ──────────────────┘
                                ↓
                      /assistant_speaking → echo_suppressor (mutes mic)

User Interruption Flow:
User Speech → VAD → /voice_chunks → Agent (while assistant speaking) →
  1. response.cancel → OpenAI API (stops generation)
  2. conversation.item.truncate → OpenAI API (cleans context)
  3. /interruption_signal → simple_audio_player → PyAudio abort() (immediate cutoff)
```

#### Complete System Architecture
```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│Audio Capture│     │Camera Capture│     │ Other Sensors   │
└─────┬───────┘     └──────┬───────┘     └────────┬────────┘
      ↓                    ↓                       ↓
┌─────────────┐     ┌──────────────┐              ↓
│    VAD      │     │Image Process │              ↓
└─────┬───────┘     └──────┬───────┘              ↓
      ↓                    ↓                       ↓
/voice_chunks        /camera/image_raw      /sensor_data
      ↓                    ↓                       ↓
      └────────────────────┴───────────────────────┘
                           ↓
                  ┌────────────────┐
                  │  ROS AI Bridge  │ (WebSocket Server)
                  └────────┬───────┘
                           ↓ WebSocket
                  ┌────────────────┐
                  │   LLM Agents   │ → External APIs
                  └────────┬───────┘
                           ↓ WebSocket
                  ┌────────────────┐
                  │  ROS AI Bridge  │
                  └────────┬───────┘
                           ↓
    ┌──────────────┬───────┴────────┬─────────────┐
/audio_out    /llm_transcript   /cmd_vel    /other_outputs
    ↓              ↓                ↓              ↓
┌─────────┐   ┌─────────┐      ┌────────┐    ┌────────┐
│ Speaker │   │ Logger  │      │ Motors │    │ Other  │
└─────────┘   └─────────┘      └────────┘    └────────┘
```

#### Dual-Agent Architecture
```
                        /voice_chunks
                              ↓
                    ┌─────────────────┐
                    │  ROS AI Bridge  │
                    │ (WebSocket:8765)│
                    └────────┬────────┘
                             │ WebSocket broadcast
                ┌────────────┴────────────┐
                ↓                         ↓
        ┌───────────────┐         ┌──────────────────┐
        │ Conversational│         │ Command Extractor│
        │     Agent     │         │      Agent       │
        │               │         │                  │
        │ Friendly chat │         │ COMMAND: move... │
        └───────┬───────┘         └────────┬─────────┘
                ↓                           ↓
        ┌───────────────┐         ┌──────────────────┐
        │OpenAI RT API  │         │ OpenAI RT API    │
        └───────┬───────┘         └────────┬─────────┘
                ↓ WebSocket                 ↓ WebSocket
        ┌───────────────┐         ┌──────────────────┐
        │  ROS Bridge   │         │   ROS Bridge     │
        └───────┬───────┘         └────────┬─────────┘
                ↓                           ↓
    ┌───────────┴─────────┐       ┌────────┴─────────┐
    ↓                     ↓       ↓                  ↓
/audio_out         /llm_transcript  /command_transcript
    ↓                                                ↓
┌────────┐                                  /command_detected
│Speaker │                                          ↓
└────────┘                                   ┌──────────────┐
                                            │   Command    │
                                            │  Processor   │
                                            └──────┬───────┘
                                                   ↓
                                            ┌──────────────────┐
                                            │/arm_preset       │
                                            │/behavior_command │
                                            └──────┬───────────┘
                                                   ↓
                                            ┌──────────────┐
                                            │Robot Control │
                                            └──────────────┘
```

### Package Structure
- `audio/`: Audio processing nodes (simple_audio_player, echo_suppressor)
- `voice_detection/`: Silero VAD for voice activity detection and voice chunk recording
- `msg/`: Custom ROS message definitions (AudioDataUtterance, AudioDataUtteranceStamped)
- `nodes/`: Core processing nodes (command_processor)
- `ros_ai_bridge/`: Minimal data transport layer between ROS2 and async agents
- `agents/`: LLM integration agents with asyncio concurrency
  - `common/`: Shared components across all agents
    - `websocket_bridge.py`: WebSocket bridge interface for distributed agent deployment
    - `prompt_loader.py`: Dynamic prompt loading with recursive macro expansion
    - `context.py`: Conversation context management and preservation
    - `conversation_monitor.py`: Real-time conversation state monitoring
    - `pause_detector.py`: Intelligent pause detection for session management
  - `graph.py`: Agent orchestration and workflow management
  - `oai_realtime/`: OpenAI Realtime API integration with prompt macros
  - `tools/`: Command processing and ROS action tools
- `interactions/`: Legacy Whisper → LLM interaction (being replaced by agents)
- `tests/`: Test utilities and integration tests
- `config/`: Configuration files with recursive macro support
  - `prompts.yaml`: System prompts with macro definitions
  - `oai_realtime_agent.yaml`: Agent configuration
  - `oai_command_agent.yaml`: Command extractor configuration
- `specs/`: Technical specifications and PRDs for complex components
- `bringup/`: Launch files and system orchestration
- `setup/`: Installation scripts and dependencies
- `devrules/`: Development guidelines and coding standards

#### Common Agent Components

The `agents/common/` module provides shared functionality across all agent implementations:

**Benefits**:
- **Code Reuse**: Consistent behavior across OpenAI, Gemini, and future agent types
- **Easier Maintenance**: Single implementation for core features like context management
- **Standardized APIs**: Uniform interfaces for WebSocket communication and prompt handling

**Components**:
- **WebSocketBridgeInterface**: Manages agent-to-bridge WebSocket connections with automatic reconnection
- **PromptLoader**: Handles dynamic prompt loading with recursive macro expansion
- **ConversationContext**: Preserves conversation history across session boundaries
- **ConversationMonitor**: Monitors conversation state and provides real-time insights
- **PauseDetector**: Intelligent detection of conversation pauses for session cycling

### Real-Time User Interruption System

The system supports natural interruptions where users can speak over the assistant to stop responses immediately:

#### Three-Stage Interruption Process

1. **OpenAI API Cancellation**
   - Detects user speech while assistant is speaking
   - Sends `response.cancel` to immediately stop LLM generation
   - Sends `conversation.item.truncate` to remove partial response from context
   - Prevents pollution of conversation history with incomplete text

2. **Context Cleanup**
   - Tracks the last assistant response item ID for proper truncation
   - Ensures conversation context remains clean after interruption
   - Maintains conversation flow without corrupted partial responses

3. **Audio Queue Clearing**
   - Publishes `/interruption_signal` to audio player
   - Clears buffered audio data to prevent continued playback
   - Uses PyAudio `abort()` for immediate audio cutoff (not graceful `stop()`)

#### Configuration

The interruption system requires the `/interruption_signal` topic in bridge configuration:

```yaml
# config/bridge_dual_agent.yaml
published_topics:
  - topic: "interruption_signal"
    msg_type: "std_msgs/Bool"
```

#### Troubleshooting Interruptions

- **Laggy interruptions**: Ensure `/interruption_signal` topic is configured in bridge
- **Continued audio after "stop"**: Check that audio player is using `abort()` not `stop()`
- **Context pollution**: Verify `conversation.item.truncate` is being sent with correct item ID
- **No interruption detection**: Monitor that user speech is detected while `assistant_speaking` is true

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

**Features**:
- WebSocket server for distributed agent deployment
- Zero-copy message handling with MessageEnvelope
- Dynamic topic subscription/publication
- Configurable queue management

**Topics**:
- Subscribes: `/voice_chunks`, `/camera/image_raw` (configurable)
- Publishes: `/audio_out`, `/cmd_vel`, `/llm_transcript` (configurable)

**Parameters**:
- `max_queue_size` (int, default 100): Maximum queue size before dropping messages
- `subscribed_topics` (list): Topics to bridge from ROS to agents
- `published_topics` (list): Topics to publish from agents to ROS
- `websocket_server.enabled` (bool): Enable WebSocket server
- `websocket_server.port` (int, default 8765): WebSocket server port

### simple_audio_player
A lightweight audio player specifically designed for playing AudioData messages at 24kHz from OpenAI Realtime API with real-time interruption support.

**Subscribed Topics**:
- `/audio_out` (audio_common_msgs/AudioData): Audio data to play
- `/interruption_signal` (std_msgs/Bool): Signal to immediately clear audio queue and stop playback

**Published Topics**:
- `assistant_speaking` (std_msgs/Bool): True when playing audio, False when stopped (respects namespace)

**Parameters**:
- `topic` (string, default "audio_out"): Input audio topic (relative, respects namespace)
- `sample_rate` (int, default 16000): Audio sample rate (standardized from 24kHz)
- `channels` (int, default 1): Number of audio channels
- `device` (int, default -1): Audio output device (-1 for default)

**Features**:
- Direct PyAudio playback without format conversion
- Automatic start/stop based on audio presence
- Queue-based buffering for smooth playback
- Assistant speaking status for echo suppression
- **Real-time interruption support**: Immediate audio cutoff via `interruption_signal`
- **Aggressive audio stopping**: Uses PyAudio `abort()` for instant termination without buffer drainage

### command_processor
A node that listens for command transcripts from AI agents and routes them to appropriate robot subsystems.

**Subscribed Topics**:
- `command_transcript` (std_msgs/String): Commands extracted by the AI agent

**Published Topics**:
- `/grunt1/arm_preset` (std_msgs/String): Arm preset commands (absolute path, no namespace)
- `/grunt1/behavior_command` (std_msgs/String): Behavior commands (absolute path, no namespace)

**Parameters**:
- `command_transcript_topic` (string, default "command_transcript"): Input topic for commands
- `arm_preset_topic` (string, default "/grunt1/arm_preset"): Output topic for arm commands
- `behavior_command_topic` (string, default "/grunt1/behavior_command"): Output topic for behavior commands

**Features**:
- Parses compound commands with @ separator (e.g., "tenhut@rightish")
- Routes arm presets to arm control system
- Routes behavior commands to behavior system
- Validates command syntax and modifiers
- Supports bearings as standalone pan commands

**Supported Commands**:
- **Arm Presets**: bumper, tenhut, lookup, lookout, reach, pan (with bearing modifier)
- **Behavior Commands**: stop, follow, track, sleep, wake, move, turn
- **Bearings** (standalone becomes pan@bearing): back-left, full-left, left, leftish, forward, rightish, right, full-right, back-right, back

### echo_suppressor
A fallback echo suppression solution that prevents audio feedback loops by muting microphone input while the assistant is speaking. This should only be used when hardware AEC or PulseAudio echo cancellation are not available.

**Subscribed Topics**:
- `audio` (audio_common_msgs/AudioStamped): Raw audio from microphone (respects namespace)
- `assistant_speaking` (std_msgs/Bool): Assistant speaking status (respects namespace)

**Published Topics**:
- `audio_filtered` (audio_common_msgs/AudioStamped): Filtered audio (muted when assistant speaks, respects namespace)

**Features**:
- Real-time audio gating based on assistant status
- Zero-latency passthrough when assistant is quiet
- Prevents feedback loops in open-mic scenarios

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

### OpenAI Realtime API (Fully Implemented)
WebSocket-based streaming with real-time voice conversations:

**Features**:
- ✅ Bidirectional audio streaming (16kHz input, 24kHz output)
- ✅ Real-time speech-to-text transcription
- ✅ Natural voice responses with multiple voice options
- ✅ Manual response triggering (server VAD limitation workaround)
- ✅ Session cost optimization through intelligent cycling
- ✅ Echo suppression for open-mic scenarios

**Models**: 
- `gpt-4o-realtime-preview` (recommended)
- `gpt-4o-realtime-preview-2024-12-17`

**Configuration**:
```yaml
openai_api_key: "sk-..."  # Or set OPENAI_API_KEY env var
model: "gpt-4o-realtime-preview"
voice: "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer
session_pause_timeout: 10.0  # Seconds before cycling session
```

### Google Gemini Live API (Planned)
Low-latency multimodal conversations:
- Models: `gemini-2.0-flash-exp`
- Features: Native audio support, 15-minute context window
- Status: Architecture ready, implementation pending

### Agent Architecture
The system uses a distributed agent-based approach:

**Key Components**:
- **ROS AI Bridge**: WebSocket server for agent connections
- **OpenAI Realtime Agent**: Manages WebSocket sessions with OpenAI
- **Session Manager**: Handles connection lifecycle and cost optimization
- **Context Manager**: Preserves conversation continuity across sessions
- **Named Prompt System**: Dynamic system prompts based on context

**Design Principles**:
- **Separation of Concerns**: ROS handles sensors/actuators, agents handle AI
- **Asyncio Concurrency**: Optimal for WebSocket and streaming APIs
- **Cost Optimization**: Aggressive session cycling on conversation pauses
- **Fault Tolerance**: Automatic reconnection and state recovery

### Performance Characteristics
- **Voice Detection**: < 50ms latency (Silero VAD)
- **Speech-to-Text**: Real-time streaming transcription
- **Response Generation**: 1-2 seconds for voice response
- **Audio Playback**: < 100ms from API to speakers
- **Echo Suppression**: < 50ms response time

### Dual-Agent Architecture
The system supports running multiple specialized agents simultaneously:

**Benefits**:
- **Separation of Concerns**: One agent for conversation, one for commands
- **Better Accuracy**: Specialized prompts for each task
- **Parallel Processing**: Both agents process the same audio simultaneously
- **No Conflicts**: Different output topics prevent interference

**Configuration**:
- Conversational agent publishes to: `/audio_out`, `/llm_transcript`
- Command agent publishes to: `/command_transcript`, `/command_detected`
- Both subscribe to: `/voice_chunks`

**Usage**:
```bash
# Launch dual agents
ros2 launch by_your_command oai_dual_agent.launch.py

# Monitor command detection
ros2 topic echo /command_transcript
ros2 topic echo /command_detected
```

## Troubleshooting

### No Audio Output
- Check that PyAudio is installed: `pip3 install pyaudio`
- Verify default audio device: `pactl info | grep "Default Sink"`
- Check topic has data: `ros2 topic echo /audio_out --no-arr`
- Save audio for debugging: `ros2 launch by_your_command oai_realtime.launch.py enable_voice_recorder:=true`

### Feedback/Echo Issues
- Ensure echo_suppressor is running: `ros2 node list | grep echo`
- Use headphones instead of speakers
- Increase distance between microphone and speakers
- Check `/assistant_speaking` topic: `ros2 topic echo /assistant_speaking`

### OpenAI Connection Issues
- Verify API key is set: `echo $OPENAI_API_KEY`
- Check agent logs for connection errors
- Ensure WebSocket connectivity (no proxy blocking wss://)
- Try standalone test: `python3 -m agents.oai_realtime.standalone_demo`

### Voice Not Detected
- Check VAD sensitivity in `config/config.yaml` (lower threshold = more sensitive)
- Monitor VAD output: `ros2 topic echo /voice_activity`
- Verify audio input: `ros2 topic hz /audio`

## Future Considerations

### Wake Word Attention Management
The OpenAI Realtime API's "always-on" nature makes it unsuitable for public deployments where background conversations trigger unwanted responses. A comprehensive analysis of implementing sleep/wake functionality using OpenWakeWord for attention management is documented in:

📄 **[Wake Word Attention Management Analysis](specs/analysis/wakeword_wakeword_partytime_excellent.md)**

This analysis covers multiple approaches for implementing robot attention states, ultimately recommending OpenWakeWord as a solution that combines VAD and wake word detection at the ROS level. The approach would enable natural sleep commands ("go to sleep", "be quiet") and wake phrases ("Hey Barney") while eliminating API costs and inappropriate responses during sleep periods.

## Contributing

Contributions are welcome! Please follow the development guidelines in `devrules/agentic_rules.md`.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- Silero Team for the excellent VAD model
- OpenAI for the Realtime API
- ROS 2 community for the audio_common package

