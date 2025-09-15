# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build the package (from ROS 2 workspace root) - always use --symlink-install for faster config changes
colcon build --packages-select by_your_command --symlink-install
source install/setup.bash

# Build audio_common_msgs if modified (with override warning)
colcon build --packages-select audio_common_msgs --allow-overriding audio_common_msgs --symlink-install

# Install Python dependencies
cd setup && chmod +x setup.sh && ./setup.sh

# Install ROS 2 dependencies
sudo apt install ros-$ROS_DISTRO-audio-common ros-$ROS_DISTRO-audio-common-msgs portaudio19-dev libportaudio2 ffmpeg
```

## Launch Commands

```bash
# Launch all nodes
ros2 launch by_your_command byc.launch.py

# Launch individual nodes
ros2 run by_your_command silero_vad_node
ros2 run by_your_command interaction_node  
ros2 run by_your_command voice_chunk_recorder

# Test scripts
ros2 run by_your_command test_utterance_chunks
ros2 run by_your_command test_recorder_integration
ros2 run by_your_command test_sleep_clap_integration
ros2 run by_your_command test_clap_detection

# Voice control commands
ros2 topic pub /response_cmd std_msgs/String "data: 'sleep'"  # Sleep command

# Remote mute/unmute control for Silero VAD
ros2 topic pub /voice_active std_msgs/Bool "data: false"  # Mute
ros2 topic pub /voice_active std_msgs/Bool "data: true"   # Unmute
```

## Architecture

ByYourCommand is a ROS 2 package for voice-controlled interactions using:

- **Audio Pipeline**: `audio_capturer_node` → `silero_vad_node` → `voice_chunk_recorder`/`interaction_node`
- **Voice Activity Detection**: Silero VAD model with configurable chunking and buffering
- **Real-time Conversational AI**: OpenAI Realtime API integration with intelligent session management
- **Multi-Agent System**: Distributed agents for conversation and command extraction
- **Interruption Handling**: Real-time user interruption support with immediate audio cutoff
- **Common Agent Components**: Shared modules for WebSocket communication, context management, and prompt loading

### Key Components

- `voice_detection/silero_vad_node.py`: Core VAD processing with frame-based buffering, pre-roll, and chunking
- `voice_detection/voice_chunk_recorder.py`: Records voice chunks to WAV files with utterance metadata
- `audio/simple_audio_player.py`: Real-time audio playback with interruption support
- `ros_ai_bridge/ros_ai_bridge.py`: WebSocket bridge for distributed agent deployment
- `agents/common/`: Shared components across all AI agents
  - `websocket_bridge.py`: WebSocket client interface with automatic reconnection
  - `context.py`: Conversation context management and preservation
  - `prompt_loader.py`: Dynamic prompt loading with recursive macro expansion
  - `conversation_monitor.py`: Real-time conversation state monitoring
  - `pause_detector.py`: Intelligent pause detection for session management
- `agents/oai_realtime/oai_realtime_agent.py`: OpenAI Realtime API integration with interruption handling
- `bringup/oai_realtime.launch.py`: Complete real-time conversational system launch
- `config/prompts.yaml`: System prompts with recursive macro definitions

### Data Flow

**Voice Input Processing**:
1. Audio capture via `audio_common` package
2. VAD processing with buffering and silence detection  
3. Voice chunk extraction with pre-roll and utterance metadata
4. WebSocket bridge forwards voice chunks to distributed agents
5. Real-time transcription and LLM response generation via OpenAI Realtime API

**Voice Output & Interruption**:
1. LLM audio responses streamed back through bridge to `/response_voice`
2. Audio player provides real-time playback with interruption monitoring
3. User speech during assistant speaking triggers three-stage interruption:
   - OpenAI API: `response.cancel` + `conversation.item.truncate`
   - Context: Clean conversation history preservation  
   - Audio: `interruption_signal` → PyAudio `abort()` for immediate cutoff

**Session & Context Management**:
1. Intelligent session cycling during conversation pauses for cost optimization
2. Text-based context preservation across session boundaries
3. Dynamic prompt loading with recursive macro expansion
4. Multi-agent coordination for simultaneous conversation and command processing

### Key Files

**Configuration**:
- `config/oai_realtime_agent.yaml`: OpenAI Realtime API agent configuration
- `config/prompts.yaml`: System prompts with recursive macro definitions
- `config/bridge_dual_agent.yaml`: Bridge configuration for multi-agent deployment
- `config/config.yaml`: Legacy VAD parameters and settings

**Agent Architecture**:
- `agents/common/`: Shared components across all AI agents
- `agents/oai_realtime/`: OpenAI Realtime API integration
- `ros_ai_bridge/ros_ai_bridge.py`: WebSocket bridge for distributed deployment

**Launch Systems**:
- `bringup/oai_realtime.launch.py`: Complete real-time conversational system
- `bringup/oai_dual_agent.launch.py`: Multi-agent conversation and command processing

**Dependencies**: `setup/requirements.txt`, `setup/setup.sh`
**Package definition**: `package.xml`, `setup.py`, `CMakeLists.txt`

## Development Rules

From `devrules/agentic_rules.md`:
- Include descriptive logging
- Keep nodes modular and decoupled
- Validate parameters at startup
- Follow ROS2 best practices
- Secure API key handling
- Write tests for new features