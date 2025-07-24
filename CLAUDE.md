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
```

## Architecture

ByYourCommand is a ROS 2 package for voice-controlled interactions using:

- **Audio Pipeline**: `audio_capturer_node` → `silero_vad_node` → `voice_chunk_recorder`/`interaction_node`
- **Voice Activity Detection**: Silero VAD model with configurable chunking and buffering
- **Speech Processing**: Whisper transcription → OpenAI LLM interaction (planned)

### Key Components

- `voice_detection/silero_vad_node.py`: Core VAD processing with frame-based buffering, pre-roll, and chunking
- `voice_detection/voice_chunk_recorder.py`: Records voice chunks to WAV files with utterance metadata
- `interactions/interaction_node.py`: Whisper transcription and LLM integration (incomplete)
- `bringup/byc.launch.py`: Launch file coordinating audio capture, VAD, and voice processing
- `config/config.yaml`: VAD parameters, API keys, and model settings

### Data Flow

1. Audio capture via `audio_common` package
2. VAD processing with buffering and silence detection  
3. Voice chunk extraction with pre-roll and utterance metadata
4. Transcription and LLM processing (in development)

### Key Files

- Configuration: `config/config.yaml` (API keys, VAD tuning)
- Dependencies: `setup/requirements.txt`, `setup/setup.sh`
- Package definition: `package.xml`, `setup.py`, `CMakeLists.txt`

## Development Rules

From `devrules/agentic_rules.md`:
- Include descriptive logging
- Keep nodes modular and decoupled
- Validate parameters at startup
- Follow ROS2 best practices
- Secure API key handling
- Write tests for new features