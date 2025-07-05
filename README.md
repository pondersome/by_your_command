# ByYourCommand (byc_)

ByYourCommand is a ROS 2 package for voice, camera, and video interactions. It uses Silero VAD for voice activity detection, Whisper for transcription, and connects to LLMs via an interaction graph for command parsing and chat.

## Installation

Install Python dependencies:

```bash
cd setup
chmod +x setup.sh
./setup.sh
```

Build and source the package:

```bash
# From your ROS 2 workspace root
colcon build --packages-select byc_
source install/setup.bash
```

## Configuration

Edit `config/config.yaml` to add your API keys and model paths.

## Usage

Launch all nodes:

```bash
ros2 launch by_your_command bringup.launch.py
```

## Structure

- `config/`: API keys and model settings
- `setup/requirements.txt`: Python dependencies
- `setup/setup.sh`: Script to install Python dependencies
- `bringup/`: Launch files
- `silero_vad/`: Voice activity detection node
- `interactions/`: Whisper → LLM interaction node
- `devrules/`: Agentic coding rules evolution

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

### speech_only
A test node that subscribes to VAD output and writes speech chunks to WAV files.
**Parameters**:
- `output_dir` (string, default "/tmp"): Directory for output WAV files.
- `sample_rate` (int, default 16000): Audio sampling rate in Hz.

### interaction_node
A node that transcribes speech chunks using Whisper and processes commands with an LLM via OpenAI.
**Parameters**:
- `openai_api_key` (string): OpenAI API key.


