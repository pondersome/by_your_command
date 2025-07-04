# ByYourCommand (byc_)

ByYourCommand is a ROS 2 package for voice, camera, and video interactions. It uses Silero VAD for voice activity detection, Whisper for transcription, and connects to LLMs via an interaction graph for command parsing and chat.

## Installation

```bash
# In your ROS 2 workspace
colcon build --packages-select byc_
source install/setup.bash
```

## Configuration

Edit `config/config.yaml` to add your API keys and model paths.

## Usage

Launch all nodes:

```bash
ros2 launch byc_ bringup.launch.py
```

## Structure

- `config/`: API keys and model settings
- `install/requirements.txt`: Python dependencies
- `bringup/`: Launch files
- `silero_vad/`: Voice activity detection node
- `interactions/`: Whisper → LLM interaction node
- `devrules/`: Agentic coding rules evolution
