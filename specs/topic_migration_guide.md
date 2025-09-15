# Topic Name Migration Guide

**Created:** 2025-09-14
**Purpose:** Reference guide for the topic renaming refactoring
**Related:** See `topic_renaming_refactoring_prd.md` for full details

## Quick Reference: Old → New Topic Names

| Old Topic Name | New Topic Name | Purpose |
|---------------|---------------|---------|
| `voice_chunks` | `prompt_voice` | User voice input chunks from VAD |
| `text_input` | `prompt_text` | User text input (keyboard/external) |
| **N/A** | `prompt_transcript` | User voice transcript (NEW - was missing!) |
| `audio_out` | `response_voice` | Agent audio response |
| `audio_out_stamped` | `response_voice_stamped` | Timestamped agent audio |
| `llm_transcript` | `response_text` | Agent text response/transcript |
| `command_transcript` | `response_cmd` | Extracted commands/scene data |

## Naming Convention Principles

- **Prefix indicates direction**:
  - `prompt_*` = User input TO the agent
  - `response_*` = Agent output FROM the agent
- **Suffix indicates modality**:
  - `_voice` = Audio data
  - `_text` = Text data
  - `_transcript` = Speech-to-text conversion
  - `_cmd` = Command extraction

## Migration Instructions

### For External Tools/Scripts

If you have external tools that subscribe to the old topics, update them:

```bash
# Old way
ros2 topic echo /voice_chunks
ros2 topic echo /audio_out
ros2 topic echo /llm_transcript
ros2 topic echo /command_transcript

# New way
ros2 topic echo /prompt_voice
ros2 topic echo /response_voice
ros2 topic echo /response_text
ros2 topic echo /response_cmd
```

### For Rosbag Files

To play back old rosbag files with the new system, use topic remapping:

```bash
# Remap old topics to new ones during playback
ros2 bag play old_recording.bag \
  --remap voice_chunks:=prompt_voice \
  --remap text_input:=prompt_text \
  --remap audio_out:=response_voice \
  --remap llm_transcript:=response_text \
  --remap command_transcript:=response_cmd
```

### For Launch Files

Update any custom launch files:

```python
# Old parameters
parameters=[{
    'audio_out_topic': 'audio_out',
    'transcript_topic': 'llm_transcript',
    'command_transcript_topic': 'command_transcript'
}]

# New parameters
parameters=[{
    'response_voice_topic': 'response_voice',
    'response_text_topic': 'response_text',
    'response_cmd_topic': 'response_cmd'
}]
```

### For Configuration Files

Update YAML configurations:

```yaml
# Old configuration
subscribed_topics:
  - topic: "voice_chunks"
  - topic: "text_input"
published_topics:
  - topic: "audio_out"
  - topic: "llm_transcript"

# New configuration
subscribed_topics:
  - topic: "prompt_voice"
  - topic: "prompt_text"
published_topics:
  - topic: "response_voice"
  - topic: "response_text"
```

## New Features Enabled

### prompt_transcript Topic
The refactoring adds a previously missing feature - user voice transcripts are now published to ROS:

```bash
# Monitor what the user said (STT from agents)
ros2 topic echo /prompt_transcript

# This was previously only logged but never published
```

## Validation Commands

Test that the new topics are working:

```bash
# List all prompt/response topics
ros2 topic list | grep -E "(prompt_|response_)"

# Should see:
# /prompt_voice
# /prompt_text
# /prompt_transcript
# /response_voice
# /response_text
# /response_cmd

# Test the refactoring
cd /home/karim/ros2_ws
source install/setup.bash
python3 src/by_your_command/tests/test_topic_renaming.py
```

## Rollback Instructions

If you need to rollback to the old topic names:

```bash
# The refactoring was committed in phases
# To rollback, revert to the commit before the refactoring started:
git log --oneline | grep "topic renaming"

# Find the commit before the first refactoring commit and checkout
git checkout <commit-before-refactoring>

# Rebuild
colcon build --packages-select by_your_command --symlink-install
```

## Component Status

All components have been updated to use the new topic names:

### ✅ Updated Components
- **Agents**: OpenAI and Gemini agents (all variants)
- **Bridge**: ROS AI bridge with WebSocket interface
- **Audio Nodes**: VAD, audio player, audio converter, recorder
- **Command Processing**: Command processor node
- **Launch Files**: All launch configurations
- **Configuration Files**: All YAML configs
- **Test Scripts**: All 17 test files updated
- **Documentation**: README, CLAUDE.md, and PRDs updated

### 🆕 New Capabilities
- User voice transcripts now published (`prompt_transcript`)
- Clearer separation of user input vs agent output
- Consistent naming across entire system
- Better support for multi-agent architectures

## Support

For questions or issues related to this migration:
1. Check the full PRD: `specs/topic_renaming_refactoring_prd.md`
2. Review the test script: `tests/test_topic_renaming.py`
3. Check git history for the refactoring commits