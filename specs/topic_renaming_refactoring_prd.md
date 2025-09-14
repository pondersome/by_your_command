# Topic Renaming Refactoring - PRD

**Created:** 2025-09-14  
**Status:** Draft  
**Author:** Karim Virani  
**Scope:** System-wide refactoring of ROS topic names for clarity and consistency

## 1. Executive Summary

This document outlines a comprehensive refactoring of topic names throughout the by_your_command system to improve clarity and consistency. The core change is adopting a directional naming convention that clearly distinguishes user inputs (prompt_*) from agent outputs (response_*), while also addressing terminology confusion around the overuse of "transcript."

## 2. Meta-Plan for Implementation

Given the scope of this refactoring, the work may span multiple development sessions. This meta-plan ensures continuity:

### 2.1 Context Preservation Strategy
1. **Create this PRD** - Comprehensive documentation to preserve context
2. **Commit current work** - Save unrelated changes before refactoring
3. **Export and compact** - Reduce conversation context
4. **Reload from PRD** - Use this document as primary context for continuation
5. **Feature branch** - Isolate changes from main branch

### 2.2 Work Phases
- **Phase 0**: Documentation and preparation (this PRD)
- **Phase 1**: Core agent and bridge changes
- **Phase 2**: Configuration and launch files
- **Phase 3**: ROS nodes and audio components
- **Phase 4**: Tests and utilities
- **Phase 5**: Documentation updates
- **Phase 6**: Migration and rollout

## 3. Problem Statement

### 3.1 Current Issues
1. **Terminology Confusion**: "transcript" is overused for different purposes
   - `llm_transcript` - Actually the agent's response text, not a transcript
   - `command_transcript` - Command extraction output, not really a transcript
   - True user voice transcript is never published

2. **Directional Ambiguity**: Unclear what's input vs output
   - `audio_out` - Generic name, doesn't indicate it's from assistant
   - `text_input` vs `llm_transcript` - Inconsistent naming patterns

3. **Missing Functionality**: User voice transcripts are logged but never published to ROS

4. **Mixed Content**: `command_transcript` contains both commands and scene descriptions

## 4. Proposed Solution

### 4.1 New Naming Convention

| Current Name | New Name | Purpose | Publisher |
|-------------|----------|---------|-----------|
| `voice_chunks` | `prompt_voice` | User voice input chunks | VAD/Bridge |
| `text_input` | `prompt_text` | User text input | External/Bridge |
| **NEW** | `prompt_transcript` | STT of user voice | Agents |
| `llm_transcript` | `response_text` | Agent text response | Both agent types |
| `audio_out` | `response_voice` | Agent audio response | Conversation agents only |
| `audio_out_stamped` | `response_voice_stamped` | Timestamped audio | Audio converter |
| `command_transcript` | `response_cmd` | Extracted commands/scene data | Command agents only |
| `conversation_id` | Keep as-is | Session/context identifier | Agents |
| `interruption_signal` | Keep as-is | Interruption control | Audio player |
| `voice_active` | Keep as-is | VAD control | Various |

### 4.2 Naming Principles
- **Prefix indicates direction**: `prompt_*` for user input, `response_*` for agent output
- **Suffix indicates modality**: `_voice`, `_text`, `_cmd`, `_transcript`
- **Clear agent differentiation**: Only conversation agents publish `response_voice`
- **Consistency**: All agent outputs start with `response_`

## 5. Affected Components

### 5.1 Agent Components
- `agents/oai_realtime/oai_realtime_agent.py`
- `agents/oai_realtime/oai_session_manager.py`
- `agents/gemini_live/gemini_live_agent.py`
- `agents/gemini_live/receive_coordinator.py`
- `agents/gemini_live/main.py`
- `agents/common/websocket_bridge.py`

### 5.2 Configuration Files
- `config/oai_realtime_agent.yaml`
- `config/oai_command_agent.yaml`
- `config/gemini_conversation_agent.yaml`
- `config/gemini_conversational_agent.yaml`
- `config/gemini_command_agent.yaml`
- `config/gemini_live_agent.yaml`
- `config/bridge_dual_agent.yaml`
- `config/bridge_gemini_vision.yaml`

### 5.3 Launch Files
- `bringup/oai_realtime.launch.py`
- `bringup/oai_dual_agent.launch.py`
- `bringup/gemini_live.launch.py`
- `bringup/gemini_dual_agent.launch.py`
- `bringup/gemini_vision.launch.py`

### 5.4 ROS Nodes
- `ros_ai_bridge/ros_ai_bridge.py`
- `audio/simple_audio_player.py`
- `audio/audio_data_to_stamped.py`
- `audio/voice_chunk_recorder.py`
- `audio/silero_vad_node.py`
- `nodes/command_processor.py`

### 5.5 Test Scripts
- `tests/test_gemini_with_image.py`
- `tests/test_gemini_hybrid.py`
- `tests/gemini/test_audio_communication.py`
- `tests/gemini/test_frame_timing.py`
- Various other test files

### 5.6 Documentation
- `README.md`
- `CLAUDE.md`
- `specs/*.md`
- Architecture diagrams
- Code comments

## 6. Implementation Plan

### 6.1 Phase 1: Core Agent Changes
**Goal**: Update agents to use new topic names and publish user transcripts

1. Update OpenAI agent:
   - Change topic name constants
   - Add `prompt_transcript` publishing
   - Update subscription lists

2. Update Gemini agents:
   - Change topic name constants
   - Add `prompt_transcript` publishing
   - Fix command agent to use `response_cmd`
   - Ensure only conversation agent publishes `response_voice`

3. Update common components:
   - WebSocket bridge interface
   - Message routing logic

**Testing**: Manual testing with single agents

### 6.2 Phase 2: Configuration Updates
**Goal**: Update all YAML configuration files

1. Agent configurations:
   - Update `audio_out_topic` → `response_voice_topic`
   - Update `transcript_topic` → `response_text_topic`
   - Update `command_transcript` → `response_cmd`

2. Bridge configurations:
   - Update published_topics lists
   - Update subscribed_topics lists

**Testing**: Launch with updated configs

### 6.3 Phase 3: Bridge and Infrastructure
**Goal**: Update ROS AI bridge and core infrastructure

1. Update ros_ai_bridge.py:
   - Topic name mappings
   - Subscription handling
   - Publishing logic

2. Update namespace/prefix handling:
   - Ensure new names work with namespacing
   - Test with prefixed deployments

**Testing**: Multi-agent deployments

### 6.4 Phase 4: Audio Components
**Goal**: Update audio-related nodes

1. Update default topic parameters:
   - `simple_audio_player.py`: `/audio_out` → `/response_voice`
   - `audio_data_to_stamped.py`: Update both input and output topics
   - `voice_chunk_recorder.py`: Update input topic default

2. Update subscription creation:
   - Change hard-coded topic names
   - Update parameter declarations

**Testing**: Audio pipeline end-to-end

### 6.5 Phase 5: Launch Files
**Goal**: Update all launch files with new topic names

1. Update parameter passing:
   - Topic remappings
   - Configuration file parameters

2. Update any hard-coded topic names in launch logic

**Testing**: Full system launches

### 6.6 Phase 6: Tests and Utilities
**Goal**: Update test infrastructure

1. Update test scripts:
   - Topic names in test assertions
   - Mock publishers/subscribers

2. Update debug utilities:
   - Topic monitoring scripts
   - Debug output formatting

**Testing**: Run test suite

### 6.7 Phase 7: Documentation
**Goal**: Update all documentation

1. Update user-facing docs:
   - README.md usage examples
   - CLAUDE.md guidance

2. Update technical docs:
   - PRDs and specs
   - Architecture diagrams
   - API documentation

3. Update code comments:
   - Inline documentation
   - TODO comments referencing old names

**Testing**: Documentation review

## 7. Testing Strategy

### 7.1 Incremental Testing
Each phase includes specific testing to catch issues early:
- **Unit**: Individual component functionality
- **Integration**: Component interactions
- **System**: Full pipeline testing
- **Regression**: Ensure existing features still work

### 7.2 Test Scenarios
1. **Single agent deployment**: Verify basic functionality
2. **Dual agent deployment**: Verify no cross-talk or confusion
3. **Namespaced deployment**: Verify prefix/namespace handling
4. **Audio pipeline**: Voice in → transcription → response → audio out
5. **Command extraction**: Voice → command → execution
6. **Text input**: Text → response without voice
7. **Interruption handling**: Still works with new topic names

### 7.3 Backwards Compatibility Testing
- Test with any external tools that might subscribe to old topics
- Verify rosbag playback strategies

## 8. Migration Strategy

### 8.1 Feature Branch Development
1. Create feature branch: `feature/topic-renaming`
2. Implement changes incrementally
3. Test thoroughly before merging

### 8.2 Migration Path
1. **Option A - Clean Break**:
   - All changes at once
   - Clear migration instructions
   - Update external tools simultaneously

2. **Option B - Compatibility Period** (if needed):
   - Publish to both old and new topics temporarily
   - Deprecation warnings for old topics
   - Remove old topics in future release

### 8.3 Rollback Plan
- Feature branch allows easy rollback
- Document any configuration changes needed
- Keep backup of original configurations

## 9. Validation Checklist

### 9.1 Functional Validation
- [ ] User voice is transcribed and published as `prompt_transcript`
- [ ] Conversation agents publish `response_text` and `response_voice`
- [ ] Command agents publish only `response_cmd`
- [ ] Text input works via `prompt_text`
- [ ] Audio playback works from `response_voice`
- [ ] Command extraction produces valid commands
- [ ] Interruption handling still functions
- [ ] Namespacing works correctly

### 9.2 Non-Functional Validation
- [ ] No performance degradation
- [ ] No additional latency
- [ ] Memory usage unchanged
- [ ] CPU usage unchanged
- [ ] Network traffic patterns similar

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking external integrations | High | Document changes clearly, provide migration guide |
| Missing a hard-coded reference | Medium | Comprehensive grep searches, thorough testing |
| Rosbag incompatibility | Low | Document remapping strategy for playback |
| Configuration errors | Medium | Validation scripts, clear error messages |
| Incomplete documentation updates | Low | Systematic review of all docs |

## 11. Success Criteria

1. **Clarity**: New topic names are self-explanatory
2. **Consistency**: All topics follow the naming convention
3. **Completeness**: User transcripts are now published
4. **Functionality**: All existing features still work
5. **Performance**: No degradation in system performance
6. **Documentation**: All references updated

## 12. Future Considerations

### 12.1 Additional Topics
Consider renaming in future:
- Stream-based topics for continuous audio
- Debug/diagnostic topics
- Metrics/telemetry topics

### 12.2 Topic Discovery
Implement topic discovery patterns that work with new naming:
- Auto-discovery based on prefixes
- Dynamic subscription based on agent role

### 12.3 Validation Tools
Create tools to:
- Validate topic naming compliance
- Detect old topic names in use
- Migration assistance scripts

## 13. Notes for Continuation

When resuming work after a compact cycle:
1. Review this PRD for context
2. Check git status for current branch and changes
3. Identify which phase was last completed
4. Continue with next phase testing
5. Update this PRD with any discoveries or changes

## 14. Command Reference

### Useful Commands for Refactoring
```bash
# Find all occurrences of old topic names
grep -r "llm_transcript" --include="*.py" --include="*.yaml"
grep -r "audio_out" --include="*.py" --include="*.yaml"
grep -r "command_transcript" --include="*.py" --include="*.yaml"
grep -r "voice_chunks" --include="*.py" --include="*.yaml"
grep -r "text_input" --include="*.py" --include="*.yaml"

# Test topic publishing
ros2 topic echo /response_text
ros2 topic echo /prompt_transcript

# Monitor all topics
ros2 topic list | grep -E "(prompt_|response_)"
```

## 15. Conclusion

This refactoring improves system clarity by establishing consistent, self-documenting topic names. The phased approach minimizes risk while ensuring comprehensive updates across the entire system. The meta-plan ensures work can continue effectively across multiple development sessions.