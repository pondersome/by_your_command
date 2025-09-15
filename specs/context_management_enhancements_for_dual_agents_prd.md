# Context Management Enhancements for Dual Agents - PRD

> **Note: This document predates the topic renaming refactoring (2025-09-14). Some topic names mentioned here have been updated in the implementation. See `topic_renaming_refactoring_prd.md` for current naming.**


**Created:** 2025-09-14  
**Status:** Draft  
**Author:** Karim Virani  
**Scope:** Enhanced context synchronization between conversation and command agents

## 1. Executive Summary

This document outlines enhancements to address context divergence between dual agents (conversation and command extraction) in the by_your_command system. The core problem is that each agent maintains isolated conversation context, leading to inconsistent responses and transcriptions over time.

The proposed solution introduces agent sibling awareness with the conversation agent serving as the authoritative source for user transcriptions, while maintaining the bridge as a pure relay and ensuring agents can operate standalone.

## 2. Background - Current Prompt and Context Architecture

### 2.1 Prompt Loading and Assembly Analysis

Based on investigation of the current codebase, the prompt and context management system works as follows:

#### System Prompt Assembly
- **When Assembled**: System prompts are assembled **once per session creation**, not per turn
  - During `connect_session()` → `_configure_session()` in both OpenAI and Gemini agents
  - This happens when establishing a new WebSocket connection or cycling sessions
- **Assembly Process**:
  - `PromptLoader` loads prompts.yaml at startup
  - Macros are expanded **at load time** (in `_load_prompts()`)
  - When session is created:
    - Agent gets prompt by ID or uses selection rules
    - `ContextManager.build_system_prompt()` adds conversation context if available
    - Final prompt is sent as part of session configuration

#### User Prompt Prefixes
- **Current Status**: **NOT ACTIVELY USED** in production agents
  - Defined in prompts.yaml (`user_prompt_prefixes` section)
  - Methods exist in `PromptLoader` (`get_user_prefix()`, `build_full_prompt()`)
  - Only used in test files, not in actual agent code
- **Intended Design**:
  - Would prepend context to user messages
  - Supports template variables like `{last_topic}`, `{last_command}`
  - Could be used per-turn to inject context

#### Key Implications
- System prompts are static for the entire session duration
- No per-turn prompt modification currently happens
- Macro expansion happens at load time, not runtime
- Context preservation happens through `ContextManager` appending to base prompt

### 2.2 Current Context Tracking Implementation

#### OpenAI Agents
- Uses `ContextManager` to track conversation turns
- `add_conversation_turn()` called when transcriptions are received:
  - User transcripts: line 666 in oai_realtime_agent.py
  - Assistant transcripts: line 694 in oai_realtime_agent.py
- Context injected during session creation via `build_system_prompt()`
- Context preserved across session cycles (for cost optimization)

#### Gemini Agents
- Same `ContextManager` infrastructure
- `add_conversation_turn()` in receive_coordinator.py:
  - User transcripts: line 339
  - Assistant transcripts: line 392
- Context injected identically during session creation
- **Critical difference**: Command agents skip adding assistant turns (line 391-392)

## 3. Problem Statement

### 3.1 Context Divergence

Each agent maintains its own isolated context, leading to divergence over multiple conversation turns:

1. **Turn 1**: User says "move forward"
   - Conversation agent: "I'll move forward for you"
   - Command agent: Extracts `move@forward`

2. **Turn 2**: User says "do that again"
   - Conversation agent (has context): Knows "that" = move forward
   - Command agent (different context): May not understand the reference

3. **Result**: Over time, agents have completely different conversation histories

### 3.2 Transcription Divergence

When context diverges significantly, the same audio input can be transcribed differently:
- User says: "Do that again"
- Conversation agent: Transcribes as "Do that again" (knows context)
- Command agent: Might transcribe as "Do the scan" (different context)

Both transcriptions are published to the same topic, creating ambiguity.

### 3.3 Command Acknowledgment Mismatch

The conversation agent often returns different command acknowledgments than what the command agent extracted, confusing users about what the robot will actually do.

## 4. Requirements

### 4.1 Functional Requirements
1. **Context Synchronization**: Agents should have consistent understanding of conversation history
2. **Authoritative Transcription**: Single source of truth for what the user said
3. **Standalone Operation**: Each agent must work independently without waiting for siblings
4. **Bridge Simplicity**: Bridge must remain a pure relay without business logic

### 4.2 Non-Functional Requirements
1. **Low Latency**: Context sharing should not add significant latency
2. **Scalability**: Solution should support future modalities and N-agent scenarios
3. **Maintainability**: Clear separation of concerns between bridge and agents
4. **Compatibility**: Preserve ability to swap bridge implementation

## 5. Options Considered

### Option A: Dedicated Transcription Agent
- **Approach**: One agent does all transcription, others consume
- **Pros**: Single source of truth, no divergence
- **Cons**: Adds latency, single point of failure
- **Rejected**: Too much architectural change, latency penalty

### Option B: Leader Election
- **Approach**: First agent to transcribe "wins"
- **Pros**: No dedicated agent needed
- **Cons**: Race conditions, complex coordination
- **Rejected**: Complexity without clear benefit

### Option C: Transcription Reconciliation
- **Approach**: Bridge or coordinator reconciles different transcriptions
- **Pros**: Could provide best transcription
- **Cons**: Complex, adds latency, violates bridge simplicity requirement
- **Rejected**: Adds business logic to bridge

### Option D: Accept Divergence with Tagging
- **Approach**: Tag all messages with agent ID, accept multiple versions
- **Pros**: Honest about divergence, simple
- **Cons**: Doesn't solve the problem, just makes it visible
- **Rejected**: Doesn't improve user experience

### Option E: Conversation Agent as Authority (CHOSEN)
- **Approach**: Conversation agent's transcription is authoritative for context
- **Pros**: 
  - Leverages better context for accuracy
  - Natural authority (primary user interface)
  - Simple implementation
  - No latency for commands
- **Cons**: Requires agent awareness of siblings
- **Selected**: Best balance of simplicity and effectiveness

## 6. Proposed Solution

### 6.1 Agent Sibling Awareness

Agents become aware of their role and potential siblings:

```python
# In agent config
agent_role: "conversation"  # or "command"
sibling_agents: ["command_agent_id"]  # optional
```

### 6.2 Message Tagging

Agents tag their messages with metadata:

```python
transcript_data = {
    "data": transcript,
    "metadata": {
        "agent_id": self.agent_id,
        "agent_role": self.agent_role,
        "timestamp": time.time()
    }
}
```

Bridge remains a pure relay - just passes enriched messages through.

### 6.3 Subscription Patterns

Agents subscribe to each other's output topics:

```python
# Conversation agent subscribes to:
- voice_chunks
- command_transcript  # From command agent

# Command agent subscribes to:
- voice_chunks  
- llm_transcript  # From conversation agent
- user_transcript  # For authoritative transcription
```

### 6.4 Context Incorporation Logic

```python
def handle_transcript(self, envelope):
    metadata = envelope.data.get("metadata", {})
    
    # Check if this is from conversation agent (authoritative)
    if metadata.get("agent_role") == "conversation":
        # Use for context
        self.context_manager.add_turn("user", transcript)
    else:
        # Log for analysis but don't add to context
        self.logger.debug(f"Alt transcript: {transcript}")
```

### 6.5 Standalone Operation

Agents check for sibling configuration but don't require it:

```python
def __init__(self):
    self.sibling_agents = config.get("sibling_agents", [])
    self.has_siblings = len(self.sibling_agents) > 0
    
    # Subscribe to sibling topics only if configured
    if self.has_siblings:
        subscriptions.extend(self._get_sibling_subscriptions())
```

## 7. Implementation Plan

### 7.1 Prerequisites
- **Complete topic renaming** (discussed in separate planning session)
- This ensures clean topic namespace before adding cross-subscriptions

### 7.2 Phase 1: Message Tagging
- Add metadata to all agent-published messages
- Include agent_id and agent_role
- No behavior changes yet

### 7.3 Phase 2: Sibling Configuration
- Add agent_role and sibling_agents to config files
- Implement conditional subscription logic
- Still no context sharing

### 7.4 Phase 3: Cross-Subscription
- Conversation agent subscribes to command_transcript
- Command agent subscribes to llm_transcript
- Add filtering to prevent self-loops

### 7.5 Phase 4: Context Incorporation
- Implement authoritative transcription logic
- Add context updates from sibling outputs
- Test convergence over multiple turns

### 7.6 Phase 5: User Prefix Activation (Optional)
- Activate the dormant user prefix system
- Use for injecting recent cross-agent context
- Template variables from shared context

## 8. Technical Design

### 8.1 Message Format

```json
{
  "topic": "/user_transcript",
  "data": "move forward please",
  "metadata": {
    "agent_id": "gemini_conversational",
    "agent_role": "conversation",
    "timestamp": 1736819200.123,
    "conversation_id": "conv_abc123",
    "turn_number": 5
  }
}
```

### 8.2 Configuration Schema

```yaml
# conversation_agent.yaml
agent_config:
  agent_id: "oai_conversation"
  agent_role: "conversation"
  sibling_agents: ["oai_command"]
  
  subscriptions:
    - topic: "prompt_voice"
    - topic: "response_cmd"
      filter:
        agent_role: "command"  # Only from command agents
        
  context_sharing:
    accept_authoritative_transcripts: false  # Is authority
    provide_authoritative_transcripts: true
```

### 8.3 Fallback Mechanisms

1. **Missing Sibling**: Continue normal operation, no waiting
2. **Delayed Transcription**: Use own transcription, update on next turn
3. **Metadata Missing**: Treat as legacy message, log warning

## 9. Edge Cases

### 9.1 Command Agent Finishes First
- Publishes its transcript immediately
- ROS systems record it
- Conversation agent's authoritative version updates context later
- Commands execute without delay

### 9.2 Conversation Agent Timeout
- Command agent proceeds with its own transcription
- Context may diverge slightly
- Resynchronizes on next successful turn

### 9.3 Solo Agent Deployment
- Agent detects no siblings configured
- Operates normally without cross-subscriptions
- No performance impact

## 10. Testing Strategy

### 10.1 Unit Tests
- Message tagging validation
- Metadata parsing
- Sibling detection logic

### 10.2 Integration Tests
- Dual agent context convergence
- Transcription authority selection
- Solo agent operation

### 10.3 System Tests
- Multi-turn conversations with context references
- Command extraction consistency
- Latency measurements

## 11. Success Metrics

1. **Context Convergence**: Agents have consistent context within 2 turns
2. **Transcription Consistency**: 95% agreement on transcriptions
3. **Command Alignment**: Conversation acknowledgments match extracted commands
4. **Latency Impact**: < 50ms additional latency for context sharing
5. **Standalone Success**: 100% functionality when deployed solo

## 12. Future Enhancements

### 12.1 Multi-Modal Agents
- Vision agents sharing scene descriptions
- Gesture recognition agents sharing interpretations
- Emotion detection agents sharing sentiment

### 12.2 Dynamic Authority
- Authority changes based on context type
- Vision agent authoritative for "what do you see"
- Command agent authoritative for robot state

### 12.3 Context Summarization
- Intelligent summarization for long conversations
- Selective context based on relevance
- Token optimization for LLM calls

## 13. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Message loops | High | Agent-side filtering, never subscribe to own outputs |
| Increased complexity | Medium | Phased rollout, extensive testing |
| Bridge coupling | Low | Metadata in messages, not bridge logic |
| Version skew | Medium | Graceful handling of missing metadata |

## 14. Decision Log

1. **Bridge remains pure relay**: Preserves swapability with other ROS bridges
2. **Conversation agent as authority**: Best context leads to best transcriptions
3. **Sibling awareness over coordination**: Simple, no synchronization required
4. **Eventual consistency over perfect sync**: Accepts some divergence for low latency

## 15. Conclusion

This enhancement addresses the fundamental context divergence problem in dual-agent deployments while maintaining system simplicity and performance. By making agents aware of their siblings without requiring tight coupling, we achieve better consistency without sacrificing the ability to run agents independently.

The phased implementation plan allows for gradual rollout with validation at each step, minimizing risk while delivering incremental value.