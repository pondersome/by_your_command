# OpenAI Realtime vs Gemini Live API Mapping

## Purpose
This document maps OpenAI Realtime API calls to their Gemini Live equivalents, identifying gaps and architectural differences for agent implementation.

## API Call Mapping Table

Based on analysis of `oai_realtime_agent.py`, the following OpenAI API calls are used:

| OpenAI API Call | Context/Purpose | Code Location | Gemini Equivalent | Notes |
|-----------------|-----------------|---------------|-------------------|-------|
| `session.update` | Configure session with prompts/voice | session_manager.py | **[To be researched]** | Initial setup |
| `input_audio_buffer.append` | Stream audio chunks | _process_bridge_messages() | **[To be researched]** | Real-time audio streaming |
| `input_audio_buffer.commit` | Finalize audio for processing | End of utterance | **[To be researched]** | Trigger transcription |
| `conversation.item.create` | Add text/context to conversation | _send_text_to_openai() | **[To be researched]** | Text input injection |
| `response.create` | Trigger LLM response generation | After text input | **[To be researched]** | Manual response trigger |
| `response.cancel` | Interrupt ongoing response | Interruption system | **[To be researched]** | User interruption |
| `conversation.item.truncate` | Clean partial responses from context | After interruption | **[To be researched]** | Context cleanup |

## Common Interface Methods

Methods that should be consistent across all agent implementations:

### Core Communication Methods
```python
async def send_text_to_llm(text: str) -> bool:
    """Send text input to the LLM as a conversation item"""
    
async def send_audio_to_llm(audio_data: bytes) -> bool:
    """Send audio data to the LLM for processing"""
    
async def interrupt_response() -> bool:
    """Interrupt currently generating response"""
    
async def clear_context() -> bool:
    """Clear conversation context/history"""
```

### Session Management
```python
async def create_session() -> bool:
    """Establish connection with LLM provider"""
    
async def close_session() -> bool:
    """Cleanly close LLM session"""
    
def is_session_active() -> bool:
    """Check if session is ready for communication"""
```

### Response Tracking
```python
def setup_response_expectations():
    """Set up tracking for expected responses"""
    
def clear_response_expectations():
    """Clear response tracking (timeout recovery)"""
```

## Implementation Analysis Needed

The following analysis tasks remain:

### Phase 1: OpenAI API Audit
- [ ] Line-by-line review of `oai_realtime_agent.py`
- [ ] Document all `websocket.send()` calls with message types
- [ ] Identify context and purpose of each API call
- [ ] Map to conversation flow (setup, streaming, interruption, cleanup)

### Phase 2: Gemini Live Research
- [ ] Research Gemini Live API documentation
- [ ] Find equivalent calls for each OpenAI operation
- [ ] Identify missing features or architectural differences
- [ ] Document alternative approaches where direct mapping isn't possible

### Phase 3: Architecture Differences
- [ ] Compare session management approaches
- [ ] Analyze audio streaming differences
- [ ] Evaluate interruption capabilities
- [ ] Document context/conversation management differences

## Template Agent Design

Based on this analysis, the OpenAI agent should serve as a template with:

1. **Common public interface methods** (listed above)
2. **Provider-specific private methods** (prefixed with `_send_to_openai`, `_send_to_gemini`, etc.)
3. **Shared response tracking and timeout logic**
4. **Unified session management patterns**

This will enable consistent behavior across different LLM providers while accommodating their unique API requirements.

## Next Steps

1. Complete the line-by-line analysis of OpenAI agent
2. Research Gemini Live API equivalents
3. Update this mapping table with findings
4. Use insights to design the Gemini agent architecture