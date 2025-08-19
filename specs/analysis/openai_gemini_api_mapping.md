# OpenAI Realtime vs Gemini Live API Mapping

## Purpose
This document maps OpenAI Realtime API calls to their Gemini Live equivalents, identifying gaps and architectural differences for agent implementation.

## API Call Mapping Table

Based on complete line-by-line analysis of `oai_realtime_agent.py` and `session_manager.py`:

| OpenAI API Call | Context/Purpose | Code Location | Gemini Equivalent | Notes |
|-----------------|-----------------|---------------|-------------------|-------|
| `session.update` | Configure session with prompts/voice | session_manager.py:170, 405, 487 | `client.live.send(config)` | Initial setup & prompt updates |
| `input_audio_buffer.append` | Stream audio chunks | serializers.py:46,56,116 + agent:297 | `session.send(audio_data)` | Real-time audio streaming |
| `input_audio_buffer.commit` | Finalize audio for processing | agent.py:306 | **Not needed** | Gemini processes continuously |
| `conversation.item.create` | Add text to conversation | serializers.py:73 + agent:799 | `session.send(text)` | Simpler text injection |
| `conversation.item.truncate` | Clean partial responses | agent.py:1124 | **Not available** | Need alternative approach |
| `response.create` | Trigger LLM response | agent.py:670,734,804 | **Not needed** | Automatic responses (or use proactive audio) |
| `response.cancel` | Cancel ongoing response | agent.py:1112 | `session.interrupt()` | User interruption stage 1 |
| `output_audio_buffer.clear` | Stop audio playback | agent.py:1117 | **Handled by interrupt** | Automatic with interrupt

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

## Architectural Differences

Based on analysis and Gemini Live API documentation:

### Session Management
- **OpenAI**: Stateful sessions with explicit buffer management
  - Requires `input_audio_buffer.commit` to process audio
  - Manual `response.create` to trigger responses
  - Session persists until explicitly closed
  
- **Gemini**: Stream-based with automatic processing
  - Continuous audio processing without commit
  - Automatic response generation (configurable with proactive audio)
  - Connection limits: 10min connection, 15min text/voice session, 2min with video
  
- **Impact**: Gemini needs aggressive reconnection strategy and different buffering approach

### Interruption Handling
- **OpenAI**: Three-stage interruption process
  1. `response.cancel` - Stop generation
  2. `output_audio_buffer.clear` - Stop playback
  3. `conversation.item.truncate` - Clean context
  
- **Gemini**: Single-call interruption
  - `session.interrupt()` handles everything
  - Less granular control but simpler implementation
  
- **Impact**: Simpler Gemini implementation but may need custom context cleanup

### Response Triggering
- **OpenAI**: Explicit control with `response.create`
- **Gemini**: Automatic responses (can use proactive audio for control)
- **Impact**: Different conversation flow patterns

### Context Management
- **OpenAI**: Can inject context via `conversation.item.create`
- **Gemini**: Send text/audio directly, history managed internally
- **Impact**: May need different approach for context preservation

## SessionManager Refactoring Analysis

The current `session_manager.py` is approximately 90% provider-agnostic:

### Reusable Components (Lines ~1-160, 194-end)
- WebSocket connection management
- Session state tracking (enum-based)
- Context management integration
- Prompt loading and macro expansion
- Session cycling logic
- Metrics and monitoring
- Connection retry logic

### Provider-Specific Parts (Lines ~169-188)
- Session configuration message format (`session.update` structure)
- Voice settings format
- Audio format specifications (pcm16)
- VAD/turn detection settings
- Model-specific parameters (whisper-1)

### Proposed Common SessionManager Design

```python
class BaseSessionManager(ABC):
    """Provider-agnostic session manager base class"""
    
    # Common implementation (90% of current code)
    def __init__(self, config: Dict):
        self.state = SessionState.IDLE
        self.context_manager = ContextManager(...)
        self.prompt_loader = PromptLoader(...)
        # ... all common initialization
    
    async def connect_session(self) -> bool:
        """Common connection logic with provider-specific hook"""
        url = await self._get_websocket_url()
        self.websocket = await websockets.connect(url)
        await self._configure_session()  # Provider-specific
        return True
    
    async def close_session(self) -> Optional[ConversationContext]:
        """Common cleanup logic"""
        # ... common cleanup
    
    # Provider-specific abstract methods
    @abstractmethod
    async def _configure_session(self):
        """Send provider-specific configuration"""
        pass
    
    @abstractmethod
    def _get_websocket_url(self) -> str:
        """Return provider-specific WebSocket URL"""
        pass
    
    @abstractmethod
    def _format_session_config(self, prompt: str, voice: str) -> Dict:
        """Format provider-specific session configuration"""
        pass

class OpenAISessionManager(BaseSessionManager):
    """OpenAI-specific implementation"""
    
    async def _configure_session(self):
        config_msg = {
            "type": "session.update",
            "session": { ... }  # OpenAI format
        }
        await self.websocket.send(json.dumps(config_msg))
    
    def _get_websocket_url(self) -> str:
        return "wss://api.openai.com/v1/realtime?model=..."

class GeminiSessionManager(BaseSessionManager):
    """Gemini-specific implementation"""
    
    async def _configure_session(self):
        config = { ... }  # Gemini format
        await self.client.live.send(config)
    
    def _get_websocket_url(self) -> str:
        # Gemini connection details
        pass
```

## Implementation Analysis Status

### Phase 1: OpenAI API Audit
- [x] Line-by-line review of `oai_realtime_agent.py`
- [x] Document all `websocket.send()` calls with message types
- [x] Identify context and purpose of each API call
- [x] Map to conversation flow (setup, streaming, interruption, cleanup)

### Phase 2: Gemini Live Research
- [x] Research Gemini Live API documentation (using provided references)
- [x] Find equivalent calls for each OpenAI operation
- [x] Identify missing features or architectural differences
- [x] Document alternative approaches where direct mapping isn't possible

### Phase 3: Architecture Differences
- [x] Compare session management approaches
- [x] Analyze audio streaming differences
- [x] Evaluate interruption capabilities
- [x] Document context/conversation management differences

## Template Agent Design

Based on this analysis, the OpenAI agent should serve as a template with:

1. **Common public interface methods** (listed above)
2. **Provider-specific private methods** (prefixed with `_send_to_openai`, `_send_to_gemini`, etc.)
3. **Shared response tracking and timeout logic**
4. **Unified session management patterns**

This will enable consistent behavior across different LLM providers while accommodating their unique API requirements.

## Gemini-Specific Features (No OpenAI Equivalent)

Based on the Gemini Live API documentation, these features have no direct OpenAI mapping:

### Proactive Audio
- **Purpose**: Model decides when to respond vs staying silent
- **Usage**: `config.generation_config.speech_config = SpeechConfig(proactive_audio=True)`
- **Benefit**: More natural conversation flow, reduces unnecessary responses
- **OpenAI Alternative**: Manual response triggering with `response.create`

### Built-in Video Support
- **Purpose**: Process video streams alongside audio
- **Limitation**: 2-minute session limit with video
- **OpenAI Alternative**: Would require separate vision API calls

### Automatic Session Limits
- **10-minute connection limit**: WebSocket disconnects automatically
- **15-minute text/voice session**: Content limit for audio-only
- **2-minute video session**: Strict limit when video included
- **OpenAI Alternative**: No documented limits, sessions persist

### Direct Tool/Function Calling
- **Purpose**: Live API can call functions during conversation
- **Usage**: Register tools with session configuration
- **OpenAI Alternative**: Function calling in separate API, not in Realtime

## Reference material for Gemini Live APIs
Gemini Live API Reference:
https://googleapis.github.io/python-genai/genai.html#module-genai.live

Getting started guide:
https://ai.google.dev/gemini-api/docs/live
https://ai.google.dev/gemini-api/docs/live-guide
https://ai.google.dev/gemini-api/docs/live-tools
https://ai.google.dev/gemini-api/docs/live-session
There's another page about ephemeral tokens, but we won't be using those - we're not calling from a browser. We can use the API key which is protected enough in this environment.

This is probably the best example to illustrate how to work with live streaming including audio and video. Importantly it shows how to enable user interruptions - which we want to support.
https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.py

There are some intricacies not covered. This is just a sample and doesn't really cover longer duration conversations or content reinjection (which in our case should include the cached last frame sent if fresh frames are not available). A connection in glive is limited to 10 minutes. A text/voice session is limited to 15 minutes. A session that includes video as well is limited to 2 minutes. So we will need a way to rebuild connections and sessions if we think we have an ongoing conversation. But this example does give a working async structure that can be  connected to our ros_ai_bridge instead of to OpenCV and direct audio. I think this example still uses some deprecated calls as documented in the API Reference. 

There's also a notebook with good examples here:
https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/multimodal-live-api/intro_multimodal_live_api_genai_sdk.ipynb
This includes a section on proactive audio which lets the model decide if it is being addressed and otherwise remains silent. This is not a feature that maps neatly onto something in the openai realtime api.




## Next Steps

### Completed ✅
1. ✅ Line-by-line analysis of OpenAI agent and session manager
2. ✅ Mapped all OpenAI API calls to Gemini equivalents
3. ✅ Documented architectural differences
4. ✅ Analyzed SessionManager for refactoring potential

### To Do
1. **Refactor SessionManager**
   - Extract base class with common functionality (90% of code)
   - Create OpenAISessionManager subclass
   - Design GeminiSessionManager subclass

2. **Implement Gemini Agent**
   - Use common agent interface methods
   - Handle Gemini-specific features (proactive audio, auto-reconnect)
   - Adapt to streaming model (no buffer commits)
   - Implement aggressive reconnection for time limits

3. **Test Cross-Provider Compatibility**
   - Verify common interface works for both providers
   - Test interruption handling differences
   - Validate context preservation approaches

4. **Optimize for Provider Strengths**
   - Use Gemini proactive audio for natural conversation
   - Leverage OpenAI's stateful sessions for complex contexts
   - Implement provider-specific optimizations