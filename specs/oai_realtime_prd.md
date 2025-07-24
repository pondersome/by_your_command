# OpenAI Realtime API Agent Product Requirements Document
## Session Management & Conversation Continuity

**Author**: Karim Virani  
**Version**: 1.0  
**Date**: July 2025

## 1. Problem Statement

### 1.1 The Cost Explosion Problem
Multimodal realtime API sessions can become exponentially expensive with session length due to the need to maintain large amounts of audio and image tokens throughout the session. Unlike text-only APIs where context is relatively small, realtime APIs must retain:
- All audio tokens from both user and assistant
- Video frame tokens (if using vision)
- The growing conversation context

**Example Cost Escalation**:
- Minute 1: $0.50 (base audio tokens)
- Minute 5: $3.00 (accumulated audio context)
- Minute 10: $12.00 (exponential growth)
- Minute 20: $50.00+ (unsustainable)

### 1.2 The Continuity Challenge
Simply terminating sessions breaks the conversational flow. Users expect:
- Seamless conversations without apparent interruptions
- Maintained context across extended interactions
- Natural handling of pauses and resumptions
- Consistent personality and memory of prior discussion

## 2. Solution Overview

### 2.1 Key Architectural Assumption: Local VAD Pre-filtering
The `/speech_chunks` topic receives audio that has **already been VAD-filtered** by a separate ROS2 node (e.g., `silero_vad_node`). This is critical because:

1. **True Silence Detection**: The bridge can detect actual pauses by the absence of messages on `/speech_chunks`, not by analyzing audio content
2. **Independent of LLM VAD**: While the Realtime API includes VAD, we don't depend on it for pause detection
3. **Network Efficiency**: Only meaningful speech is transmitted, not continuous audio streams
4. **Clean Pause Boundaries**: When `/speech_chunks` stops arriving, we know the user has actually stopped speaking

This architectural choice enables the aggressive pause-based session cycling to work reliably.

### 2.2 Intelligent Session Cycling
We implement a sophisticated session management system that:
1. **Leverages local VAD** to detect true speech pauses at the ROS2 level
2. **Preserves conversation continuity** through text-based context transfer
3. **Optimizes costs** by cycling WebSocket sessions during every pause
4. **Maintains illusion of single session** from user perspective

### 2.2 Key Innovation
The system gracefully tears down expensive multimodal sessions while preserving conversation state in text form, then spins up fresh sessions with injected context to continue seamlessly.

## 3. Detailed Requirements

### 3.0 ROS Bridge Integration & Message Serialization

#### 3.0.1 Zero-Copy Message Handling
The OpenAI Realtime agent receives ROS messages directly from the bridge via zero-copy `MessageEnvelope` objects:

```python
@dataclass
class MessageEnvelope:
    msg_type: str           # 'topic', 'service_request', 'service_response'
    topic_name: str         # ROS topic/service name
    raw_data: Any          # Raw ROS message object (zero-copy)
    ros_msg_type: str      # ROS message type for serialization dispatch
    timestamp: float       # Unix timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### 3.0.2 Agent Serialization Responsibilities
The bridge passes ROS messages without serialization. **The agent is responsible for**:

1. **Direct ROS Message Access**: Extract data from ROS message objects
2. **API-Specific Serialization**: Convert to OpenAI Realtime API format
3. **Format Optimization**: Handle different message types efficiently

```python
class OpenAIRealtimeSerializer:
    """Handle ROS → OpenAI Realtime API serialization"""
    
    def serialize_audio_data(self, envelope: MessageEnvelope) -> dict:
        """Convert ROS AudioData to OpenAI format"""
        if envelope.ros_msg_type == "audio_common_msgs/AudioData":
            # Direct access to ROS message fields
            audio_msg = envelope.raw_data
            pcm_bytes = np.array(audio_msg.int16_data, dtype=np.int16).tobytes()
            base64_audio = base64.b64encode(pcm_bytes).decode()
            
            return {
                "type": "input_audio_buffer.append",
                "audio": base64_audio
            }
    
    def serialize_image_data(self, envelope: MessageEnvelope) -> dict:
        """Convert ROS Image to OpenAI format (future)"""
        # Note: Current OpenAI Realtime API doesn't support images
        # This is prepared for future multimodal support
        pass
```

#### 3.0.3 Message Type Support Matrix

| ROS Message Type | OpenAI Realtime Support | Serialization Method |
|------------------|------------------------|---------------------|
| `audio_common_msgs/AudioData` | ✅ Full Support | PCM → base64 |
| `audio_common_msgs/AudioStamped` | ✅ Full Support | Extract audio, PCM → base64 |
| `sensor_msgs/Image` | ⏳ Future | Awaiting API support |
| `std_msgs/String` | ✅ Text Input | Direct text injection |
| `geometry_msgs/Twist` | ❌ Output Only | Command generation only |

#### 3.0.4 Performance Requirements for Serialization
- **Audio Serialization**: < 2ms for 20ms audio chunks (1:10 real-time ratio)
- **Memory Overhead**: < 1.5x original message size during serialization
- **Zero-Copy Access**: Direct field access without intermediate copying
- **Batch Processing**: Support for multiple message envelopes per batch

#### 3.0.5 Error Handling in Serialization
```python
class SerializationError(Exception):
    """Raised when ROS message cannot be serialized for API"""
    pass

async def safe_serialize(self, envelope: MessageEnvelope) -> Optional[dict]:
    """Safe serialization with error handling"""
    try:
        if envelope.ros_msg_type == "audio_common_msgs/AudioData":
            return self.serialize_audio_data(envelope)
        else:
            self.logger.warn(f"Unsupported message type: {envelope.ros_msg_type}")
            return None
    except Exception as e:
        self.logger.error(f"Serialization failed: {e}")
        self.metrics.serialization_errors += 1
        return None
```

#### 3.0.6 Bridge Integration Example
```python
class OpenAIRealtimeAgent:
    def __init__(self, bridge: ROSAIBridge):
        self.bridge_interface = bridge.register_agent_interface("openai_realtime")
        self.serializer = OpenAIRealtimeSerializer()
        self.websocket = None
        
    async def run(self):
        """Main agent loop - consumes from bridge, sends to OpenAI"""
        while True:
            # Get ROS message from bridge interface (zero-copy)
            envelope = await self.bridge_interface.get_inbound_message(timeout=1.0)
            if envelope is None:
                continue  # Timeout, check websocket or continue
            
            # Direct access to ROS message data
            if envelope.ros_msg_type == "audio_common_msgs/AudioData":
                # Serialize only when sending to API
                api_msg = self.serializer.serialize_audio_data(envelope)
                
                if self.websocket and api_msg:
                    await self.websocket.send(json.dumps(api_msg))
                    
            # Handle other message types...
            elif envelope.ros_msg_type == "std_msgs/String":
                # Direct text input injection
                await self.inject_text_input(envelope.raw_data.data)
    
    async def handle_llm_response(self):
        """Process responses from OpenAI and send back through bridge"""
        async for message in self.websocket:
            data = json.loads(message)
            
            if data.get("type") == "response.audio_transcript.done":
                # Send transcript back through bridge
                from std_msgs.msg import String
                transcript_msg = String(data=data["transcript"])
                self.bridge_interface.put_outbound_message(
                    "/llm_transcript", 
                    transcript_msg, 
                    "std_msgs/String"
                )
                
            elif data.get("type") == "response.audio.delta":
                # Send audio response back through bridge
                audio_data = base64.b64decode(data["delta"])
                audio_msg = self.create_audio_message(audio_data)
                self.bridge_interface.put_outbound_message(
                    "/audio_out", 
                    audio_msg, 
                    "audio_common_msgs/AudioData"
                )
```

#### 3.0.7 Agent Startup and Initialization
```python
async def main():
    \"\"\"Agent startup sequence\"\"\"
    # Initialize ROS2 and bridge
    rclpy.init()
    bridge = ROSAIBridge()
    
    # Start bridge in background thread
    bridge_task = asyncio.create_task(bridge.start_bridge())
    
    # Create and start agent
    agent = OpenAIRealtimeAgent(bridge)
    await agent.initialize()
    
    # Run agent and bridge concurrently
    await asyncio.gather(
        agent.run(),
        bridge_task
    )
```

#### 3.0.8 Required Dependencies
The agent implementation requires these additional dependencies beyond the bridge:
```python
# setup.py additions for OpenAI Realtime agent
install_requires=[
    'openai>=1.0.0',        # OpenAI Python SDK
    'websockets>=11.0',     # WebSocket client
    'aiohttp>=3.8.0',       # Async HTTP client
    'pydantic>=2.0',        # Data validation
    'numpy>=1.20.0',        # Audio processing
]
```

### 3.1 Session Lifecycle Management

#### 3.1.1 Session States
```
IDLE → CONNECTING → ACTIVE → CLOSING → CLOSED → IDLE
                        ↓                          ↑
                        └──────────────────────────┘
                         (aggressive cycling on pause)
```

#### 3.1.2 Core Strategy: Aggressive Pause-Based Cycling

**Key Insight**: Every pause is an opportunity to reset the expensive token accumulation by cycling the session. This works because `/speech_chunks` arrives pre-filtered by local VAD, allowing true pause detection.

**VAD Architecture**:
```
[Microphone] → [silero_vad_node] → /speech_chunks → [ros_ai_bridge] → [LLM API]
                     ↑                                        ↓
              (Local VAD filtering)                  (API also has VAD)
```

**Pause Detection**:
```python
class PauseDetector:
    def __init__(self, pause_timeout: float = 10.0):
        self.pause_timeout = pause_timeout
        self.last_message_time = None
        self.llm_response_complete = True
        
    async def monitor_pause(self, bridge_interface: AgentInterface):
        \"\"\"Monitor for pause conditions\"\"\"
        while True:
            # Try to get message with short timeout
            envelope = await bridge_interface.get_inbound_message(timeout=0.5)
            
            if envelope:
                self.last_message_time = time.time()
                # Process message...
            else:
                # Check if we're in pause condition
                if (self.last_message_time and 
                    self.llm_response_complete and
                    time.time() - self.last_message_time > self.pause_timeout):
                    
                    return True  # Pause detected, trigger session cycle
                    
    def mark_llm_response_complete(self):
        \"\"\"Called when LLM finishes speaking\"\"\"
        self.llm_response_complete = True
        
    def mark_llm_response_active(self):
        \"\"\"Called when LLM starts speaking\"\"\"
        self.llm_response_complete = False
```

**Key Implementation Details**:
- Agent monitors bridge interface for message timing, not raw audio
- Pause timer starts ONLY when both conditions met:
  1. No `MessageEnvelope` objects from bridge for `pause_timeout` seconds
  2. LLM has completed its response (tracked by response events)
- Default `session_pause_timeout`: 10 seconds
- This is **independent** of the OpenAI Realtime API's internal VAD

**Session Creation Triggers**:
- First `/speech_chunks` received when no session exists
- New `/speech_chunks` after pause-induced closure
- After any session rotation (pause-based or limit-based)

**Session Rotation Triggers**:
1. **Pause-Based (Primary)**: 
   - No activity for `session_pause_timeout` seconds
   - Aggressively close and prepare for fresh session
   - Resets the session duration timer!

2. **Limit-Based (Fallback)**:
   - Continuous audio streaming for `session_max_duration` (default: 120 seconds)
   - Token count exceeds `session_max_tokens` threshold
   - Cost estimate exceeds `session_max_cost` threshold

**Conversation Reset Triggers**:
- Total conversation timeout exceeded
- Explicit reset command (`/conversation_reset`)
- User change detected (voice/face recognition)

### 3.2 Prompt Provisioning System

#### 3.2.1 Named System Prompts
System prompts are stored in a YAML configuration file with support for:
- **Multiple named variants** for A/B testing
- **Version tracking** for prompt evolution
- **Conditional selection** based on context
- **Inheritance** for prompt variations

```python
class PromptManager:
    def __init__(self, config_path: str):
        self.prompts = self.load_prompts(config_path)
        self.active_prompt = None
        self.ab_test_state = {}
        
    def select_prompt(self, context: Dict) -> str:
        """Select appropriate prompt based on context and A/B tests"""
        # Check conditional rules first
        for rule in self.prompts['selection_rules']['conditional']:
            if self.evaluate_condition(rule['condition'], context):
                return self.prompts['prompts'][rule['prompt']]
        
        # Check A/B tests
        if self.ab_test_active():
            return self.select_ab_variant()
            
        # Default prompt
        return self.prompts['prompts'][self.default_prompt]
```

#### 3.2.2 User Prompt Prefixes
Prefixes can be dynamically injected based on conversation state:
- Context reminders
- Goal orientation
- Safety emphasis
- Previous command history

#### 3.2.3 Prompt Hot-Reloading
```python
class PromptWatcher:
    """Watch prompt file for changes and reload without restart"""
    async def watch_prompts(self):
        last_modified = 0
        while True:
            current_modified = os.path.getmtime(self.prompt_file)
            if current_modified > last_modified:
                await self.reload_prompts()
                last_modified = current_modified
            await asyncio.sleep(1.0)
```

### 3.2 Context Preservation Strategy

#### 3.2.1 What Gets Preserved
```python
class ConversationContext:
    text_buffer: List[Turn]  # All transcribed text
    system_state: Dict       # Robot state, goals, context
    active_tools: List[str]  # Currently enabled functions
    personality_modifiers: Dict  # Tone, style adjustments
    
class Turn:
    role: str  # 'user' or 'assistant'
    timestamp: float
    text: str  # Transcribed content
    metadata: Dict  # Commands, emotions, etc.
```

#### 3.2.2 What Gets Discarded
- Raw audio tokens (primary cost driver)
- Video frame history (except key frames)
- Intermediate processing states
- Redundant turn-taking markers

### 3.3 Aggressive Cycling Algorithm

#### 3.3.1 Pause Detection and Cycling
```python
async def monitor_for_pause_cycling(self):
    """Aggressively cycle session on any pause to reset token accumulation"""
    
    while self.conversation_active:
        # Check conditions for pause
        if (self.no_incoming_audio() and 
            self.llm_response_complete() and
            time.time() - self.last_activity > self.session_pause_timeout):
            
            # Opportunity to reset the expensive session!
            await self.cycle_session_on_pause()
            
            # Reset session timer - this is the key benefit!
            self.session_start_time = None
            
        await asyncio.sleep(0.1)

async def cycle_session_on_pause(self):
    """Quick session cycle during natural pause"""
    
    # 1. Capture current context
    context = await self.finalize_current_context()
    
    # 2. Close expensive session immediately
    await self.close_session()
    
    # 3. Mark as ready for new session
    self.session_state = SessionState.IDLE
    self.prepared_context = context
    
    # New session will be created when speech_chunks arrive
```

#### 3.3.2 Smart Session Timing
```python
class SessionTimer:
    """Manages session duration with pause-based resets"""
    
    def __init__(self):
        self.session_start_time = None
        self.continuous_speech_time = 0
        
    def start_session(self):
        """Called when new session created"""
        if self.session_start_time is None:
            # Fresh timer after pause cycle!
            self.session_start_time = time.time()
            self.continuous_speech_time = 0
    
    def check_limits(self):
        """Only enforced during continuous speech"""
        if self.session_start_time:
            elapsed = time.time() - self.session_start_time
            return elapsed > self.session_max_duration
        return False
```

**Benefits of Aggressive Pause Cycling**:
1. **Extended Conversations**: A 10-minute conversation with natural pauses might only accumulate 2 minutes of session time
2. **Cost Optimization**: Each pause resets token accumulation, preventing exponential cost growth
3. **Natural Flow**: Pauses align with conversation rhythm
4. **Lower Latency**: Pre-closed sessions mean faster reconnection

#### 3.3.3 Continuous Speech Handling
```python
async def handle_continuous_speech_rotation(self):
    """Handle rotation when user speaks continuously without pauses"""
    
    # This only triggers if no pause opportunities for recycling
    if self.session_timer.check_limits():
        self.log.warn("Hit session limit without pause - forcing rotation")
        
        # More complex rotation during active speech
        await self.prepare_hot_rotation()
        await self.execute_seamless_rotation()
        
        # Can't reset timer - speech is continuous
        # Next pause will reset it
```

#### 3.3.4 Session Lifecycle Example
```
Timeline with aggressive pause cycling:

[0:00-0:30] User speaks → Session A created, timer starts
[0:30-0:45] LLM responds
[0:45-0:55] Silence detected → 10 second pause timer starts
[0:55] Session A closed, context saved
      
[1:00] User speaks again → Session B created, timer RESET to 0:00!
[1:00-2:30] Extended conversation (90 seconds of fresh session time)
[2:30-2:40] Pause → Session B closed

[3:00] User continues → Session C created, timer RESET again!
[3:00-4:00] Another 60 seconds of conversation...

Result: 4 minutes of conversation, but no session exceeded 90 seconds!
Without aggressive cycling: Would hit 120-second limit at 2:00, forcing
rotation during active conversation.
```

#### 3.4.1 System Prompt Structure
```
[ORIGINAL SYSTEM PROMPT]

CONVERSATION CONTEXT:
You are continuing an ongoing conversation with the user. Here is the relevant context from our discussion so far:

[SUMMARIZED CONTEXT if > 2000 tokens]
or
[FULL TRANSCRIPT if < 2000 tokens]

Current robot state:
- Location: [location]
- Active task: [task]
- Recent commands: [commands]

Please continue naturally from where we left off. The user may continue speaking momentarily.
```

#### 3.4.2 Intelligent Summarization
When conversation history exceeds token limits:
1. Preserve all text from last 5 minutes verbatim
2. Summarize older content with focus on:
   - Decisions made
   - Goals established  
   - Key facts learned
   - Emotional context
3. Always preserve command history

### 3.5 Edge Case Handling

#### 3.5.1 Mid-Response Rotation
**Problem**: Session duration expires while assistant is speaking

**Solution**:
```python
async def handle_mid_response_rotation(self):
    # Option 1: Extend session briefly
    if self.response_nearly_complete():
        await self.extend_session(seconds=10)
    
    # Option 2: Graceful interruption
    else:
        # Cache partial response
        self.partial_response = self.current_response
        
        # In new session, prepend context:
        # "I was just explaining that [summary]..."
```

#### 3.5.2 Rapid Speech Switching
**Problem**: User rapidly switches between speaking and pausing

**Solution**:
- Implement debouncing on pause detection
- Minimum pause duration before session actions
- Hysteresis on resume triggers

#### 3.5.3 Network Interruptions
**Problem**: WebSocket disconnection during critical moments

**Solution**:
- Local transcription cache
- Automatic reconnection with context recovery
- User notification if context lost

### 3.6 Performance Requirements

#### 3.6.1 Timing Constraints
- **Session rotation**: < 500ms total
- **Context injection**: < 100ms
- **Audio gap during rotation**: < 200ms (imperceptible)
- **Transcription finalization**: < 1 second

#### 3.6.2 Resource Constraints
- **Memory per session**: < 100MB
- **Context buffer size**: < 50KB compressed
- **Max concurrent rotations**: 1 (serialize operations)

### 3.7 Monitoring & Metrics

#### 3.7.1 Cost Tracking
```python
class CostMonitor:
    def track_session_cost(self):
        return {
            'audio_tokens_in': self.audio_in_tokens,
            'audio_tokens_out': self.audio_out_tokens,
            'text_tokens': self.text_tokens,
            'video_frames': self.video_frame_count,
            'estimated_cost': self.calculate_cost(),
            'cost_per_minute': self.cost_rate()
        }
```

#### 3.7.2 Quality Metrics
- Conversation continuity score (1-10)
- User interruption frequency
- Context preservation accuracy
- Session rotation smoothness

## 4. Implementation TODOs

### 4.1 Phase 1: Core Session Management & Bridge Integration
- [ ] Create ROSAIBridge instance and register agent interface
- [ ] Implement OpenAI Realtime API serializer for audio messages (base64 PCM)
- [ ] Create WebSocket connection manager for OpenAI Realtime API
- [ ] Implement basic session lifecycle state machine (IDLE→CONNECTING→ACTIVE→CLOSING→CLOSED)
- [ ] Build conversation buffer with text-only persistence for context transfer
- [ ] Implement PauseDetector class for bridge interface message monitoring
- [ ] Add OpenAI API response handlers (audio deltas, transcripts, errors)
- [ ] Create message routing from bridge interface to OpenAI WebSocket
- [ ] Add cost estimation calculator with token tracking
- [ ] Add serialization performance monitoring and metrics

### 4.2 Phase 2: Intelligent Session Cycling
- [ ] Implement session cycling on pause detection (PauseDetector integration)
- [ ] Create conversation context extraction and text-based persistence  
- [ ] Build context injection system for new sessions (system prompt modification)
- [ ] Implement graceful session termination with context preservation
- [ ] Add session rotation metrics and timing optimization
- [ ] Create conversation summarizer for contexts > 2000 tokens
- [ ] Test aggressive cycling with various pause timeout settings

### 4.3 Phase 3: Advanced Features
- [ ] Implement predictive rotation (anticipate before limits)
- [ ] Add multi-provider rotation (OpenAI → Gemini fallback)
- [ ] Create conversation analytics dashboard
- [ ] Build automated testing for edge cases
- [ ] Add support for conversation branching/merging

### 4.4 Phase 4: Optimization
- [ ] Implement token usage prediction models
- [ ] Add dynamic timeout adjustment based on conversation flow
- [ ] Create cost optimization recommendations
- [ ] Build conversation quality scoring
- [ ] Implement A/B testing framework for parameters

## 5. Configuration Parameters

```yaml
session_management:
  # Pause Detection
  session_pause_timeout: 10.0  # seconds of silence before pause
  pause_debounce_time: 0.5     # avoid rapid pause/resume
  
  # Session Limits  
  session_max_duration: 120.0   # seconds before forced rotation
  session_max_tokens: 50000     # token limit before rotation
  session_max_cost: 5.00        # USD cost limit per session
  
  # Conversation Limits
  conversation_max_duration: 600.0  # total conversation timeout
  conversation_buffer_size: 10000   # max text tokens to preserve
  
  # Performance Tuning
  rotation_overlap_ms: 200      # overlap time during rotation
  audio_buffer_size: 1024       # samples to buffer during rotation
  
  # Quality Settings
  preserve_emotion_context: true
  summarize_after_minutes: 10
  min_utterance_for_persistence: 5  # words
```

## 6. Testing Scenarios

### 6.1 Cost Optimization Tests
1. **Token Accumulation Test**: Stream continuous audio for 30 minutes, verify costs stay linear
2. **Rotation Frequency Test**: Verify optimal rotation intervals for different conversation types
3. **Context Preservation Test**: Ensure no critical information lost during rotations

### 6.2 User Experience Tests
1. **Seamless Rotation Test**: Users should not notice session rotations
2. **Pause/Resume Test**: Natural conversation flow across pauses
3. **Long Conversation Test**: Multi-hour conversations remain coherent

### 6.3 Edge Case Tests
1. **Rapid Speaking Test**: Continuous speech at rotation boundary
2. **Network Failure Test**: Recovery from disconnections
3. **Context Overflow Test**: Graceful handling of very long conversations

## 7. Success Criteria

### 7.1 Cost Metrics
- 80% reduction in API costs for conversations > 5 minutes
- Linear cost scaling (not exponential) with conversation length
- < $0.50 per 10-minute conversation average

### 7.2 Quality Metrics
- 95% of users report conversations feel continuous
- < 2% context loss during rotations
- Zero mid-sentence cutoffs

### 7.3 Performance Metrics
- 99.9% successful rotation completion
- < 500ms rotation latency
- < 100ms audio gap during rotation

## 8. Risks & Mitigations

### 8.1 Technical Risks
- **Risk**: LLM may reference lost audio context
- **Mitigation**: Include disclaimer in context injection about audio history

- **Risk**: Rotation during critical command sequences
- **Mitigation**: Implement rotation inhibit flags for critical operations

### 8.2 User Experience Risks
- **Risk**: Noticeable personality shifts after rotation
- **Mitigation**: Preserve emotion and style markers in context

- **Risk**: Repeated information after rotation
- **Mitigation**: Track recent topics to avoid repetition

## 9. Future Enhancements

### 9.1 LangGraph Integration for Adversarial Checks
While the initial implementation uses sophisticated prompt engineering with realtime APIs, the architecture is designed to support LangGraph integration:

```python
class LangGraphAdversarialLayer:
    """Future: Parallel processing for safety and goal management"""
    
    async def process_parallel(self, user_input, llm_response):
        # Run adversarial checks without blocking realtime response
        safety_check = await self.check_command_safety(llm_response)
        goal_alignment = await self.check_goal_alignment(llm_response)
        
        # Can intervene if necessary
        if not safety_check.passed:
            await self.send_correction_to_user()
            await self.block_command_execution()
```

**Architecture for Dual Processing**:
```
User Input → Realtime API → Fast Response to User
     ↓                              ↓
     └→ LangGraph Agent →  Background Processing
                           - Safety validation
                           - Goal tracking
                           - Long-term planning
                           - Learning from interactions
```

### 9.2 Predictive Rotation
- ML model to predict optimal rotation points
- Natural pause detection for seamless rotations
- Conversation phase awareness

### 9.3 Advanced Context Management
- Hierarchical context summarization
- Semantic deduplication
- Multi-session memory banking
- Cross-conversation learning

### 9.4 Cost Optimization ML
- Per-user cost prediction
- Dynamic parameter tuning
- Conversation complexity scoring

### 9.5 Enhanced Prompt Management
- Database-backed prompt storage
- A/B test result tracking
- Automatic prompt optimization
- Multi-robot prompt sharing

## 10. Appendix: Example Session Flow

```
[0:00] User: "Hey robot, let's plan your route."
       → Session A CREATED, WebSocket connected
       → session_timer = 0:00
       
[0:30] User explains known waypoints to robot...
       → Token count: 2,000
       → session_timer = 0:30

[0:45] Robot: "Here is my planned route..."
[1:00] Robot finishes response
[1:00-1:10] SILENCE (no speech_chunks, LLM complete)
       → Pause timer triggered
       
[1:10] Aggressive cycle activated!
       → Context saved (2KB text)
       → Session A CLOSED
       → Token accumulation RESET
       
[1:30] User: "Ok, let's go!"
       → Session B CREATED with injected context
       → session_timer = 0:00 (RESET!)
       → Robot begins waypoint navigation
       
[1:30-3:00] Robot navigates waypoints
       → Only 90 seconds on session timer (not 3:00!)
       
[3:00-3:10] Natural pause
       → Session B CLOSED
       → Ready for Session C
       
[5:00] User: "STOP"
       → Session C CREATED
       → session_timer = 0:00 (RESET again!)
       
Total: 5+ minutes of conversation
Actual max session time: 90 seconds
Cost: Linear, not exponential!
```

### Key Insight Demonstrated:
By aggressively cycling on every pause, we transform an exponentially expensive 5-minute continuous session into multiple cheap 30-90 second sessions, while maintaining perfect conversation continuity. The user experiences one fluid conversation, but we've optimized costs by 80-90%.