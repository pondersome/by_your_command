# OpenAI Realtime API Agent Product Requirements Document

> **Note: This document predates the topic renaming refactoring (2025-09-14). Some topic names mentioned here have been updated in the implementation. See `topic_renaming_refactoring_prd.md` for current naming.**

## Real-Time Conversational AI with Intelligent Session Management

**Author**: Karim Virani  
**Version**: 2.0  
**Date**: August 2025

## 1. Problem Statement

Real-time conversational AI faces multiple interconnected challenges that require careful coordination:

### 1.1 Technical Complexity
- **Multi-modal Session Management**: Coordinating audio, text, and potential visual streams in real-time
- **Latency Requirements**: Sub-second response times for natural conversation flow
- **Context Preservation**: Maintaining conversation continuity across technical boundaries
- **Interruption Handling**: Supporting natural human conversation patterns with mid-response interruptions
- **Audio Pipeline Coordination**: Synchronizing VAD, speech recognition, LLM generation, and audio playback

### 1.2 User Experience Requirements
Users expect natural conversation experiences:
- Seamless conversations without apparent technical interruptions
- Maintained context and personality across extended interactions
- Natural interruption support (speaking over the assistant)
- Consistent response quality and personality
- Multi-agent coordination (conversation + command extraction)

## 2. Architecture

### 2.1 System Overview

The OpenAI Realtime Agent implements a distributed, real-time conversational AI system with these key architectural principles:

**Distributed Agent Model**:
- Agents run as separate processes, connecting to ROS via WebSocket bridge
- Enables flexible deployment (local, containerized, or remote agents)
- Supports multiple specialized agents (conversation + command extraction)

**Real-Time Audio Pipeline**:
- Local VAD pre-filtering ensures only meaningful speech reaches the agent
- Direct audio streaming to OpenAI Realtime API for minimal latency
- Coordinated interruption handling across audio generation and playback

**Context-Aware Session Management**:
- Intelligent session cycling based on conversation pauses
- Text-based context preservation across session boundaries  
- Cost optimization without sacrificing user experience

### 2.2 Session Management

#### Cost-Driven Architecture Decision
Multimodal realtime API sessions can become exponentially expensive with session length due to token accumulation:
- Audio tokens from both user and assistant accumulate
- Video frame tokens (if using vision) persist throughout session
- Growing conversation context increases costs non-linearly

**Example Cost Escalation**:
- Minute 1: $0.50 (base audio tokens)
- Minute 5: $3.00 (accumulated context)
- Minute 10: $12.00 (exponential growth)
- Minute 20: $50.00+ (unsustainable for extended use)

#### Session Cycling Solution
**Key Architectural Innovation**: Local VAD pre-filtering enables intelligent session cycling:

- `/prompt_voice` topic receives VAD-filtered audio from `silero_vad_node`
- True silence detection via absence of `/prompt_voice` messages
- Independent of OpenAI's server-side VAD for pause detection
- Clean pause boundaries enable aggressive session cycling

**Session Lifecycle**:
```
IDLE → CONNECTING → CONNECTED → ACTIVE → PAUSED → CYCLING → IDLE
```

**Context Preservation Strategy**:
- User transcripts stored as conversation turns
- Assistant responses preserved in text form
- Previous conversation history injected into system prompt for new sessions
- Seamless continuity despite underlying technical session changes

#### Session Configuration
```yaml
# config/oai_realtime_agent.yaml
openai_realtime:
  model: "gpt-4o-realtime-preview"
  voice: "alloy"
  session_pause_timeout: 10.0  # Seconds before cycling
  input_audio_format: "pcm16"
  output_audio_format: "pcm16"
  modalities: ["text", "audio"]
```

### 2.3 Real-Time Interruption System

The agent supports natural conversational interruptions where users can speak over the assistant:

#### Three-Stage Interruption Process
```
User Speech (while assistant speaking) → Agent Detection →
┌─ 1. OpenAI API: response.cancel + conversation.item.truncate
├─ 2. Context: Clean conversation history  
└─ 3. Audio: interruption_signal → PyAudio abort()
```

**Stage 1 - API Cancellation**:
- Detects user speech while `assistant_speaking` is true
- Sends `response.cancel` to immediately stop LLM generation
- Sends `conversation.item.truncate` to remove partial response from context

**Stage 2 - Context Cleanup**:
- Tracks last assistant response item ID for proper truncation
- Prevents conversation history pollution with incomplete responses
- Maintains clean context for subsequent interactions

**Stage 3 - Audio Termination**:
- Publishes `interruption_signal` to audio player via bridge
- Audio player immediately clears queued audio data
- Uses PyAudio `abort()` for instant termination (not graceful `stop()`)

#### Bridge Configuration for Interruptions
```yaml
# config/bridge_dual_agent.yaml
published_topics:
  - topic: "interruption_signal"
    msg_type: "std_msgs/Bool"
```

## 3. Detailed Requirements

### 3.0 ROS Bridge Integration & Message Serialization

#### 3.0.1 ROS Bridge Integration

The agent connects to the ROS system via WebSocket bridge for distributed deployment:

**Message Flow**:
```
ROS Topics → Bridge → WebSocket → Agent → OpenAI API
```

**Agent Responsibilities**:
1. **Message Reception**: Receive ROS messages via WebSocket from bridge
2. **Format Conversion**: Convert ROS message formats to OpenAI Realtime API format
3. **Real-time Processing**: Handle audio streaming with minimal latency

**Key Message Types**:
- `AudioDataUtterance` → OpenAI audio streaming format
- `String` messages → Text input injection
- Bridge coordination for session management

#### 3.0.3 Message Type Support Matrix

| ROS Message Type | OpenAI Realtime Support | Serialization Method |
|------------------|------------------------|---------------------|
| `audio_common_msgs/AudioData` | ✅ Full Support | PCM → base64 |
| `audio_common_msgs/AudioStamped` | ✅ Full Support | Extract audio, PCM → base64 |
| `sensor_msgs/Image` | ⏳ Future | Awaiting API support |
| `std_msgs/String` | ✅ Text Input | Direct text injection |
| `geometry_msgs/Twist` | ❌ Output Only | Command generation only |

#### 3.0.2 Performance Requirements
- **Real-time Processing**: < 2ms for 20ms audio chunks 
- **Memory Efficiency**: Minimal overhead during message conversion
- **Error Resilience**: Graceful handling of malformed or unsupported message types

#### 3.0.3 WebSocket Bridge Connection

The agent uses the common `WebSocketBridgeInterface` for distributed deployment:

**Bridge Connection Process**:
```
Agent → WebSocket Connect → Bridge Registration → Message Streaming
```

The agent uses the shared `WebSocketBridgeInterface` from `agents/common/` with these capabilities:
    """WebSocket client for bridge communication"""
    
    def __init__(self, config: Dict):
        self.host = config.get('bridge_connection', {}).get('host', 'localhost')
        self.port = config.get('bridge_connection', {}).get('port', 8765)
        self.agent_id = config.get('agent_id', 'openai_realtime')
        self.websocket = None
        self.message_queue = asyncio.Queue()
        self.connected = False
        
    async def connect(self):
        """Connect to bridge WebSocket server with comprehensive error handling"""
        try:
            uri = f"ws://{self.host}:{self.port}"
            # IMPLEMENTATION NOTE: Use proper ping settings for production reliability
            self.websocket = await websockets.connect(
                uri,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            # Register with bridge
            registration = {
                "type": "register",
                "agent_id": self.agent_id,
                "capabilities": ["audio_processing", "realtime_api"],
                "subscriptions": [
                    {"topic": "/prompt_voice", "msg_type": "by_your_command/AudioDataUtterance"},
                    {"topic": "/prompt_text", "msg_type": "std_msgs/String"}
                ]
            }
            await self.websocket.send(json.dumps(registration))
            
            # Wait for registration response with timeout
            response = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
            data = json.loads(response)
            
            if data["status"] == "success":
                self.connected = True
                # Start message listener
                asyncio.create_task(self._message_listener())
                return True
            else:
                raise ConnectionError(f"Registration failed: {data}")
                
        except asyncio.TimeoutError:
            self.logger.error("Registration timed out")
            return False
        except Exception as e:
            self.logger.error(f"Failed to connect to bridge: {e}")
            return False
            
    async def _message_listener(self):
        """Listen for messages from bridge"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                if data["type"] == "message":
                    await self.message_queue.put(data["envelope"])
        except Exception as e:
            self.logger.error(f"Message listener error: {e}")
            self.connected = False
            
    async def get_inbound_message(self, timeout: float = 1.0):
        """Get message from bridge (compatible with existing interface)"""
        try:
            envelope_data = await asyncio.wait_for(self.message_queue.get(), timeout)
            return WebSocketMessageEnvelope(envelope_data)
        except asyncio.TimeoutError:
            return None
            
    async def put_outbound_message(self, topic: str, msg, msg_type: str):
        """Send message back to bridge"""
        if not self.connected:
            return False
            
        try:
            outbound = {
                "type": "outbound_message",
                "topic": topic,
                "msg_type": msg_type,
                "data": self._serialize_outbound_message(msg)
            }
            await self.websocket.send(json.dumps(outbound))
            return True
        except Exception as e:
            self.logger.error(f"Failed to send outbound message: {e}")
            return False

class WebSocketMessageEnvelope:
    """Envelope wrapper for WebSocket messages (compatible with MessageEnvelope)"""
    
    def __init__(self, envelope_data: Dict):
        self.msg_type = envelope_data["msg_type"]
        self.topic_name = envelope_data["topic_name"] 
        self.ros_msg_type = envelope_data["ros_msg_type"]
        self.timestamp = envelope_data["timestamp"]
        self.metadata = envelope_data.get("metadata", {})
        self.raw_data = self._deserialize_data(envelope_data["data"])
        
    def _deserialize_data(self, data: Dict):
        """Convert JSON data back to message-like object"""
        if self.ros_msg_type == "by_your_command/AudioDataUtterance":
            return AudioDataUtteranceProxy(data)
        elif self.ros_msg_type == "std_msgs/String":
            return StringProxy(data)
        else:
            return DataProxy(data)

class AudioDataUtteranceProxy:
    """Proxy object for AudioDataUtterance from WebSocket"""
    
    def __init__(self, data: Dict):
        self.audio_data = data.get("audio_data", [])
        self.utterance_id = data.get("utterance_id", "")
        self.start_time = data.get("start_time", 0.0)
        self.confidence = data.get("confidence", 0.0)
        # Add other AudioDataUtterance fields as needed

class OpenAIRealtimeAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.bridge_interface = WebSocketBridgeInterface(config)
        self.serializer = OpenAIRealtimeSerializer()
        self.websocket = None  # OpenAI WebSocket
        
    async def initialize(self):
        """Initialize agent with WebSocket bridge connection"""
        # Connect to bridge via WebSocket
        success = await self.bridge_interface.connect()
        if not success:
            self.logger.warning("Bridge connection failed - running in standalone mode")
            
    async def run(self):
        """Main agent loop - consumes from bridge via WebSocket, sends to OpenAI"""
        while True:
            # Get ROS message from bridge interface via WebSocket
            envelope = await self.bridge_interface.get_inbound_message(timeout=1.0)
            if envelope is None:
                continue  # Timeout, check OpenAI websocket or continue
            
            # Handle AudioDataUtterance with metadata
            if envelope.ros_msg_type == "by_your_command/AudioDataUtterance":
                # Agent-side transcoding: Extract audio data and metadata
                audio_data = envelope.raw_data.audio_data
                utterance_metadata = {
                    "utterance_id": envelope.raw_data.utterance_id,
                    "confidence": envelope.raw_data.confidence,
                    "start_time": envelope.raw_data.start_time
                }
                
                # Serialize for OpenAI Realtime API (agent responsibility)
                api_msg = self.serializer.serialize_audio_utterance(envelope)
                
                if self.websocket and api_msg:
                    await self.websocket.send(json.dumps(api_msg))
                    
                # Store metadata for context injection
                self.session_manager.add_utterance_context(utterance_metadata)
                    
            # Handle other message types...
            elif envelope.ros_msg_type == "std_msgs/String":
                await self.send_text_to_llm(envelope.raw_data.data)
    
    async def handle_llm_response(self):
        """Process responses from OpenAI and send back through WebSocket bridge"""
        async for message in self.websocket:
            data = json.loads(message)
            
            if data.get("type") == "response.audio_transcript.done":
                # Send transcript back through bridge
                transcript_msg = {"data": data["transcript"]}
                await self.bridge_interface.put_outbound_message(
                    "/response_text", 
                    transcript_msg, 
                    "std_msgs/String"
                )
                
            elif data.get("type") == "response.audio.delta":
                # Send audio response back through bridge
                audio_data = base64.b64decode(data["delta"])
                audio_msg = {"int16_data": np.frombuffer(audio_data, dtype=np.int16).tolist()}
                await self.bridge_interface.put_outbound_message(
                    "/response_voice", 
                    audio_msg, 
                    "audio_common_msgs/AudioData"
                )
```

#### 3.0.7 Agent Startup and Initialization (WebSocket-Based)
```python
async def main():
    \"\"\"Agent startup sequence with WebSocket bridge connection\"\"\"
    # Load configuration with bridge connection settings
    config = load_agent_config()
    
    # Create agent (no ROS2 initialization required)
    agent = OpenAIRealtimeAgent(config)
    await agent.initialize()  # Connects to bridge via WebSocket
    
    # Run agent main loop
    await agent.run()

def load_agent_config():
    \"\"\"Load agent configuration including bridge connection\"\"\"
    return {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'model': 'gpt-4o-realtime-preview',
        'voice': 'alloy',
        
        # Bridge connection settings
        'bridge_connection': {
            'type': 'websocket',
            'host': os.getenv('BRIDGE_HOST', 'localhost'),
            'port': int(os.getenv('BRIDGE_PORT', '8765')),
            'reconnect_interval': 5.0,
            'max_reconnect_attempts': 10
        },
        
        'agent_id': 'openai_realtime',
        'session_pause_timeout': 10.0,
        'session_max_duration': 120.0
    }
```

#### 3.0.8 Required Dependencies
The agent implementation requires these additional dependencies:
```python
# setup.py additions for OpenAI Realtime agent
install_requires=[
    'openai>=1.0.0',        # OpenAI Python SDK
    'websockets>=11.0',     # WebSocket client for bridge communication
    'aiohttp>=3.8.0',       # Async HTTP client
    'pydantic>=2.0',        # Data validation
    'numpy>=1.20.0',        # Audio processing
    'pyyaml>=6.0',          # Configuration file parsing
]

# No ROS2 dependencies required - agent runs standalone and connects via WebSocket
```

#### 3.0.9 AudioDataUtterance Transcoding and Metadata Handling

The agent performs transcoding from custom `AudioDataUtterance` messages to OpenAI Realtime API format:

```python
class OpenAIRealtimeSerializer:
    """Handle ROS → OpenAI Realtime API serialization with metadata support"""
    
    def serialize_audio_utterance(self, envelope: WebSocketMessageEnvelope) -> dict:
        """Convert AudioDataUtterance to OpenAI format with metadata preservation"""
        if envelope.ros_msg_type == "by_your_command/AudioDataUtterance":
            audio_data = envelope.raw_data.audio_data
            
            # Convert audio data to base64 PCM
            if isinstance(audio_data, list):
                pcm_bytes = np.array(audio_data, dtype=np.int16).tobytes()
            else:
                pcm_bytes = audio_data  # Already bytes
                
            base64_audio = base64.b64encode(pcm_bytes).decode()
            
            # OpenAI API message (audio only)
            api_msg = {
                "type": "input_audio_buffer.append",
                "audio": base64_audio
            }
            
            # Store metadata separately for context injection
            self.current_utterance_metadata = {
                "utterance_id": envelope.raw_data.utterance_id,
                "start_time": envelope.raw_data.start_time,
                "confidence": envelope.raw_data.confidence,
                "chunk_sequence": envelope.raw_data.chunk_sequence,
                "is_utterance_end": envelope.raw_data.is_utterance_end,
                "timestamp": envelope.timestamp
            }
            
            return api_msg
            
    def get_utterance_metadata(self) -> Dict:
        """Get metadata from last processed utterance"""
        return getattr(self, 'current_utterance_metadata', {})

class SessionManager:
    """Enhanced session manager with utterance context support"""
    
    def add_utterance_context(self, metadata: Dict):
        """Store utterance metadata for session context"""
        # IMPLEMENTATION NOTE: Track utterance sequences for end-of-utterance detection
        self.utterance_contexts.append({
            "utterance_id": metadata["utterance_id"],
            "confidence": metadata["confidence"], 
            "start_time": metadata["start_time"],
            "chunk_sequence": metadata.get("chunk_sequence", 0),
            "is_utterance_end": metadata.get("is_utterance_end", False),
            "processed_at": time.time()
        })
        
        # Keep only recent utterances (last 10)
        if len(self.utterance_contexts) > 10:
            self.utterance_contexts = self.utterance_contexts[-10:]

class SessionManager:
    """Enhanced session manager with utterance context support"""
    
    def add_utterance_context(self, metadata: Dict):
        """Store utterance metadata for session context"""
        self.utterance_contexts.append({
            "utterance_id": metadata["utterance_id"],
            "confidence": metadata["confidence"], 
            "start_time": metadata["start_time"],
            "processed_at": time.time()
        })
        
        # Keep only recent utterances (last 10)
        if len(self.utterance_contexts) > 10:
            self.utterance_contexts = self.utterance_contexts[-10:]
            
    def build_context_prompt(self, base_prompt: str) -> str:
        """Inject utterance context into system prompt"""
        if not self.utterance_contexts:
            return base_prompt
            
        recent_contexts = self.utterance_contexts[-3:]  # Last 3 utterances
        context_info = []
        
        for ctx in recent_contexts:
            context_info.append(
                f"Utterance {ctx['utterance_id']}: confidence={ctx['confidence']:.2f}"
            )
            
        context_section = "\\n".join([
            "\\nRecent speech context:",
            *context_info,
            "Use this context to better understand speech quality and user intent.\\n"
        ])
        
        return base_prompt + context_section
```

#### 3.0.10 Future Media Stream Support

The architecture is designed to support video/image streams with similar agent-side transcoding:

```python
class MediaStreamSerializer:
    """Future: Handle video/image stream transcoding"""
    
    def serialize_image_data(self, envelope: WebSocketMessageEnvelope) -> dict:
        """Convert ROS Image to OpenAI format (when supported)"""
        if envelope.ros_msg_type == "sensor_msgs/Image":
            # Extract image data and metadata
            image_data = envelope.raw_data.data
            encoding = envelope.raw_data.encoding
            width = envelope.raw_data.width
            height = envelope.raw_data.height
            
            # Agent-side transcoding for API requirements
            # - Resize to API requirements (e.g., 1024x1024)
            # - Convert format (JPEG, PNG)
            # - Apply compression
            processed_image = self.process_image_for_api(
                image_data, encoding, width, height
            )
            
            return {
                "type": "input_image",
                "image": base64.b64encode(processed_image).decode(),
                "metadata": {
                    "original_size": (width, height),
                    "encoding": encoding,
                    "timestamp": envelope.timestamp
                }
            }
```

#### 3.0.11 Implementation Lessons Learned

**Critical Implementation Details from Production Development:**

1. **WebSocket Library Evolution**: The `websockets.WebSocketClientProtocol` type hint is deprecated. Use connection objects directly and handle deprecation warnings.

2. **Connection Resilience Patterns**:
```python
async def _handle_disconnection(self):
    """Real-world reconnection requires exponential backoff and connection state tracking"""
    if self.reconnect_attempts >= self.max_reconnect_attempts:
        self.logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
        return
        
    self.reconnect_attempts += 1
    backoff_time = min(self.reconnect_interval * (2 ** self.reconnect_attempts), 60.0)
    await asyncio.sleep(backoff_time)
    
    success = await self.connect()
    # Recursive retry pattern handles multiple failure scenarios
```

3. **AudioDataUtterance Processing Complexity**: The actual transcoding involves more metadata fields than initially specified:
   - `chunk_sequence`: Critical for message ordering
   - `is_utterance_end`: Essential for conversation boundary detection
   - `confidence`: Used for adaptive speech processing

4. **Memory Management for Audio Data**: Large audio arrays (>1000 samples) can cause memory spikes during base64 encoding. Consider streaming or chunking for very long utterances.

5. **Error Handling Patterns**: Production deployment requires comprehensive error handling at every WebSocket operation:
```python
async def put_outbound_message(self, topic: str, msg_data: Dict, msg_type: str) -> bool:
    if not self.connected or not self.websocket:
        self.logger.warning("Cannot send message - not connected to bridge")
        return False
        
    try:
        # Always wrap WebSocket operations in try-catch
        await self.websocket.send(json.dumps(outbound))
        return True
    except websockets.exceptions.ConnectionClosed:
        self.connected = False
        await self._handle_disconnection()
        return False
    except Exception as e:
        self.logger.error(f"Failed to send outbound message: {e}")
        return False
```

6. **Testing Requirements**: End-to-end testing revealed that unit tests alone are insufficient. System-level integration tests are essential for validating the WebSocket communication layer.

### 3.1 Conversation Lifecycle Terminology

#### 3.1.1 Three-Tier Hierarchy
The system manages audio interactions through a well-defined three-tier hierarchy:

1. **Utterance** (Lowest Level)
   - **Definition**: A single continuous speech segment from VAD detection start to end
   - **Identifier**: `utterance_id` - timestamp in nanoseconds from first audio frame
   - **Boundaries**: Determined by Silero VAD detecting speech start/stop
   - **Duration**: Typically 1-30 seconds
   - **Message Type**: `AudioDataUtterance` with metadata (utterance_id, is_utterance_end, chunk_sequence)
   - **Purpose**: Track individual speech segments for recording and processing

2. **Session** (Middle Level)
   - **Definition**: A single WebSocket connection to OpenAI Realtime API
   - **Identifier**: Internal session ID managed by SessionManager
   - **Boundaries**: Connection establishment to closure (pause-based or limit-based)
   - **Duration**: 10s-120s depending on pause patterns
   - **Lifecycle**: IDLE → CONNECTING → ACTIVE → CLOSING → CLOSED
   - **Purpose**: Cost optimization unit - reset token accumulation

3. **Conversation** (Highest Level)
   - **Definition**: A complete dialog that may span multiple sessions
   - **Identifier**: `conversation_id` - timestamp format `conv_YYYYMMDD_HHMMSS_microseconds`
   - **Boundaries**: Timeout-based (600s default) or external reset
   - **Duration**: Minutes to hours
   - **Persistence**: Text context preserved across session boundaries
   - **Purpose**: Maintain conversational continuity for user experience

#### 3.1.2 Relationship Mapping
```
Conversation (10+ minutes)
├── Session 1 (30 seconds)
│   ├── Utterance 1 (5 seconds) 
│   ├── Utterance 2 (3 seconds)
│   └── Utterance 3 (8 seconds)
├── [Pause - Session Cycling]
├── Session 2 (45 seconds)
│   ├── Utterance 4 (12 seconds)
│   └── Utterance 5 (20 seconds)
└── ... (continues until conversation timeout)
```

### 3.2 Session Lifecycle Management

#### 3.2.1 Session States
```
IDLE → CONNECTING → ACTIVE → CLOSING → CLOSED → IDLE
                        ↓                          ↑
                        └──────────────────────────┘
                         (aggressive cycling on pause)
```

#### 3.1.2 Core Strategy: Aggressive Pause-Based Cycling

**Key Insight**: Every pause is an opportunity to reset the expensive token accumulation by cycling the session. This works because `/prompt_voice` arrives pre-filtered by local VAD, allowing true pause detection.

**VAD Architecture**:
```
[Microphone] → [silero_vad_node] → /prompt_voice → [ros_ai_bridge] → [LLM API]
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
- First `/prompt_voice` received when no session exists
- New `/prompt_voice` after pause-induced closure
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
1. **Timeout-Based Reset** (Automatic):
   - Triggered when no voice chunks received for `conversation_timeout` seconds (default: 600s)
   - ConversationMonitor generates new conversation_id with timestamp format
   - Publishes new ID to `/conversation_id` topic for system-wide coordination
   - Session manager clears conversation context on ID change

2. **External Reset** (Manual/System):
   - Any component can publish new conversation_id to `/conversation_id` topic
   - Agent subscribes to topic and detects ID changes
   - Triggers immediate context reset and session cycling if active
   - Used for user switching, manual reset, or multi-agent coordination

3. **Future Triggers** (Planned):
   - User change detection via voice/face recognition
   - Explicit "new conversation" command from user
   - Context overflow or quality degradation

### 3.2 Conversation Management Implementation

#### 3.2.1 ConversationMonitor Class
The `ConversationMonitor` class manages conversation lifecycle and ID generation:

```python
class ConversationMonitor:
    """Monitors conversation lifecycle and manages conversation IDs"""
    
    def __init__(self, timeout: float = 600.0, 
                 on_conversation_change: Optional[Callable[[str, str], None]] = None):
        # Generates initial conversation_id on startup
        self.current_conversation_id = self._generate_conversation_id()
        
    def _generate_conversation_id(self) -> str:
        """Format: conv_YYYYMMDD_HHMMSS_microseconds"""
        now = datetime.now()
        return f"conv_{now.strftime('%Y%m%d_%H%M%S_%f')}"
        
    def record_activity(self):
        """Called on each voice chunk to reset timeout"""
        
    async def _monitor_loop(self):
        """Background task checking for conversation timeout"""
```

#### 3.2.2 Response Tracking and Timeout Protection

The agent implements robust response expectation tracking to prevent deadlock states:

**Response Expectations**:
```python
self.pending_responses = {
    'transcription': True,      # Expect user transcript from OpenAI
    'assistant_response': True, # Expect assistant to start responding
    'audio_complete': True      # Expect response completion
}
```

**Timeout Protection**:
- **10-second timeout** prevents indefinite waiting for responses that never arrive
- **Automatic recovery** clears expectations and allows session cycling
- **Log throttling** (5-second intervals) prevents spam during waiting periods
- **Long response protection** - timeout only applies during waiting phase, not active responses

**Common Deadlock Scenarios**:
1. OpenAI doesn't provide transcription (no speech detected, API issues)
2. Assistant doesn't respond (prompt issues, API problems)
3. Response never completes (WebSocket disconnection)

**Recovery Mechanism**:
```python
# After 10 seconds of waiting
if timeout_exceeded:
    self.logger.warning("⏰ Response timeout - forcing cycle")
    self._clear_response_expectations()
    # Session cycling can now proceed
```

#### 3.2.3 ROS Topic Integration
**Topic**: `conversation_id` (relative topic - respects namespace/prefix)
- **Message Type**: `std_msgs/String`
- **Publishers**: Agent on timeout or reset
- **Subscribers**: All agents and components needing conversation boundaries
- **With Namespaces**: e.g., `/robot1/voice/conversation_id`

**Publishing Behavior**:
```python
# On timeout detection
if self.check_timeout():
    new_id = self._generate_conversation_id()
    # Publish to ROS topic
    await self.publish_conversation_id(new_id)
    
# On external reset
elif envelope.topic_name == "/conversation_id":
    new_id = envelope.raw_data.data
    if new_id != self.current_conversation_id:
        self.conversation_monitor.handle_external_reset(new_id)
```

#### 3.2.3 Context Reset on Conversation Change
When conversation ID changes (timeout or external):
1. Current session context is finalized and cleared
2. Active sessions are cycled to start fresh
3. New conversation ID is used for all logging
4. Conversation statistics are updated

### 3.3 Named Prompt System (Implemented)

#### 3.3.1 Prompt Configuration Structure
The agent uses a sophisticated named prompt system via `config/prompts.yaml`:

```yaml
prompts:
  barney_command_visual:
    name: "Barney - Command and Visual Mode"
    description: "Primary prompt for Barney robot with command/conversation modes"
    version: "1.0"
    tested_with: ["openai_realtime", "gemini_live"]
    system_prompt: |
      You are a real skid-steer robot that can move around and you have one 4dof arm...
      # (5,604 character sophisticated robot personality)
      
  friendly_assistant:
    name: "Simple Friendly Assistant"  
    description: "Basic conversational robot without command modes"
    system_prompt: |
      You are a friendly robot assistant. You can see through a camera...

selection_rules:
  default: "barney_command_visual"
  conditional:
    - condition: "user_age < 10"
      prompt: "friendly_assistant"
    - condition: "environment == 'crowded'" 
      user_prefix: "safety_emphasis"
  ab_tests:
    visual_description_test:
      enabled: false
      variants:
        - prompt: "barney_command_visual"
          weight: 50
        - prompt: "barney_command_visual_v2"
          weight: 50
```

#### 3.2.2 PromptLoader Implementation
```python
class PromptLoader:
    """Load and select named system prompts with A/B testing"""
    
    def __init__(self, prompts_file: str = None):
        # Loads config/prompts.yaml by default
        self.prompts: Dict[str, PromptInfo] = {}
        self.user_prefixes: Dict[str, UserPrefixInfo] = {}
        self.selection_rules: Dict[str, Any] = {}
        self._load_prompts()
        
    def select_prompt(self, context: Dict[str, Any] = None) -> str:
        """Select appropriate prompt based on rules and context"""
        # 1. Check conditional rules first
        for rule in self.selection_rules.get('conditional', []):
            if self._evaluate_condition(rule.get('condition', ''), context):
                return self.prompts[rule['prompt']].system_prompt
                
        # 2. Check A/B testing
        for test_name, test_config in self.selection_rules.get('ab_tests', {}).items():
            if test_config.get('enabled', False):
                selected_prompt = self._select_ab_variant(test_config)
                return self.prompts[selected_prompt].system_prompt
                
        # 3. Use default prompt
        default_prompt = self.selection_rules.get('default', 'fallback')
        return self.prompts[default_prompt].system_prompt
```

#### 3.2.3 SessionManager Integration
The SessionManager automatically uses the prompt system:

```python
class SessionManager:
    def __init__(self, config: Dict):
        # Initialize named prompt system
        self.prompt_loader = PromptLoader()
        self.fallback_system_prompt = config.get('system_prompt', self._default_system_prompt())
        
        # Context for prompt selection
        self.prompt_context = {
            'user_age': config.get('user_age'),
            'environment': config.get('environment', 'normal'),
            'robot_name': config.get('robot_name', 'robot'),
            'agent_id': config.get('agent_id', 'openai_realtime')
        }
        
    async def _configure_session(self, context: Optional[ConversationContext] = None):
        """Configure session with system prompt and context"""
        # Select system prompt using named prompt system
        try:
            selected_prompt = self.prompt_loader.select_prompt(self.prompt_context)
            self.logger.info(f"Using named prompt: {self.prompt_loader.selection_rules.get('default')}")
        except Exception as e:
            self.logger.warning(f"Named prompt selection failed, using fallback: {e}")
            selected_prompt = self.fallback_system_prompt
            
        # Build final system prompt with conversation context
        system_prompt = self.context_manager.build_system_prompt(selected_prompt, context)
        
        # Send to OpenAI Realtime API
        config_msg = {
            "type": "session.update",
            "session": {
                "instructions": system_prompt,  # Final prompt sent to API
                # ... other session config
            }
        }
```

#### 3.2.4 Prompt Selection Logic 

**Current Implementation Behavior**:
- **Adults (age ≥ 10)**: Get full `barney_command_visual` prompt (5,604 chars) with:
  - Command vs Conversation mode detection
  - Preset arm positions (bumper, tenhut, lookup, lookout, reach)  
  - Directional bearings (left, right, forward, etc.)
  - Visual JSON object detection capabilities
  - Witty, engaging robot personality
  
- **Children (age < 10)**: Get simplified `friendly_assistant` prompt (259 chars)
  - Basic conversational responses only
  - No complex command parsing
  - Child-appropriate interaction style

- **Crowded environments**: Same prompt selection, with optional safety prefix injection

#### 3.2.5 User Prompt Prefixes
Dynamic context injection via prefix templates:

```yaml
user_prompt_prefixes:
  context_reminder:
    description: "Reminds the LLM about ongoing context"
    prefix: |
      Remember, we were just discussing: {last_topic}
      The user's last command was: {last_command}
      
  safety_emphasis:
    description: "Emphasizes safety in command interpretation"
    prefix: |
      Safety reminder: Verify all movement commands are safe before executing.
      Current obstacles detected: {obstacle_list}
```

#### 3.2.6 Runtime Management
```python
# Update prompt context dynamically
session_manager.update_prompt_context({'environment': 'crowded', 'user_age': 30})

# Get current prompt information
prompt_info = session_manager.get_current_prompt_info()
# Returns: {'prompt_id': 'barney_command_visual', 'name': 'Barney - Command and Visual Mode', ...}

# Reload prompts from file without restart
session_manager.reload_prompts()

# List available prompts
available = session_manager.list_available_prompts()
# Returns: ['barney_command_visual', 'barney_command_visual_v2', 'friendly_assistant']
```

#### 3.2.7 Runtime Prompt Switching (Implemented)
The agent supports changing system prompts during runtime without restart:

```python
# Agent-level prompt switching API
agent = OpenAIRealtimeAgent(config)

# Switch to specific prompt (overrides context-based selection)
success = await agent.switch_system_prompt(prompt_id='friendly_assistant')

# Update context and let system reselect prompt
success = await agent.switch_system_prompt(context_updates={'user_age': 8, 'environment': 'crowded'})

# Get current prompt information
prompt_info = agent.get_current_system_prompt_info()
# Returns: {'selection_type': 'override', 'prompt_id': 'friendly_assistant', ...}

# Clear override and return to context-based selection
agent.clear_system_prompt_override()

# Reload prompts from prompts.yaml without restart
agent.reload_system_prompts()
```

**Active Session Behavior**: If a WebSocket session is active with OpenAI, the prompt change is applied immediately via `session.update` message. The user will notice the personality change in the next response. If no session is active, the change is stored and applied when the next session is created.

### 3.3 Standalone Mode & Debug Interface (Implemented)

#### 3.3.1 Standalone Mode Architecture
When the agent cannot connect to the ROS AI Bridge (no bridge running), it automatically enables **standalone mode** with a debug interface for testing:

```
Normal Mode:    ROS Topics → Bridge → Agent → OpenAI
Standalone Mode: Test Scripts → Debug Interface → Agent → OpenAI
```

**Activation Logic**:
```python
async def _connect_to_bridge(self):
    # Try WebSocket connection to bridge
    success = await self.bridge_interface.connect_with_retry()
    
    if not success:
        # Bridge connection failed - enable debug mode
        self.bridge_interface = None
        self.debug_interface = DebugInterface(self)
        await self.debug_interface.start()
        self.logger.info("🔧 Debug interface enabled for standalone mode")
```

#### 3.3.2 Debug Interface Capabilities

**Audio Data Injection**:
```python
# Generate synthetic test audio
audio_data = create_test_audio_sine_wave(440, 2.0)  # 440Hz for 2 seconds
noise_data = create_test_audio_noise(1.0)           # 1 second white noise

# Inject into agent (processes as if from /prompt_voice topic)
success = await agent.debug_inject_audio(
    audio_data,
    utterance_id="test_speech_001",
    confidence=0.95,
    is_utterance_end=True
)
```

**Text Message Injection**:
```python
# Inject text messages (processes as if from /prompt_text topic)
success = await agent.debug_inject_text("Hello Barney, move your arm up!")
```

**Runtime Management**:
```python
# Check if in standalone mode
is_standalone = agent.is_standalone_mode()  # True when bridge_interface is None

# Get debug statistics
stats = agent.get_debug_stats()
# Returns: {'messages_injected': 5, 'responses_received': 3, 'running': True, ...}
```

#### 3.3.3 Test Audio Data Generation

**Built-in Audio Generators**:
```python
def create_test_audio_sine_wave(frequency: int = 440, duration: float = 1.0, 
                               sample_rate: int = 16000) -> List[int]:
    """Generate sine wave test audio (pure tone)"""
    
def create_test_audio_noise(duration: float = 1.0, sample_rate: int = 16000) -> List[int]:
    """Generate white noise test audio"""
    
def load_wav_file(file_path: str, target_sample_rate: int = 16000) -> Optional[List[int]]:
    """Load WAV file and convert to 16kHz mono PCM (requires scipy)"""
```

**Audio Data Format**:
- **Type**: `List[int]` (16-bit PCM samples)
- **Sample Rate**: 16kHz (standard for OpenAI Realtime API)
- **Value Range**: -32767 to +32767
- **Compatibility**: Direct injection into agent processing pipeline

#### 3.3.4 Testing Scenarios & Use Cases

**1. Basic Audio Processing Validation**:
```python
# Test agent's core audio handling
audio = create_test_audio_sine_wave(440, 2.0)
await agent.debug_inject_audio(audio, is_utterance_end=True)
# Expected: Creates session → Sends to OpenAI → Processes responses
```

**2. Prompt Personality Testing**:
```python
# Test adult vs child prompt selection
await agent.switch_system_prompt(context_updates={'user_age': 30})
await agent.debug_inject_audio(test_audio)  # Should use Barney command prompt

await agent.switch_system_prompt(context_updates={'user_age': 7})  
await agent.debug_inject_audio(test_audio)  # Should use friendly assistant prompt
```

**3. Session Cycling Validation**:
```python
# Test cost-optimized session management
await agent.debug_inject_audio(audio1, is_utterance_end=True)
await asyncio.sleep(12)  # Wait longer than pause_timeout (10s)
await agent.debug_inject_audio(audio2, is_utterance_end=True)
# Expected: Session cycles between messages to optimize costs
```

**4. Stress Testing**:
```python
# Test rapid message injection and queue handling
for i in range(10):
    audio = create_test_audio_sine_wave(440 + i*50, 0.5)  # Different frequencies
    await agent.debug_inject_audio(audio, f"stress_test_{i}")
    await asyncio.sleep(0.1)
```

#### 3.3.5 Development Workflow Integration

**Agent Development Cycle**:
1. **Develop Features**: Implement agent capabilities using standalone mode
2. **Unit Testing**: Test with synthetic audio data and controlled scenarios
3. **Integration Testing**: Validate OpenAI API integration and session management
4. **System Testing**: Test with full ROS system when ready

**Standalone Testing Script Template**:
```python
#!/usr/bin/env python3
import asyncio
from oai_realtime_agent import OpenAIRealtimeAgent
from debug_interface import create_test_audio_sine_wave

async def test_agent_feature():
    # Configuration (bridge connection will fail)
    config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'user_age': 30,  # Adult context
        'bridge_connection': {'host': 'localhost', 'port': 8765}
    }
    
    # Initialize agent in standalone mode
    agent = OpenAIRealtimeAgent(config)
    await agent.initialize()  # Bridge fails → debug interface enabled
    
    # Start agent background task
    agent_task = asyncio.create_task(agent.run())
    await asyncio.sleep(1)  # Let agent start
    
    try:
        # Test scenario
        audio = create_test_audio_sine_wave(440, 2.0)
        success = await agent.debug_inject_audio(audio, is_utterance_end=True)
        
        # Wait for processing and check results
        await asyncio.sleep(3)
        metrics = agent.get_metrics()
        
        print(f"Audio injection: {'✅' if success else '❌'}")
        print(f"Messages processed: {metrics['messages_processed']}")
        print(f"OpenAI messages sent: {metrics['messages_sent_to_openai']}")
        
    finally:
        await agent.stop()
        await agent_task

# Run test
asyncio.run(test_agent_feature())
```

#### 3.3.6 Capabilities Matrix

| Feature | Normal Mode | Standalone Mode | Notes |
|---------|-------------|-----------------|-------|
| Audio Processing | ✅ ROS Topics | ✅ Debug Injection | Same pipeline |
| OpenAI API Integration | ✅ | ✅ | Full functionality |
| Session Management | ✅ | ✅ | Same cycling logic |
| Prompt Switching | ✅ | ✅ | Runtime changes work |
| Metrics & Monitoring | ✅ | ✅ | Full stats available |
| ROS Topic Publishing | ✅ | ❌ | No bridge connection |
| Hardware Integration | ✅ | ❌ | No real sensors |
| End-to-end Testing | ✅ | ❌ | Requires full system |

**Key Advantage**: Standalone mode provides **complete agent functionality testing** without requiring the full ROS ecosystem, dramatically improving development efficiency and debugging capabilities.

### 3.4 Context Preservation Strategy

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
    
    # New session will be created when voice_chunks arrive
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

## 4. Audio Architecture & Implementation Details

### 4.1 Audio Pipeline
The complete audio flow through the system:

```
Input Pipeline:
Microphone → audio_capturer (16kHz) → echo_suppressor → /audio_filtered → 
silero_vad → /prompt_voice (AudioDataUtterance) → ROS Bridge → 
WebSocket → Agent → OpenAI API

Output Pipeline:
OpenAI API → response.audio.delta (24kHz PCM16) → Agent → 
/response_voice (AudioData) → simple_audio_player → PyAudio → Speakers
```

### 4.2 Critical Implementation Discoveries

#### 4.2.1 Manual Response Triggering Required
- **Discovery**: OpenAI's server VAD successfully transcribes speech but does NOT automatically generate responses
- **Solution**: Must manually send `{"type": "response.create"}` after receiving transcription
- **Impact**: This undocumented behavior required significant debugging to discover

#### 4.2.2 Audio Format Handling
- **Issue**: OpenAI outputs 24kHz PCM16 audio (not the standard 16kHz)
- **Challenge**: audio_common's audio_player_node requires AudioStamped messages with sample rate info
- **Solution**: Created simple_audio_player that directly handles AudioData at 24kHz using PyAudio

#### 4.2.3 Echo Suppression Architecture
- **Problem**: Assistant hears itself when mic is near speakers, creating feedback loops
- **Solution**: echo_suppressor node that:
  - Monitors /assistant_speaking status from simple_audio_player
  - Mutes microphone input while assistant is speaking
  - Re-enables mic after assistant finishes
  - Publishes filtered audio to /audio_filtered for VAD

### 4.3 Message Type Considerations

#### 4.3.1 AudioData vs AudioStamped
- **AudioData**: Simple array of audio samples, no metadata
- **AudioStamped**: Includes header + AudioInfo (format, rate, channels)
- **Issue**: Bridge serialization issues with nested message types
- **Resolution**: Use AudioData for simplicity, hardcode sample rates in nodes

#### 4.3.2 Field Mapping Discovery
- **Critical Finding**: AudioDataUtterance uses `int16_data` field (not `audio_data`)
- **Impact**: Initial implementation sent empty audio due to incorrect field access
- **Lesson**: Always verify message field names match between publishers and subscribers

### 4.4 Audio Debugging Tools
- **voice_chunk_recorder**: Enhanced with audio_data mode to save /response_voice to WAV files
- **Parameters**: input_mode='audio_data', input_sample_rate=24000, audio_timeout=10.0
- **Usage**: Verify audio format and content before debugging playback issues

## 5. Implementation TODOs

### 5.1 Phase 1: Core Session Management & Bridge Integration
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

### 5.2 Phase 2: Intelligent Session Cycling
- [ ] Implement session cycling on pause detection (PauseDetector integration)
- [ ] Create conversation context extraction and text-based persistence  
- [ ] Build context injection system for new sessions (system prompt modification)
- [ ] Implement graceful session termination with context preservation
- [ ] Add session rotation metrics and timing optimization
- [ ] Create conversation summarizer for contexts > 2000 tokens
- [ ] Test aggressive cycling with various pause timeout settings

### 5.3 Phase 3: Advanced Features
- [ ] Implement predictive rotation (anticipate before limits)
- [ ] Add multi-provider rotation (OpenAI → Gemini fallback)
- [ ] Create conversation analytics dashboard
- [ ] Build automated testing for edge cases
- [ ] Add support for conversation branching/merging

### 5.4 Phase 4: Optimization
- [ ] Implement token usage prediction models
- [ ] Add dynamic timeout adjustment based on conversation flow
- [ ] Create cost optimization recommendations
- [ ] Build conversation quality scoring
- [ ] Implement A/B testing framework for parameters

## 6. Configuration Parameters

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
  conversation_timeout: 600.0      # Seconds of inactivity before new conversation_id (default: 10 min)
  conversation_buffer_size: 10000  # Max text tokens to preserve across sessions
  # Note: max_context_age is deprecated - use conversation_timeout instead
  
  # Performance Tuning
  rotation_overlap_ms: 200      # overlap time during rotation
  audio_buffer_size: 1024       # samples to buffer during rotation
  
  # Quality Settings
  preserve_emotion_context: true
  summarize_after_minutes: 10
  min_utterance_for_persistence: 5  # words
```

## 6. Testing Scenarios

### 6.1 Standalone Mode Testing (Implemented)

**Primary Testing Method**: Agent runs in standalone mode with debug interface when ROS bridge is unavailable.

#### 6.1.1 Development Testing
```python
# Basic functionality test
async def test_basic_audio_processing():
    agent = OpenAIRealtimeAgent(config)
    await agent.initialize()  # Bridge fails → standalone mode
    
    audio = create_test_audio_sine_wave(440, 2.0)
    success = await agent.debug_inject_audio(audio, is_utterance_end=True)
    
    # Verify: Session created, audio sent to OpenAI, responses processed
    assert success
    assert agent.session_manager.sessions_created > 0
```

#### 6.1.2 Prompt System Testing
```python
# Test prompt switching and personality changes
async def test_prompt_switching():
    agent = OpenAIRealtimeAgent(config)
    await agent.initialize()
    
    # Test adult → child prompt switching
    await agent.switch_system_prompt(context_updates={'user_age': 30})
    adult_info = agent.get_current_system_prompt_info()
    assert adult_info['prompt_id'] == 'barney_command_visual'
    
    await agent.switch_system_prompt(context_updates={'user_age': 7})
    child_info = agent.get_current_system_prompt_info()
    assert child_info['prompt_id'] == 'friendly_assistant'
```

#### 6.1.3 Session Cycling Testing
```python
# Test pause-based session cycling
async def test_session_cycling():
    agent = OpenAIRealtimeAgent(config)
    await agent.initialize()
    
    # First message creates session
    audio1 = create_test_audio_sine_wave(440, 1.0)
    await agent.debug_inject_audio(audio1, is_utterance_end=True)
    session_1_id = agent.session_manager.sessions_created
    
    # Wait for pause timeout
    await asyncio.sleep(12)  # > pause_timeout (10s)
    
    # Second message should cycle session
    audio2 = create_test_audio_sine_wave(880, 1.0)
    await agent.debug_inject_audio(audio2, is_utterance_end=True)
    session_2_id = agent.session_manager.sessions_created
    
    assert session_2_id > session_1_id  # Session was cycled
```

#### 6.1.4 Test Audio Generation
```python
# Various audio test data types
test_scenarios = [
    ("Pure Tone", create_test_audio_sine_wave(440, 1.0)),
    ("High Frequency", create_test_audio_sine_wave(1000, 0.5)),
    ("Low Frequency", create_test_audio_sine_wave(200, 2.0)),
    ("White Noise", create_test_audio_noise(1.0)),
    ("Short Burst", create_test_audio_sine_wave(440, 0.1)),
    ("Long Speech", create_test_audio_sine_wave(440, 5.0)),
]

for name, audio_data in test_scenarios:
    success = await agent.debug_inject_audio(audio_data, f"test_{name}")
    print(f"{name}: {'✅' if success else '❌'}")
```

### 6.2 Cost Optimization Tests
1. **Token Accumulation Test**: Stream continuous audio for 30 minutes, verify costs stay linear
2. **Rotation Frequency Test**: Verify optimal rotation intervals for different conversation types
3. **Context Preservation Test**: Ensure no critical information lost during rotations

### 6.3 User Experience Tests
1. **Seamless Rotation Test**: Users should not notice session rotations
2. **Pause/Resume Test**: Natural conversation flow across pauses
3. **Long Conversation Test**: Multi-hour conversations remain coherent

### 6.4 Integration Testing

#### 6.4.1 Full System Testing (requires ROS bridge)
```bash
# Start ROS bridge
ros2 launch by_your_command oai_realtime.launch.py

# Agent connects to bridge instead of using debug interface
# Test with real audio input from microphone via VAD
```

#### 6.4.2 Standalone vs Full System Comparison
| Test Type | Standalone Mode | Full System | Notes |
|-----------|-----------------|-------------|-------|
| Audio Processing | ✅ Synthetic data | ✅ Real microphone | Same audio pipeline |
| Prompt Switching | ✅ Full testing | ✅ Full testing | Identical functionality |
| Session Management | ✅ Full testing | ✅ Full testing | Same cycling logic |
| OpenAI Integration | ✅ Real API calls | ✅ Real API calls | Identical |
| Response Handling | ⚠️ Logging only | ✅ ROS topics | Debug vs real output |
| End-to-end Flow | ❌ Incomplete | ✅ Complete | Hardware integration |

**Testing Strategy**: Use standalone mode for rapid development and unit testing, then validate with full system for integration testing.

## 7. Implementation Status

### 7.1 Completed Features ✅

#### 7.1.1 Core Agent Architecture
- ✅ **WebSocket-based distributed deployment** - Agent connects to bridge via WebSocket
- ✅ **Connection retry logic** - Multiple attempts with configurable intervals
- ✅ **Standalone mode with debug interface** - Automatic fallback when bridge unavailable
- ✅ **Session management with intelligent cycling** - Pause-based and limit-based rotation
- ✅ **OpenAI Realtime API integration** - Full WebSocket API support

#### 7.1.2 Named Prompt System  
- ✅ **YAML-based prompt configuration** - Multiple named prompts with metadata
- ✅ **Context-based prompt selection** - Age, environment, robot-specific selection
- ✅ **A/B testing framework** - Weighted variant selection (configurable)
- ✅ **Runtime prompt switching** - Change prompts without restart
- ✅ **User prompt prefixes** - Dynamic context injection templates

#### 7.1.3 Audio Processing & Message Handling
- ✅ **AudioDataUtterance transcoding** - Agent-side serialization to OpenAI format
- ✅ **Zero-copy message handling** - Efficient ROS message processing
- ✅ **Metadata preservation** - Utterance context and conversation state
- ✅ **Message envelope system** - Unified message format across components
- ✅ **Manual response triggering** - Workaround for server VAD limitation
- ✅ **24kHz audio playback** - Direct AudioData playback via simple_audio_player
- ✅ **Echo suppression** - Prevents feedback loops by muting mic during assistant speech

#### 7.1.4 Debug & Testing Infrastructure
- ✅ **Debug interface** - Direct message injection for testing
- ✅ **Test audio generation** - Sine waves, noise, WAV file loading
- ✅ **Comprehensive test suites** - Unit tests for all major components
- ✅ **Development workflow support** - Standalone testing without ROS

#### 7.1.5 Configuration & Management
- ✅ **Runtime configuration updates** - Change settings without restart
- ✅ **Metrics and monitoring** - Comprehensive stats and performance tracking
- ✅ **Error handling and recovery** - Graceful failure handling
- ✅ **Hot-reloading capabilities** - Reload prompts and config from files

#### 7.1.6 Conversation Lifecycle Management
- ✅ **Three-tier hierarchy** - Utterance → Session → Conversation tracking
- ✅ **ConversationMonitor class** - Timeout detection and ID generation
- ✅ **Bidirectional conversation_id topic** - System-wide conversation coordination
- ✅ **Timestamp-based conversation IDs** - Format: conv_YYYYMMDD_HHMMSS_microseconds
- ✅ **Automatic timeout detection** - 600s default conversation timeout
- ✅ **External reset handling** - Subscribe to conversation_id changes
- ✅ **Context reset on conversation change** - Clean session cycling

### 7.2 Architecture Validation ✅

#### 7.2.1 Distributed Deployment Proven
- ✅ **Bridge and agent can run on different systems**
- ✅ **WebSocket communication handles network issues**
- ✅ **Service discovery via configuration**
- ✅ **Connection resilience with exponential backoff**

#### 7.2.2 Development Efficiency Achieved
- ✅ **Standalone mode enables rapid development**
- ✅ **Debug interface allows controlled testing**
- ✅ **Same processing pipeline in both modes**
- ✅ **No ROS dependencies for agent development**

#### 7.2.3 Prompt Management Sophistication
- ✅ **Barney robot personality (5,604 chars) vs simple assistant (259 chars)**
- ✅ **Adult users get command/visual mode with arm presets**
- ✅ **Child users get simplified conversational interface** 
- ✅ **Runtime switching between personalities works flawlessly**

### 7.3 Ready for Production Testing ✅

**Current Status**: The OpenAI Realtime Agent is fully implemented with:
- Complete WebSocket-based architecture
- Sophisticated named prompt system with runtime switching
- Robust debug interface for development
- Comprehensive error handling and recovery
- Full integration with ROS AI Bridge

**Next Steps**: 
1. Deploy with actual OpenAI API key for live testing
2. Test with real ROS bridge and audio input
3. Validate end-to-end robot integration
4. Performance tune session cycling parameters
5. Add monitoring and analytics for production use

### 6.3 Edge Case Tests
1. **Rapid Speaking Test**: Continuous speech at rotation boundary
2. **Network Failure Test**: Recovery from disconnections
3. **Context Overflow Test**: Graceful handling of very long conversations

## 7. Success Criteria

### 7.1 Conversational Quality
- **Natural Flow**: 95% of users report conversations feel natural and continuous
- **Context Preservation**: < 2% context loss during session boundaries
- **Interruption Responsiveness**: < 200ms from user speech to audio cutoff
- **Multi-Turn Coherence**: Maintains personality and context across extended conversations

### 7.2 Technical Performance
- **Audio Latency**: < 100ms from user speech end to LLM response start
- **Session Reliability**: 99.9% successful session management operations
- **Interruption Accuracy**: Zero false interruption triggers, 95% successful real interruptions
- **Multi-Agent Coordination**: Simultaneous conversation and command processing without conflicts

### 7.3 System Integration
- **Bridge Connectivity**: 99.9% uptime for WebSocket bridge connections
- **ROS Topic Reliability**: < 0.1% message loss rate across all topics
- **Agent Fault Tolerance**: Automatic recovery from agent crashes within 5 seconds
- **Configuration Flexibility**: Support for runtime prompt switching and parameter updates

### 7.4 Cost Optimization (Secondary Goal)
- **Session Efficiency**: Linear cost scaling for conversations > 5 minutes
- **Resource Utilization**: < $0.50 per 10-minute conversation in typical usage
- **Pause Detection Accuracy**: 95% accurate detection of conversation boundaries for session cycling

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
[1:00-1:10] SILENCE (no voice_chunks, LLM complete)
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