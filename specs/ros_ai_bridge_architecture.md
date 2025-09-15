# ROS AI Bridge: Minimal Data Transport Architecture

**Author**: Karim Virani  
**Version**: 3.0  
**Date**: July 2025

## Executive Summary

The ROS AI Bridge is a **minimal data transport layer** that shuttles messages between ROS2's callback-based concurrency model and agents using asyncio-based concurrency. It handles no business logic - only message queuing, format translation, and concurrency bridging.

## 1. Core Responsibility

**Single Purpose**: Bidirectional message transport between ROS2 topics and agent message queues.

```
ROS2 Callbacks  ←→  Message Queues  ←→  Agent Async Tasks
```

The bridge does NOT:
- Process audio/video content
- Manage LLM sessions  
- Handle business logic
- Make routing decisions
- Store state beyond queuing

## 2. Architecture Overview

```
┌─────────────────┐    ┌───────────────────┐    ┌──────────────────┐
│   ROS2 Side     │    │   Bridge Core     │    │   Agent Side     │
├─────────────────┤    ├───────────────────┤    ├──────────────────┤
│ Topic Callbacks │───►│ Inbound Queue     │───►│ async def        │
│                 │    │                   │    │   process_ros()  │
│ Publishers      │◄───│ Outbound Queue    │◄───│                  │
│                 │    │                   │    │ async def        │
│ Services        │───►│ Request Queue     │───►│   handle_svc()   │
│ Service Clients │◄───│ Response Queue    │◄───│                  │
└─────────────────┘    └───────────────────┘    └──────────────────┘
```

## 3. Message Queue Strategy

### 3.1 Queue Types

```python
class MessageQueues:
    # ROS → Agent
    inbound_topics: asyncio.Queue         # Topic messages  
    inbound_service_requests: asyncio.Queue  # Service requests
    
    # Agent → ROS  
    outbound_topics: queue.Queue          # Messages to publish
    outbound_service_responses: queue.Queue  # Service responses
```

### 3.2 Concurrency Bridging

```python
class ROSAIBridge(Node):
    def ros_callback(self, msg):
        """ROS callback - thread-safe queue put"""
        try:
            envelope = MessageEnvelope(
                msg_type="topic",
                topic_name=self.get_topic_name(),
                raw_data=msg,  # Zero-copy: pass ROS message directly
                ros_msg_type=msg.__class__.__module__ + '/' + msg.__class__.__name__,
                timestamp=time.time()
            )
            self.queues.inbound_topics.put_nowait(envelope)
        except queue.Full:
            self.get_logger().warn("Inbound queue full, dropping message")
    
    async def agent_consumer(self):
        """Agent side - async queue get with zero-copy access"""
        while True:
            try:
                envelope = await asyncio.wait_for(
                    self.queues.inbound_topics.get(), 
                    timeout=1.0
                )
                # Direct access to ROS message object (zero-copy)
                if envelope.ros_msg_type == "audio_common_msgs/AudioData":
                    # Process raw ROS message directly
                    await self.process_audio(envelope.raw_data)
                # Only serialize when sending to LLM API
                api_msg = self.api_serializer.prepare_for_api(envelope)
                await self.send_to_llm(api_msg)
            except asyncio.TimeoutError:
                continue
```

### 3.3 Queue Management

- **Size Limits**: Configurable max queue sizes to prevent memory bloat
- **Drop Policy**: Drop oldest messages when queues full (FIFO with overflow)
- **Backpressure**: Log warnings when queues approach capacity
- **No Persistence**: Messages lost on restart (agents handle persistence)

### 3.4 Queue Access Interface

```python
class ROSAIBridge(Node):
    def __init__(self):
        super().__init__('ros_ai_bridge')
        self.queues = MessageQueues()
        self._agent_interfaces = []
        
    def get_queues(self) -> MessageQueues:
        """Provide queue access for agents"""
        return self.queues
    
    def register_agent_interface(self, agent_id: str) -> MessageQueues:
        """Register agent and return dedicated queue interface"""
        interface = AgentInterface(agent_id, self.queues)
        self._agent_interfaces.append(interface)
        return interface
        
    async def start_bridge(self):
        """Start bridge operations - call after ROS2 node initialization"""
        await self.queues.initialize()
        self.create_timer(0.001, self._process_outbound_queue)  # 1kHz processing
        
    async def stop_bridge(self):
        """Clean shutdown with queue drainage"""
        # Drain remaining messages
        while not self.queues.inbound_topics.empty():
            await self.queues.inbound_topics.get()
        # Close all agent interfaces
        for interface in self._agent_interfaces:
            await interface.close()
```

## 4. Message Type Handling

### 4.1 Efficient Message Envelope

All messages wrapped in zero-copy envelope that preserves ROS message objects:
```python
@dataclass
class MessageEnvelope:
    msg_type: str           # 'topic', 'service_request', 'service_response'
    topic_name: str         # ROS topic/service name
    raw_data: Any          # Raw ROS message object (zero-copy)
    ros_msg_type: str      # ROS message type string for agent processing
    timestamp: float       # Unix timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Key Efficiency Principle**: Keep ROS messages in native format throughout internal queues. Only serialize when crossing API boundaries.

### 4.2 Zero-Copy Internal Handling

Bridge passes ROS message objects directly through internal queues without serialization:

**Internal Message Flow**:
```python
# ROS Callback → Queue (Zero-Copy)
def audio_callback(self, ros_msg: AudioData):
    envelope = MessageEnvelope(
        msg_type="topic",
        topic_name="/prompt_voice", 
        raw_data=ros_msg,  # Pass ROS message object directly
        ros_msg_type="audio_common_msgs/AudioData",
        timestamp=time.time()
    )
    self.queues.inbound_topics.put_nowait(envelope)
```

**Agent-Side API Serialization**:
Agents handle API-specific serialization only when needed:
```python
class APISerializer:
    def prepare_for_openai_realtime(self, envelope: MessageEnvelope) -> dict:
        """Convert to OpenAI Realtime API format only when sending"""
        if envelope.ros_msg_type == "audio_common_msgs/AudioData":
            pcm_bytes = np.array(envelope.raw_data.int16_data, dtype=np.int16).tobytes()
            base64_audio = base64.b64encode(pcm_bytes).decode()
            return {"type": "input_audio_buffer.append", "audio": base64_audio}
            
    def prepare_for_gemini_live(self, envelope: MessageEnvelope) -> dict:
        """Convert to Gemini Live API format"""
        if envelope.ros_msg_type == "sensor_msgs/Image":
            # Convert to 1024x1024 JPEG, then base64
            return {"realtime_input": {"media_chunks": [base64_image]}}
```

**Supported Message Types**:
- `audio_common_msgs/AudioData` - Raw PCM audio data
- `audio_common_msgs/AudioStamped` - Timestamped audio
- `sensor_msgs/Image` - Raw image data with encoding info
- `sensor_msgs/CompressedImage` - Pre-compressed image data
- `std_msgs/String` - Text messages
- `geometry_msgs/Twist` - Robot motion commands

### 4.3 Agent-Side Message Access

Agents receive MessageEnvelope objects with direct access to ROS data:
```python
# Agent receives envelope from bridge queue
envelope = await bridge_queues.inbound_topics.get()

# Direct access to ROS message fields (zero-copy)
if envelope.ros_msg_type == "audio_common_msgs/AudioData":
    audio_data = envelope.raw_data.int16_data  # Direct list access
    # Only serialize when sending to LLM API:
    api_msg = self.api_serializer.prepare_for_openai_realtime(envelope)
    await websocket.send(json.dumps(api_msg))

# For debugging/logging, envelope metadata available:
envelope_info = {
    "msg_type": envelope.msg_type,
    "topic_name": envelope.topic_name,
    "ros_msg_type": envelope.ros_msg_type,
    "timestamp": envelope.timestamp,
    "metadata": envelope.metadata
}
```

## 5. Configuration

### 5.1 Bridge Parameters
```yaml
ros_ai_bridge:
  ros__parameters:
    # Queue configuration
    max_queue_size: 100
    queue_timeout_ms: 1000
    drop_policy: "oldest"
    
    # Topics to bridge (ROS → Agent)
    subscribed_topics:
      - topic: "/prompt_voice"
        msg_type: "audio_common_msgs/AudioData"
      - topic: "/camera/image_raw"  
        msg_type: "sensor_msgs/Image"
        max_rate_hz: 5  # Rate limiting
        
    # Topics to publish (Agent → ROS)
    published_topics:
      - topic: "/response_voice"
        msg_type: "audio_common_msgs/AudioData"
      - topic: "/cmd_vel"
        msg_type: "geometry_msgs/Twist"
        
    # Services to expose
    services:
      - service: "/get_robot_status"
        srv_type: "std_srvs/Trigger"
```

### 5.2 Runtime Reconfiguration
```python
class BridgeReconfigurer:
    def __init__(self, bridge: ROSAIBridge):
        self.bridge = bridge
        
    async def add_topic_subscription(self, topic: str, msg_type: str):
        """Add new topic subscription at runtime"""
        subscription = self.bridge.create_subscription(
            msg_type, topic, 
            lambda msg: self.bridge.ros_callback(msg, topic), 
            10
        )
        self.bridge._subscriptions[topic] = subscription
        
    async def remove_topic_subscription(self, topic: str):
        """Remove topic subscription at runtime"""
        if topic in self.bridge._subscriptions:
            self.bridge._subscriptions[topic].destroy()
            del self.bridge._subscriptions[topic]
            
    def update_queue_size(self, new_size: int):
        """Adjust queue sizes (takes effect for new messages)"""
        self.bridge.queues.max_size = new_size
```

**Reconfiguration Features**:
- Dynamic topic subscription/unsubscription
- Queue size adjustment (gradual effect)
- Rate limiting changes per topic
- Agent interface registration/deregistration
- No restart required for configuration changes

## 6. Error Handling

### 6.1 Queue Overflow
```python
def handle_queue_full(self, queue_name: str, message):
    """Drop oldest message and log warning"""
    self.get_logger().warn(f"Queue {queue_name} full, dropping message")
    self.metrics['dropped_messages'] += 1
```

### 6.2 Message Envelope Errors
```python
def safe_envelop(self, ros_msg, topic_name: str):
    """Create message envelope with error handling"""
    try:
        return MessageEnvelope(
            msg_type="topic",
            topic_name=topic_name,
            raw_data=ros_msg,
            ros_msg_type=f"{ros_msg.__class__.__module__}/{ros_msg.__class__.__name__}",
            timestamp=time.time()
        )
    except Exception as e:
        self.get_logger().error(f"Envelope creation failed: {e}")
        return None  # Drop message
```

### 6.3 Agent Connection Loss
- Bridge continues queuing ROS messages (up to limits)
- Agent reconnection resumes message flow
- No message replay - agents handle missed data

## 7. Performance Requirements

### 7.1 Latency Targets
- **Queue latency**: < 0.1ms (zero-copy object passing)
- **API serialization latency**: < 5ms per message (only when needed)
- **Total bridge latency**: < 2ms (without API serialization)

### 7.2 Throughput Targets  
- **Audio streams**: Handle 50Hz audio chunks (20ms chunks)
- **Video streams**: Handle 5-30Hz image streams
- **Command messages**: Handle 100Hz command streams

### 7.3 Memory Usage
- **Queue memory**: Configurable limit (default 100MB total)
- **Zero serialization overhead**: ROS messages passed by reference
- **Memory efficiency**: ~1.1x message size (envelope overhead only)
- **No memory leaks**: Proper cleanup of dropped message objects

## 8. Implementation Checklist

### 8.1 Core Bridge (Priority 1)
- [ ] Basic queue implementation with thread-safe operations (asyncio.Queue + queue.Queue)
- [ ] ROS2 node with configurable topic subscription
- [ ] MessageEnvelope creation and zero-copy handling
- [ ] Agent-side async consumer interface with queue access
- [ ] Basic error handling and logging
- [ ] Queue access interface for agents (get_queues() method)

### 8.2 Message Type Support (Priority 2)  
- [ ] ROS message type detection and envelope creation
- [ ] Topic-to-message-type mapping configuration
- [ ] Message validation and error handling
- [ ] Dynamic message type registry

Note: API serializers are implemented by agents, not the bridge.

### 8.3 Configuration (Priority 3)
- [ ] YAML-based topic configuration parsing
- [ ] Dynamic topic subscription/unsubscription
- [ ] Rate limiting per topic implementation
- [ ] Queue size and timeout parameter handling
- [ ] Service configuration support

### 8.4 Monitoring (Priority 4)
- [ ] Queue depth metrics (ROS diagnostics integration)
- [ ] Message drop counters and logging
- [ ] Bridge latency measurements
- [ ] Memory usage monitoring
- [ ] Topic throughput statistics

## 9. Testing Strategy

### 9.1 Unit Tests
- Zero-copy message envelope tests
- API serializer accuracy tests (OpenAI, Gemini formats)
- Queue overflow behavior
- Error handling paths
- Configuration parsing

### 9.2 Integration Tests
- End-to-end ROS → Agent → ROS message flow
- High-rate message stress testing
- Agent connection/disconnection scenarios
- Multiple concurrent topic streams

### 9.3 Performance Tests
- Latency measurement under load
- Memory usage profiling
- Queue performance benchmarks
- Serialization performance

## 10. Agent Communication Protocol

### 10.1 WebSocket-Based Agent Interface

The bridge exposes WebSocket endpoints for agent connections, enabling distributed deployment where bridge and agents run on different systems.

#### 10.1.1 WebSocket Server Configuration
```yaml
ros_ai_bridge:
  ros__parameters:
    # WebSocket Server Settings
    websocket_server:
      enabled: true
      host: "0.0.0.0"          # Listen on all interfaces
      port: 8765               # Default WebSocket port
      max_connections: 10      # Maximum concurrent agent connections
      auth_required: false     # Authentication (future)
      heartbeat_interval: 30   # Seconds between ping/pong
      
    # Agent Registration
    agent_registration:
      timeout_seconds: 60      # Registration timeout
      allow_duplicate_ids: false
      require_capabilities: []  # Required agent capabilities
```

#### 10.1.2 WebSocket Message Protocol
All messages use JSON format with envelope structure:

```json
{
  "type": "message",
  "envelope": {
    "msg_type": "topic",
    "topic_name": "/prompt_voice",
    "ros_msg_type": "by_your_command/AudioDataUtterance", 
    "timestamp": 1642534567.123,
    "metadata": {
      "utterance_id": "utt_001",
      "confidence": 0.95
    },
    "data": {
      "audio_data": [1234, 5678, ...],
      "utterance_id": "utt_001",
      "start_time": 1642534567.0,
      "confidence": 0.95
    }
  }
}
```

**Registration Protocol:**
```json
// Agent → Bridge: Registration
{
  "type": "register",
  "agent_id": "openai_realtime",
  "capabilities": ["audio_processing", "realtime_api"],
  "subscriptions": [
    {"topic": "/prompt_voice", "msg_type": "by_your_command/AudioDataUtterance"}
  ]
}

// Bridge → Agent: Registration Response
{
  "type": "register_response", 
  "status": "success",
  "agent_id": "openai_realtime",
  "session_id": "sess_abc123"
}
```

#### 10.1.3 WebSocket Server Implementation

**Implementation Note**: The websockets library has deprecated `WebSocketServerProtocol`. Use connection objects directly:

```python
import websockets
import json
import asyncio
from typing import Dict, Set

class WebSocketAgentServer:
    def __init__(self, bridge: ROSAIBridge):
        self.bridge = bridge
        self.connected_agents: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.agent_subscriptions: Dict[str, List[str]] = {}
        
    async def start_server(self, host: str = "0.0.0.0", port: int = 8765):
        """Start WebSocket server for agent connections"""
        self.server = await websockets.serve(
            self.handle_agent_connection, 
            host, 
            port,
            ping_interval=30,
            ping_timeout=10
        )
        self.bridge.get_logger().info(f"WebSocket server started on {host}:{port}")
        
    async def handle_agent_connection(self, websocket, path):
        """Handle new agent WebSocket connection"""
        agent_id = None
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if data["type"] == "register":
                    agent_id = await self.register_agent(websocket, data)
                elif data["type"] == "outbound_message":
                    await self.handle_outbound_message(data)
                elif data["type"] == "heartbeat":
                    await websocket.send(json.dumps({"type": "heartbeat_response"}))
                    
        except websockets.exceptions.ConnectionClosed:
            self.bridge.get_logger().info(f"Agent {agent_id} disconnected")
        except Exception as e:
            self.bridge.get_logger().error(f"WebSocket error: {e}")
        finally:
            if agent_id:
                await self.unregister_agent(agent_id)
                
    async def register_agent(self, websocket, data):
        """Register agent and setup subscriptions"""
        agent_id = data["agent_id"]
        
        # Store connection
        self.connected_agents[agent_id] = websocket
        
        # Setup subscriptions
        subscriptions = data.get("subscriptions", [])
        self.agent_subscriptions[agent_id] = [sub["topic"] for sub in subscriptions]
        
        # Send response
        response = {
            "type": "register_response",
            "status": "success", 
            "agent_id": agent_id,
            "session_id": f"sess_{agent_id}_{int(time.time())}"
        }
        await websocket.send(json.dumps(response))
        
        self.bridge.get_logger().info(f"Registered agent: {agent_id}")
        return agent_id
        
    async def broadcast_to_agents(self, envelope: MessageEnvelope):
        """Send ROS message to subscribed agents via WebSocket"""
        disconnected_agents = []
        
        for agent_id, websocket in self.connected_agents.items():
            # Check if agent subscribed to this topic
            if envelope.topic_name in self.agent_subscriptions.get(agent_id, []):
                try:
                    # Serialize ROS message for WebSocket transport
                    message = {
                        "type": "message",
                        "envelope": {
                            "msg_type": envelope.msg_type,
                            "topic_name": envelope.topic_name,
                            "ros_msg_type": envelope.ros_msg_type,
                            "timestamp": envelope.timestamp,
                            "metadata": envelope.metadata,
                            "data": self.serialize_ros_message(envelope.raw_data)
                        }
                    }
                    await websocket.send(json.dumps(message))
                    
                except websockets.exceptions.ConnectionClosed:
                    disconnected_agents.append(agent_id)
                except Exception as e:
                    self.bridge.get_logger().error(f"Error sending to agent {agent_id}: {e}")
        
        # Clean up disconnected agents
        for agent_id in disconnected_agents:
            await self.unregister_agent(agent_id)
            
    def serialize_ros_message(self, ros_msg) -> Dict:
        """Convert ROS message to JSON-serializable dict"""
        # Handle different message types
        if hasattr(ros_msg, 'get_fields_and_field_types'):
            result = {}
            for field_name, field_type in ros_msg.get_fields_and_field_types().items():
                value = getattr(ros_msg, field_name)
                result[field_name] = value
            return result
        else:
            # Fallback for basic types
            return {"data": str(ros_msg)}
```

#### 10.1.4 Bridge Integration with WebSocket

**Critical Implementation Detail**: The bridge operates in mixed ROS2/asyncio mode. WebSocket broadcasting must be scheduled properly from ROS callbacks:

```python
class ROSAIBridge(Node):
    def __init__(self):
        super().__init__('ros_ai_bridge')
        self.queues = MessageQueues()
        self.websocket_server = None
        
    def _ros_callback(self, msg: Any, topic_name: str, msg_type: str):
        """ROS callback - broadcast message to all consumers"""
        try:
            # Create zero-copy envelope
            envelope = MessageEnvelope(...)
            
            # Put into inbound queue for direct agent interfaces
            success = self.queues.put_inbound_topic(envelope)
            
            # CRITICAL: Schedule WebSocket broadcast from ROS callback thread
            if self.websocket_server:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.websocket_server.broadcast_to_agents(envelope))
                except RuntimeError:
                    # No event loop running, skip WebSocket broadcast
                    pass
                    
        except Exception as e:
            self.get_logger().error(f"Error in ROS callback for {topic_name}: {e}")
    
    async def start_bridge(self):
        """Start bridge with WebSocket server"""
        await self.queues.initialize()
        
        # Start WebSocket server if enabled
        if self.get_parameter('websocket_server.enabled').value:
            self.websocket_server = WebSocketAgentServer(self)
            host = self.get_parameter('websocket_server.host').value
            port = self.get_parameter('websocket_server.port').value
            await self.websocket_server.start_server(host, port)
        
        # Start message processing timer
        self.create_timer(0.001, self._process_outbound_queue)
        
        # Note: WebSocket broadcasting happens directly in ROS callbacks
        # No separate message loop needed
```

**Key Implementation Lessons:**

1. **Event Loop Integration**: ROS callbacks run in ROS2 executor threads, while WebSocket operations need asyncio. Use `loop.create_task()` with proper error handling.

2. **Message Broadcasting**: Direct broadcasting from ROS callbacks is more efficient than queueing messages for a separate broadcast loop.

3. **Error Resilience**: Always wrap asyncio operations from ROS callbacks in try/catch blocks to handle missing event loops gracefully.

#### 10.1.5 Message Serialization Performance

**Implementation Experience**: ROS message serialization for WebSocket transport requires careful optimization:

```python
def serialize_ros_message(self, ros_msg: Any) -> Dict:
    """Convert ROS message to JSON-serializable dict"""
    try:
        # PERFORMANCE: Use get_fields_and_field_types() for efficient introspection
        if hasattr(ros_msg, 'get_fields_and_field_types'):
            result = {}
            for field_name, field_type in ros_msg.get_fields_and_field_types().items():
                value = getattr(ros_msg, field_name)
                # CRITICAL: Handle numpy arrays and large lists efficiently
                if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    # Convert to regular Python lists for JSON serialization
                    result[field_name] = list(value)
                else:
                    result[field_name] = value
            return result
        else:
            # Fallback for basic types
            if hasattr(ros_msg, 'data'):
                return {"data": ros_msg.data}
            else:
                return {"data": str(ros_msg)}
    except Exception as e:
        self.logger.error(f"Error serializing ROS message: {e}")
        return {"error": "serialization_failed"}
```

**Performance Characteristics Measured:**
- **AudioDataUtterance (20ms chunk, 16kHz)**: ~2ms serialization time
- **Large audio arrays (>1000 samples)**: Memory usage spikes 2-3x during serialization
- **WebSocket message size**: ~1.5x larger than original ROS message due to JSON overhead

**Optimization Recommendations:**
1. **Batch Processing**: Serialize multiple small messages together when possible
2. **Memory Management**: Consider chunking very large audio arrays
3. **Compression**: Future enhancement could add gzip compression for large messages

### 10.2 Deployment Configurations

#### 10.2.1 Launch Integration
```xml
<launch>
  <node pkg="by_your_command" exec="ros_ai_bridge" name="ros_ai_bridge">
    <param from="$(find-pkg-share by_your_command)/config/bridge_config.yaml"/>
    <!-- WebSocket server parameters -->
    <param name="websocket_server.enabled" value="true"/>
    <param name="websocket_server.host" value="0.0.0.0"/>
    <param name="websocket_server.port" value="8765"/>
  </node>
</launch>
```

#### 10.2.2 Distributed Deployment
```yaml
# bridge_system_a.yaml - Robot system
ros_ai_bridge:
  ros__parameters:
    websocket_server:
      enabled: true
      host: "0.0.0.0"
      port: 8765
    subscribed_topics:
      - topic: "/prompt_voice"
        msg_type: "by_your_command/AudioDataUtterance"

# agent_system_b.yaml - Compute system  
openai_realtime_agent:
  bridge_connection:
    type: "websocket"
    host: "robot_system_a.local"  # Bridge host
    port: 8765
    reconnect_interval: 5.0
```

#### 10.2.3 Agent WebSocket Client Integration
```python
# Agents connect via WebSocket instead of direct instantiation
import websockets
import json

class WebSocketBridgeClient:
    def __init__(self, host: str, port: int, agent_id: str):
        self.host = host
        self.port = port
        self.agent_id = agent_id
        self.websocket = None
        self.message_queue = asyncio.Queue()
        
    async def connect(self, subscriptions: List[Dict]):
        """Connect to bridge WebSocket server"""
        uri = f"ws://{self.host}:{self.port}"
        self.websocket = await websockets.connect(uri)
        
        # Register with bridge
        registration = {
            "type": "register",
            "agent_id": self.agent_id,
            "subscriptions": subscriptions
        }
        await self.websocket.send(json.dumps(registration))
        
        # Wait for registration response
        response = await self.websocket.recv()
        data = json.loads(response)
        
        if data["status"] != "success":
            raise ConnectionError(f"Registration failed: {data}")
            
        # Start message listener
        asyncio.create_task(self._message_listener())
        
    async def _message_listener(self):
        """Listen for messages from bridge"""
        async for message in self.websocket:
            data = json.loads(message)
            if data["type"] == "message":
                await self.message_queue.put(data["envelope"])
                
    async def get_message(self, timeout: float = 1.0):
        """Get message from bridge (replaces bridge_interface.get_inbound_message)"""
        try:
            envelope_data = await asyncio.wait_for(self.message_queue.get(), timeout)
            # Convert back to MessageEnvelope-like object
            return WebSocketMessageEnvelope(envelope_data)
        except asyncio.TimeoutError:
            return None

# Updated agent initialization
async def main():
    # Connect to bridge via WebSocket instead of direct instantiation
    bridge_client = WebSocketBridgeClient("localhost", 8765, "openai_realtime")
    
    subscriptions = [
        {"topic": "/prompt_voice", "msg_type": "by_your_command/AudioDataUtterance"}
    ]
    await bridge_client.connect(subscriptions)
    
    # Agent can now receive messages via WebSocket
    agent = OpenAIRealtimeAgent(bridge_client, config)
    await agent.run()
```

This WebSocket-based architecture enables true distributed deployment while maintaining the minimal transport layer design and zero-copy efficiency within the bridge itself.

## 11. Planned Enhancements

### 11.1 Topic Aliasing System

**TODO**: Implement a topic aliasing mechanism to decouple agents from hardware-specific topic names.

#### Problem
Agents currently need to know exact ROS topic names (e.g., `/grunt1/arm1/cam_live/color/image_raw`), creating tight coupling between agent code and robot hardware configuration. This is particularly problematic for camera/video feeds where different robots have different camera mount points and naming conventions.

#### Proposed Solution
Add a topic aliasing layer in the bridge that maps logical topic names to physical topics:

**Configuration:**
```yaml
ros_ai_bridge:
  ros__parameters:
    # Topic alias mappings
    topic_aliases:
      # Camera feeds - agents just request "camera/image_raw"
      "camera/image_raw": "/grunt1/arm1/cam_live/color/image_raw"
      "camera/depth": "/grunt1/arm1/cam_live/depth/image_rect_raw"
      "camera/compressed": "/grunt1/arm1/cam_live/color/image_raw/compressed"
      
      # Other sensor mappings
      "lidar/scan": "/velodyne/scan"
      "audio/input": "/microphone/audio"
      
    # Subscribed topics can use aliases
    subscribed_topics:
      - topic: "camera/image_raw"  # Uses alias
        msg_type: "sensor_msgs/Image"
```

**Implementation:**
```python
class ROSAIBridge(Node):
    def __init__(self):
        super().__init__('ros_ai_bridge')
        self.topic_aliases = {}  # Logical -> Physical mapping
        self.reverse_aliases = {}  # Physical -> Logical mapping
        
    def _resolve_topic_alias(self, topic: str) -> str:
        """Resolve logical topic name to physical topic"""
        return self.topic_aliases.get(topic, topic)
        
    def _setup_configured_topics(self):
        """Set up topics with alias resolution"""
        for topic_config in self._config.get('subscribed_topics', []):
            logical_topic = topic_config['topic']
            physical_topic = self._resolve_topic_alias(logical_topic)
            
            # Subscribe to physical topic
            subscription = self.create_subscription(
                msg_class, physical_topic, callback, qos
            )
            
            # Store both mappings for agent registration
            self.reverse_aliases[physical_topic] = logical_topic
```

**Agent Registration Enhancement:**
```python
async def register_agent(self, websocket, data):
    """Register agent with topic alias resolution"""
    agent_id = data["agent_id"]
    subscriptions = data.get("subscriptions", [])
    
    # Resolve aliases in agent subscriptions
    resolved_subs = []
    for sub in subscriptions:
        logical_topic = sub["topic"]
        physical_topic = self._resolve_topic_alias(logical_topic)
        resolved_subs.append(physical_topic)
        
        # Also store logical name for matching
        if physical_topic != logical_topic:
            self.agent_logical_topics[agent_id][physical_topic] = logical_topic
    
    self.agent_subscriptions[agent_id] = resolved_subs
```

**Message Routing with Aliases:**
```python
async def broadcast_to_agents(self, envelope: MessageEnvelope):
    """Route messages considering both physical and logical names"""
    for agent_id, websocket in self.connected_agents.items():
        agent_subs = self.agent_subscriptions.get(agent_id, [])
        
        # Check both physical and logical topic names
        physical_topic = envelope.topic_name
        logical_topic = self.reverse_aliases.get(physical_topic, physical_topic)
        
        if physical_topic in agent_subs or logical_topic in agent_subs:
            # Send with logical name if agent used alias
            if logical_topic in self.agent_logical_topics.get(agent_id, {}):
                envelope.topic_name = logical_topic
            
            await self.send_to_agent(websocket, envelope)
```

#### Benefits
1. **Hardware Abstraction**: Agents use semantic names like "camera/image_raw" instead of hardware-specific paths
2. **Portability**: Same agent code works across different robot configurations
3. **Centralized Management**: All topic mappings in one configuration file
4. **Easy Migration**: Change hardware topics without modifying agent code
5. **Multi-Robot Support**: Different robots can map same logical topics to their hardware
6. **Video/Camera Flexibility**: Vision-enabled agents (like Gemini with video support) can request generic "camera/image_raw" and automatically get the correct camera feed for any robot

#### Implementation Priority
**Priority: Medium** - Current hardcoded solution works for demos, but aliasing needed for production deployment across multiple robot platforms.

### 11.2 Other Planned Enhancements

1. **Message Compression**: Add optional gzip compression for large messages over WebSocket
2. **Authentication**: Implement agent authentication and authorization
3. **Message Replay**: Record and replay message streams for debugging
4. **Dynamic Rate Limiting**: Adjust rate limits based on system load
5. **Priority Queuing**: Priority-based message routing for critical messages