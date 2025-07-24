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
        topic_name="/speech_chunks", 
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
      - topic: "/speech_chunks"
        msg_type: "audio_common_msgs/AudioData"
      - topic: "/camera/image_raw"  
        msg_type: "sensor_msgs/Image"
        max_rate_hz: 5  # Rate limiting
        
    # Topics to publish (Agent → ROS)
    published_topics:
      - topic: "/audio_out"
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

## 10. Deployment

### 10.1 Launch Integration
```xml
<launch>
  <node pkg="by_your_command" exec="ros_ai_bridge" name="ros_ai_bridge">
    <param from="$(find-pkg-share by_your_command)/config/bridge_config.yaml"/>
  </node>
</launch>
```

### 10.2 Agent Integration
```python
# Agent connects to bridge queues
bridge = ROSAIBridge()
bridge_queues = bridge.get_queues()

async def agent_main():
    while True:
        # Receive MessageEnvelope from bridge
        envelope = await bridge_queues.inbound_topics.get()
        
        # Direct access to ROS message (zero-copy)
        if envelope.ros_msg_type == "audio_common_msgs/AudioData":
            audio_data = envelope.raw_data.int16_data
            # Process and send response back to ROS
            response_msg = create_response(audio_data)
            await bridge_queues.outbound_topics.put({
                "topic": "/audio_out",
                "msg": response_msg,
                "msg_type": "audio_common_msgs/AudioData"
            })
```

This minimal bridge design keeps the transport layer focused and allows agents to handle all business logic, session management, and LLM integration independently.