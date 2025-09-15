#!/usr/bin/env python3
"""
ROS AI Bridge: Minimal Data Transport Architecture

A minimal data transport layer that shuttles messages between ROS2's callback-based 
concurrency model and agents using asyncio-based concurrency. Handles no business logic - 
only message queuing, format translation, and concurrency bridging.

Author: Karim Virani
Version: 3.0
Date: July 2025
"""

import asyncio
import queue
import time
import threading
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Set
import yaml
import importlib
from datetime import datetime
from collections import defaultdict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# WebSocket support
try:
    import websockets
    from websockets import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = None

def get_timestamp():
    """Get formatted timestamp HH:MM:SS.mmm"""
    now = datetime.now()
    return now.strftime("%H:%M:%S.%f")[:-3]


@dataclass
class MessageEnvelope:
    """Zero-copy envelope that preserves ROS message objects"""
    msg_type: str           # 'topic', 'service_request', 'service_response'
    topic_name: str         # ROS topic/service name
    raw_data: Any          # Raw ROS message object (zero-copy)
    ros_msg_type: str      # ROS message type string for agent processing
    timestamp: float       # Unix timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)


class FrameRateLimiter:
    """Rate limiter for high-bandwidth topics like video streams
    
    Implements per-topic frame rate limiting as a safety net for
    high-frequency data streams. Only the latest frame is kept
    when rate limiting occurs.
    """
    
    def __init__(self, max_fps: float = 5.0):
        """Initialize frame rate limiter
        
        Args:
            max_fps: Maximum frames per second to allow (default 5 fps)
        """
        self.max_fps = max_fps
        self.min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self.last_pass_time = 0.0
        self.dropped_count = 0
        self.passed_count = 0
        
    def should_pass(self, current_time: Optional[float] = None) -> bool:
        """Check if this frame should pass through based on rate limit
        
        Args:
            current_time: Current timestamp (uses time.time() if not provided)
            
        Returns:
            True if frame should pass, False if it should be dropped
        """
        if self.min_interval <= 0:
            # No rate limiting
            self.passed_count += 1
            return True
            
        if current_time is None:
            current_time = time.time()
            
        time_since_last = current_time - self.last_pass_time
        
        if time_since_last >= self.min_interval:
            # Enough time has passed, allow this frame
            self.last_pass_time = current_time
            self.passed_count += 1
            return True
        else:
            # Too soon, drop this frame
            self.dropped_count += 1
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            'max_fps': self.max_fps,
            'min_interval': self.min_interval,
            'passed_count': self.passed_count,
            'dropped_count': self.dropped_count,
            'drop_rate': self.dropped_count / (self.passed_count + self.dropped_count) 
                        if (self.passed_count + self.dropped_count) > 0 else 0.0
        }
        
    def reset_stats(self):
        """Reset statistics counters"""
        self.dropped_count = 0
        self.passed_count = 0


class MessageQueues:
    """Thread-safe message queues for bridging ROS2 and asyncio concurrency models"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        
        # ROS → Agent (asyncio queues for agent consumers)
        self.inbound_topics: asyncio.Queue = None
        self.inbound_service_requests: asyncio.Queue = None
        
        # Lock for queue manipulation during frame replacement
        self.queue_lock = asyncio.Lock()
        
        # Agent → ROS (thread-safe queues for ROS publishers)
        self.outbound_topics = queue.Queue(maxsize=max_size)
        self.outbound_service_responses = queue.Queue(maxsize=max_size)
        
        # Metrics
        self.dropped_messages = 0
        self.total_messages = 0
        
    async def initialize(self):
        """Initialize asyncio queues (must be called from async context)"""
        self.inbound_topics = asyncio.Queue(maxsize=self.max_size)
        self.inbound_service_requests = asyncio.Queue(maxsize=self.max_size)
        
    def put_inbound_topic(self, envelope: MessageEnvelope) -> bool:
        """Put message into inbound topic queue (called from ROS callback)
        
        For image topics, removes old frames from same topic before adding new one.
        """
        try:
            if self.inbound_topics is None:
                return False
                
            # For image topics, we need to remove old frames from same topic
            if "Image" in envelope.ros_msg_type:  # Handles both Image and CompressedImage
                # Since we're in a sync context (ROS callback), we can't directly
                # manipulate the async queue. Instead, we'll use a simple approach:
                # Try to add the new frame. If queue is full, clear it and retry.
                try:
                    self.inbound_topics.put_nowait(envelope)
                    self.total_messages += 1
                    return True
                except asyncio.QueueFull:
                    # Queue is full - for images, clear old messages and retry
                    # This is aggressive but ensures fresh frames
                    cleared = self._clear_old_messages_sync()
                    if cleared > 0:
                        # Log only occasionally to avoid spam
                        if self.total_messages % 100 == 0:
                            print(f"[MessageQueues] Cleared {cleared} old messages for new image frame")
                    try:
                        self.inbound_topics.put_nowait(envelope)
                        self.total_messages += 1
                        return True
                    except asyncio.QueueFull:
                        self.dropped_messages += 1
                        return False
            else:
                # Regular messages go into queue normally
                self.inbound_topics.put_nowait(envelope)
                self.total_messages += 1
                return True
        except asyncio.QueueFull:
            self.dropped_messages += 1
            return False
            
    def _clear_old_messages_sync(self):
        """Clear old messages from queue (called from sync context)
        
        This is a simple approach that removes older messages to make room.
        Returns the number of messages cleared.
        """
        # Remove up to half the queue to make room
        cleared = 0
        max_to_clear = self.max_size // 2
        while cleared < max_to_clear:
            try:
                self.inbound_topics.get_nowait()
                cleared += 1
            except asyncio.QueueEmpty:
                break
        if cleared > 0:
            self.dropped_messages += cleared
        return cleared
            
    def put_inbound_topic_broadcast(self, envelope: MessageEnvelope) -> bool:
        """Put message into inbound queue for broadcasting to all consumers"""
        # This will be used by the bridge to broadcast to both direct agents and WebSocket agents
        return self.put_inbound_topic(envelope)
            
    def put_outbound_topic(self, topic: str, msg: Any, msg_type: str) -> bool:
        """Put message into outbound topic queue (called from agent)"""
        try:
            self.outbound_topics.put_nowait({
                'topic': topic,
                'msg': msg,
                'msg_type': msg_type,
                'timestamp': time.time()
            })
            self.total_messages += 1
            return True
        except queue.Full:
            self.dropped_messages += 1
            return False
            
    def get_outbound_topic(self, timeout: float = 0.0) -> Optional[Dict]:
        """Get message from outbound topic queue (called from ROS timer)"""
        try:
            return self.outbound_topics.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_metrics(self) -> Dict[str, int]:
        """Get queue metrics"""
        inbound_size = self.inbound_topics.qsize() if self.inbound_topics else 0
        return {
            'inbound_queue_size': inbound_size,
            'outbound_queue_size': self.outbound_topics.qsize(),
            'dropped_messages': self.dropped_messages,
            'total_messages': self.total_messages
        }


class AgentInterface:
    """Interface for agents to access bridge queues with tracking"""
    
    def __init__(self, agent_id: str, queues: MessageQueues):
        self.agent_id = agent_id
        self.queues = queues
        self.active = True
        self.messages_processed = 0
        
    async def get_inbound_message(self, timeout: float = 1.0) -> Optional[MessageEnvelope]:
        """Get message from inbound queue with timeout"""
        if not self.active:
            return None
        try:
            envelope = await asyncio.wait_for(
                self.queues.inbound_topics.get(), 
                timeout=timeout
            )
            self.messages_processed += 1
            return envelope
        except asyncio.TimeoutError:
            return None
            
    def put_outbound_message(self, topic: str, msg: Any, msg_type: str) -> bool:
        """Put message into outbound queue"""
        if not self.active:
            return False
        return self.queues.put_outbound_topic(topic, msg, msg_type)
        
    async def close(self):
        """Close agent interface"""
        self.active = False


class BridgeReconfigurer:
    """Runtime reconfiguration support for the bridge"""
    
    def __init__(self, bridge: 'ROSAIBridge'):
        self.bridge = bridge
        
    def add_topic_subscription(self, topic: str, msg_type: str, qos_profile: Optional[QoSProfile] = None):
        """Add new topic subscription at runtime"""
        if topic in self.bridge._topic_subscriptions:
            self.bridge.log_warning(f"Topic {topic} already subscribed")
            return
            
        try:
            # Import message type
            msg_class = self._import_message_type(msg_type)
            if not msg_class:
                return
                
            # Create QoS profile
            if qos_profile is None:
                qos_profile = QoSProfile(
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=10,
                    reliability=QoSReliabilityPolicy.BEST_EFFORT
                )
                
            # Create subscription
            subscription = self.bridge.create_subscription(
                msg_class, 
                topic, 
                lambda msg, t=topic, mt=msg_type: self.bridge._ros_callback(msg, t, mt),
                qos_profile
            )
            
            self.bridge._topic_subscriptions[topic] = subscription
            self.bridge.log_info(f"Added subscription to {topic} ({msg_type})")
            
        except Exception as e:
            import traceback
            self.bridge.log_error(f"Failed to add subscription {topic}: {e}")
            self.bridge.log_error(f"Full traceback: {traceback.format_exc()}")
            
    def remove_topic_subscription(self, topic: str):
        """Remove topic subscription at runtime"""
        if topic not in self.bridge._topic_subscriptions:
            self.bridge.log_warning(f"Topic {topic} not subscribed")
            return
            
        try:
            self.bridge._topic_subscriptions[topic].destroy()
            del self.bridge._topic_subscriptions[topic]
            self.bridge.log_info(f"Removed subscription to {topic}")
        except Exception as e:
            self.bridge.log_error(f"Failed to remove subscription {topic}: {e}")
            
    def update_queue_size(self, new_size: int):
        """Adjust queue sizes (takes effect for new queues)"""
        self.bridge.queues.max_size = new_size
        self.bridge.log_info(f"Updated queue max size to {new_size}")
        
    def _import_message_type(self, msg_type: str):
        """Import ROS message type dynamically"""
        try:
            # Parse message type (e.g., "std_msgs/String" -> std_msgs.msg.String)
            if '/' in msg_type:
                package, message = msg_type.split('/', 1)
                module_name = f"{package}.msg"
                module = importlib.import_module(module_name)
                return getattr(module, message)
            else:
                self.bridge.log_error(f"Invalid message type format: {msg_type}")
                return None
        except Exception as e:
            self.bridge.log_error(f"Failed to import message type {msg_type}: {e}")
            return None


class WebSocketAgentServer:
    """WebSocket server for agent connections with bridge integration"""
    
    def __init__(self, bridge: 'ROSAIBridge'):
        self.bridge = bridge
        self.server = None
        self.connected_agents: Dict[str, WebSocketServerProtocol] = {}
        self.agent_subscriptions: Dict[str, List[str]] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.logger = bridge  # Use bridge's logging methods
        
    def _get_full_prefix(self) -> str:
        """Get the full namespace prefix for topics"""
        parts = []
        namespace = self.bridge._config.get('namespace', '')
        prefix = self.bridge._config.get('prefix', '')
        
        if namespace:
            parts.append(namespace)
        if prefix:
            parts.append(prefix)
            
        return '/'.join(parts) if parts else ''
        
    async def start_server(self, host: str = "0.0.0.0", port: int = 8765):
        """Start WebSocket server for agent connections"""
        if not WEBSOCKETS_AVAILABLE:
            self.logger.log_error("WebSocket support not available. Install websockets package.")
            return False
            
        try:
            self.server = await websockets.serve(
                self.handle_agent_connection,
                host,
                port,
                ping_interval=30,
                ping_timeout=10,
                max_size=10**6,  # 1MB max message size
                max_queue=32
            )
            self.logger.log_info(f"WebSocket server started on {host}:{port}")
            return True
        except Exception as e:
            self.logger.log_error(f"Failed to start WebSocket server: {e}")
            return False
            
    async def stop_server(self):
        """Stop WebSocket server and disconnect all agents"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.logger.log_info("WebSocket server stopped")
            
        # Disconnect all agents
        for agent_id in list(self.connected_agents.keys()):
            await self.unregister_agent(agent_id)
            
    async def handle_agent_connection(self, websocket: WebSocketServerProtocol):
        """Handle new agent WebSocket connection"""
        agent_id = None
        remote_address = websocket.remote_address
        self.logger.log_info(f"New WebSocket connection from {remote_address}")
        
        try:
            async for raw_message in websocket:
                try:
                    message = raw_message
                    if isinstance(message, bytes):
                        message = message.decode('utf-8')
                    
                    data = json.loads(message)
                    message_type = data.get("type", "")
                    
                    if message_type == "register":
                        agent_id = await self.register_agent(websocket, data)
                    elif message_type == "outbound_message":
                        await self.handle_outbound_message(data)
                    elif message_type == "heartbeat":
                        await websocket.send(json.dumps({"type": "heartbeat_response"}))
                    else:
                        self.logger.log_warning(f"Unknown message type from {agent_id}: {message_type}")
                        
                except json.JSONDecodeError as e:
                    self.logger.log_error(f"Invalid JSON from {agent_id}: {e}")
                except Exception as e:
                    self.logger.log_error(f"Error processing message from {agent_id}: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.log_info(f"Agent {agent_id} disconnected normally")
        except Exception as e:
            self.logger.log_error(f"WebSocket connection error for {agent_id}: {e}")
        finally:
            if agent_id:
                await self.unregister_agent(agent_id)
                
    async def register_agent(self, websocket: WebSocketServerProtocol, data: Dict) -> str:
        """Register agent and setup subscriptions"""
        agent_id = data.get("agent_id", "")
        if not agent_id:
            await websocket.send(json.dumps({
                "type": "register_response",
                "status": "error",
                "message": "agent_id required"
            }))
            return None
            
        # Check for duplicate IDs
        if agent_id in self.connected_agents:
            await websocket.send(json.dumps({
                "type": "register_response", 
                "status": "error",
                "message": f"agent_id '{agent_id}' already registered"
            }))
            return None
            
        # Store connection
        self.connected_agents[agent_id] = websocket
        
        # Store capabilities
        capabilities = data.get("capabilities", [])
        self.agent_capabilities[agent_id] = capabilities
        
        # Setup subscriptions
        subscriptions = data.get("subscriptions", [])
        self.agent_subscriptions[agent_id] = [sub["topic"] for sub in subscriptions]
        
        # Send success response with namespace info
        response = {
            "type": "register_response",
            "status": "success",
            "agent_id": agent_id,
            "session_id": f"sess_{agent_id}_{int(time.time())}",
            "namespace_info": {
                "namespace": self.bridge._config.get('namespace', ''),
                "prefix": self.bridge._config.get('prefix', ''),
                "full_prefix": self._get_full_prefix()
            }
        }
        await websocket.send(json.dumps(response))
        
        self.logger.log_info(f"Registered agent: {agent_id} with capabilities: {capabilities}")
        self.logger.log_info(f"Agent {agent_id} subscribed to topics: {self.agent_subscriptions[agent_id]}")
        
        return agent_id
        
    async def unregister_agent(self, agent_id: str):
        """Unregister agent and clean up resources"""
        # Check if already unregistered to avoid spam
        if agent_id not in self.connected_agents:
            return  # Already unregistered
            
        if agent_id in self.connected_agents:
            try:
                websocket = self.connected_agents[agent_id]
                # Check if websocket has closed attribute (websockets library compatibility)
                if hasattr(websocket, 'closed') and not websocket.closed:
                    await websocket.close()
                elif hasattr(websocket, 'state') and websocket.state.value <= 1:  # CONNECTING=0, OPEN=1
                    await websocket.close()
            except Exception as e:
                self.logger.log_error(f"Error closing WebSocket for {agent_id}: {e}")
                
            del self.connected_agents[agent_id]
            
        if agent_id in self.agent_subscriptions:
            del self.agent_subscriptions[agent_id]
            
        if agent_id in self.agent_capabilities:
            del self.agent_capabilities[agent_id]
            
        self.logger.log_info(f"Unregistered agent: {agent_id}")
        
    async def handle_outbound_message(self, data: Dict):
        """Handle outbound message from agent to ROS"""
        self.logger.log_info(f"📥 Received outbound message: topic={data.get('topic')}, type={data.get('msg_type')}")
        try:
            topic = data.get("topic", "")
            msg_type = data.get("msg_type", "")
            msg_data = data.get("data", {})
            
            if not topic or not msg_type:
                self.logger.log_error("Outbound message missing topic or msg_type")
                return
                
            # Convert message data back to ROS message object
            ros_msg = self.deserialize_message_data(msg_data, msg_type)
            if ros_msg is None:
                self.logger.log_error(f"Failed to deserialize message for {topic}")
                return
                
            # Put into bridge outbound queue
            self.logger.log_info(f"📤 Queueing outbound message for topic: {topic}")
            success = self.bridge.queues.put_outbound_topic(topic, ros_msg, msg_type)
            if success:
                self.logger.log_info(f"✅ Successfully queued outbound message for {topic}")
            else:
                self.logger.log_warning(f"Failed to queue outbound message for {topic}")
                
        except Exception as e:
            self.logger.log_error(f"Error handling outbound message: {e}")
            
    def deserialize_message_data(self, data: Dict, msg_type: str) -> Any:
        """Convert JSON data back to ROS message object"""
        try:
            # Import message class
            msg_class = self.bridge.reconfigurer._import_message_type(msg_type)
            if not msg_class:
                return None
                
            # Create message instance
            if hasattr(msg_class, 'get_fields_and_field_types'):
                # Use ROS2 message structure
                ros_msg = msg_class()
                for field_name, field_type in msg_class.get_fields_and_field_types().items():
                    if field_name in data:
                        setattr(ros_msg, field_name, data[field_name])
                return ros_msg
            else:
                # Fallback for simple messages
                ros_msg = msg_class()
                if hasattr(ros_msg, 'data') and 'data' in data:
                    ros_msg.data = data['data']
                return ros_msg
                
        except Exception as e:
            self.logger.log_error(f"Error deserializing message data for {msg_type}: {e}")
            return None
            
    async def broadcast_to_agents(self, envelope: MessageEnvelope):
        """Send ROS message to subscribed agents via WebSocket"""
        # Only log non-image broadcasts to reduce spam
        if "Image" not in envelope.ros_msg_type:
            self.logger.log_info(f"🔊 Broadcasting {envelope.ros_msg_type} from {envelope.topic_name} to {len(self.connected_agents)} agents")
        
        if not self.connected_agents:
            # Silent return for images, warn for other messages
            if "Image" not in envelope.ros_msg_type:
                self.logger.log_warning("No connected agents to broadcast to")
            return
            
        disconnected_agents = []
        
        # Create a copy to avoid concurrent modification
        agents_copy = dict(self.connected_agents)
        for agent_id, websocket in agents_copy.items():
            # Check if agent subscribed to this topic
            agent_subscriptions = self.agent_subscriptions.get(agent_id, [])
            
            # Extract base topic name from full namespaced topic
            # e.g., /robot1/voice/audio_out -> audio_out
            base_topic = envelope.topic_name.split('/')[-1] if '/' in envelope.topic_name else envelope.topic_name
            
            # Also check if agent subscribed with the base topic name
            matched = False
            for sub_topic in agent_subscriptions:
                # Match either:
                # 1. Exact match (including absolute paths)
                # 2. Base topic match
                # 3. Relative topic match (agent sub without /, topic with /)
                if (envelope.topic_name == sub_topic or 
                    base_topic == sub_topic.lstrip('/') or
                    envelope.topic_name.endswith('/' + sub_topic)):
                    matched = True
                    break
                    
            # Log subscription checking (including images when triggered)
            triggered = envelope.metadata and envelope.metadata.get('triggered_by') == 'voice_detection'
            if "Image" not in envelope.ros_msg_type or triggered:
                self.logger.log_info(f"Agent {agent_id} subscriptions: {agent_subscriptions}, checking {envelope.topic_name} (base: {base_topic})")
            
            if matched:
                # Log sending (including triggered images)
                if "Image" not in envelope.ros_msg_type or triggered:
                    self.logger.log_info(f"📤 Sending to agent {agent_id}: {envelope.topic_name}")
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
                    
                    # Log when sending triggered images
                    if triggered:
                        self.logger.log_info(f"📨 Sending triggered frame to {agent_id} via WebSocket")
                    
                    await websocket.send(json.dumps(message))
                    
                    if triggered:
                        self.logger.log_info(f"✅ Successfully sent triggered frame to {agent_id}")
                    
                except websockets.exceptions.ConnectionClosed:
                    disconnected_agents.append(agent_id)
                except Exception as e:
                    self.logger.log_error(f"Error sending message to agent {agent_id}: {e}")
            else:
                self.logger.log_warning(f"❌ NOT sending to agent {agent_id} - no subscription match for {envelope.topic_name}")
                    
        # Clean up disconnected agents
        for agent_id in disconnected_agents:
            await self.unregister_agent(agent_id)
            
    def serialize_ros_message(self, ros_msg: Any) -> Dict:
        """Convert ROS message to JSON-serializable dict"""
        try:
            # Handle different message types
            if hasattr(ros_msg, 'get_fields_and_field_types'):
                result = {}
                for field_name, field_type in ros_msg.get_fields_and_field_types().items():
                    value = getattr(ros_msg, field_name)
                    
                    # Debug logging for audio data fields
                    if field_name == 'int16_data':
                        self.logger.log_info(f"🎧 Bridge serializing int16_data: type={type(value)}, length={len(value) if hasattr(value, '__len__') else 'N/A'}")
                    
                    # Handle special types that need conversion
                    if value is None:
                        result[field_name] = None
                    elif field_name == 'data' and ros_msg.__class__.__name__ in ['Image', 'CompressedImage']:
                        # Use base64 for compressed images to avoid huge JSON arrays
                        if ros_msg.__class__.__name__ == 'CompressedImage':
                            import base64
                            # Convert bytes to base64 string (much more efficient than JSON array)
                            result[field_name] = base64.b64encode(value).decode('utf-8')
                        else:
                            # Skip large raw image data
                            result[field_name] = f"<image_data_{len(value)}_bytes>"
                    elif hasattr(value, 'get_fields_and_field_types'):
                        # Nested ROS message (like Header, Time) - recursively serialize
                        result[field_name] = self.serialize_ros_message(value)
                    elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                        # Convert lists/arrays to regular Python lists
                        result[field_name] = list(value)
                    elif hasattr(value, 'sec') and hasattr(value, 'nanosec'):
                        # Handle Time objects specifically
                        result[field_name] = {'sec': value.sec, 'nanosec': value.nanosec}
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
            self.logger.log_error(f"Error serializing ROS message: {e}")
            return {"error": "serialization_failed"}
            
    def get_agent_status(self) -> Dict:
        """Get status of all connected agents"""
        return {
            "connected_agents": list(self.connected_agents.keys()),
            "agent_subscriptions": self.agent_subscriptions.copy(),
            "agent_capabilities": self.agent_capabilities.copy()
        }


class ROSAIBridge(Node):
    """
    Minimal ROS2-to-Agent bridge that handles only data transport.
    
    Core responsibilities:
    - Zero-copy message passing between ROS2 callbacks and agent async tasks
    - Thread-safe queue management
    - Configuration-based topic subscription
    - Agent interface management
    """
    
    def __init__(self, node_name: str = 'ros_ai_bridge'):
        super().__init__(node_name)
        
        # Core components
        self.queues = MessageQueues()
        self._agent_interfaces: List[AgentInterface] = []
        self._topic_subscriptions: Dict[str, Any] = {}  # Our tracking dict
        self._topic_publishers: Dict[str, Any] = {}     # Our tracking dict
        self._topic_rate_limiters: Dict[str, FrameRateLimiter] = {}  # Per-topic rate limiters
        self._latest_frames: Dict[str, Tuple[Any, float]] = {}  # Store latest frame per topic
        self._voice_chunk_counter = 0  # Track voice chunks for continuous forwarding
        self.reconfigurer = BridgeReconfigurer(self)
        
        # WebSocket server
        self.websocket_server: Optional[WebSocketAgentServer] = None
        
        # Configuration
        self._declare_parameters()
        self._config = self._load_configuration()
        
        # State
        self._started = False
        self._shutdown_event = threading.Event()
        
        # Store reference to asyncio event loop for thread-safe operations
        self.asyncio_loop = None
        
        # Initialize logger level
        self.log_info("ROS AI Bridge initialized")
        
    def log_info(self, msg):
        """Log with custom format"""
        self.get_logger().info(f"[{get_timestamp()}] [bridge] {msg}")
        
    def log_debug(self, msg):
        """Debug log with custom format"""
        self.get_logger().debug(f"[{get_timestamp()}] [bridge] DEBUG: {msg}")
            
    def log_warning(self, msg):
        """Warning log with custom format"""
        self.get_logger().warning(f"[{get_timestamp()}] [bridge] WARNING: {msg}")
        
    def log_error(self, msg):
        """Error log with custom format"""
        self.get_logger().error(f"[{get_timestamp()}] [bridge] ERROR: {msg}")
        
    def _declare_parameters(self):
        """Declare ROS2 parameters with defaults"""
        # Queue configuration
        self.declare_parameter('max_queue_size', 100)
        self.declare_parameter('queue_timeout_ms', 1000)
        self.declare_parameter('drop_policy', 'oldest')
        
        # Video frame rate limiting (legacy - can be disabled by setting to 0)
        self.declare_parameter('max_video_fps', 2.0)  # Global rate limit for video topics
        
        # Smart frame forwarding configuration
        self.declare_parameter('frame_forwarding.enabled', True)
        self.declare_parameter('frame_forwarding.trigger_on_voice', True)
        self.declare_parameter('frame_forwarding.trigger_on_text', True)
        self.declare_parameter('frame_forwarding.continuous_nth_frame', 5)
        self.declare_parameter('frame_forwarding.max_frame_age_ms', 1000)
        
        # WebSocket server configuration
        self.declare_parameter('websocket_server.enabled', False)
        self.declare_parameter('websocket_server.host', '0.0.0.0')
        self.declare_parameter('websocket_server.port', 8765)
        self.declare_parameter('websocket_server.max_connections', 10)
        self.declare_parameter('websocket_server.heartbeat_interval', 30)
        
        # Agent registration configuration
        self.declare_parameter('agent_registration.timeout_seconds', 60)
        self.declare_parameter('agent_registration.allow_duplicate_ids', False)
        
        # Namespace configuration
        self.declare_parameter('namespace', '')
        self.declare_parameter('prefix', '')
        
        # Topic configuration - will be loaded from YAML if provided
        self.declare_parameter('config_file', '')
        
    def _load_configuration(self) -> Dict:
        """Load bridge configuration from parameters and config file"""
        config = {
            'namespace': self.get_parameter('namespace').value,
            'prefix': self.get_parameter('prefix').value,
            'max_queue_size': self.get_parameter('max_queue_size').value,
            'queue_timeout_ms': self.get_parameter('queue_timeout_ms').value,
            'drop_policy': self.get_parameter('drop_policy').value,
            'max_video_fps': self.get_parameter('max_video_fps').value,
            'subscribed_topics': [],
            'published_topics': [],
            'services': [],
            'rate_limits': {},
            
            # WebSocket server configuration
            'websocket_server': {
                'enabled': self.get_parameter('websocket_server.enabled').value,
                'host': self.get_parameter('websocket_server.host').value,
                'port': self.get_parameter('websocket_server.port').value,
                'max_connections': self.get_parameter('websocket_server.max_connections').value,
                'heartbeat_interval': self.get_parameter('websocket_server.heartbeat_interval').value
            },
            
            # Agent registration configuration
            'agent_registration': {
                'timeout_seconds': self.get_parameter('agent_registration.timeout_seconds').value,
                'allow_duplicate_ids': self.get_parameter('agent_registration.allow_duplicate_ids').value
            },
            
            # Smart frame forwarding configuration
            'frame_forwarding': {
                'enabled': self.get_parameter('frame_forwarding.enabled').value,
                'trigger_on_voice': self.get_parameter('frame_forwarding.trigger_on_voice').value,
                'trigger_on_text': self.get_parameter('frame_forwarding.trigger_on_text').value,
                'continuous_nth_frame': self.get_parameter('frame_forwarding.continuous_nth_frame').value,
                'max_frame_age_ms': self.get_parameter('frame_forwarding.max_frame_age_ms').value
            }
        }
        
        # Load from config file if specified
        config_file = self.get_parameter('config_file').value
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if 'ros_ai_bridge' in file_config:
                        bridge_config = file_config['ros_ai_bridge']
                        if 'ros__parameters' in bridge_config:
                            # Merge configuration carefully
                            params = bridge_config['ros__parameters']
                            for key, value in params.items():
                                if key in ['subscribed_topics', 'published_topics', 'services']:
                                    # These should be lists, ensure they are
                                    self.log_info(f"Processing {key}: type={type(value)}, value={value}")
                                    if isinstance(value, list):
                                        config[key] = value
                                    else:
                                        self.log_error(f"Configuration key '{key}' should be a list, got {type(value)}")
                                else:
                                    config[key] = value
                            
                            self.log_info(f"Loaded configuration from {config_file}")
                            self.log_debug(f"Subscribed topics: {config['subscribed_topics']}")
                            self.log_debug(f"Published topics: {config['published_topics']}")
            except Exception as e:
                self.log_error(f"Failed to load config file {config_file}: {e}")
                
        return config
        
    def _construct_topic_name(self, base_topic: str) -> str:
        """Construct full topic name with namespace and prefix handling empty values"""
        # If already absolute (starts with /), return as-is - absolute topics override namespace/prefix
        if base_topic.startswith('/'):
            return base_topic
            
        # Build topic parts
        parts = []
        
        # Always start with / for absolute topics
        if not base_topic.startswith('/'):
            parts.append('')  # This creates leading / when joined
            
        # Add namespace if present
        namespace = self._config.get('namespace', '')
        if namespace:
            parts.append(namespace)
            
        # Add prefix if present
        prefix = self._config.get('prefix', '')
        if prefix:
            parts.append(prefix)
            
        # Add base topic (remove leading / if present)
        base = base_topic.lstrip('/')
        parts.append(base)
        
        # Join with / and handle multiple slashes
        full_topic = '/'.join(parts)
        
        # Clean up any double slashes except at the beginning
        while '//' in full_topic[1:]:
            full_topic = full_topic[:1] + full_topic[1:].replace('//', '/', 1)
            
        return full_topic
        
    async def start_bridge(self):
        """Start bridge operations - call after ROS2 node initialization"""
        if self._started:
            self.log_warning("Bridge already started")
            return
            
        try:
            # Store reference to current asyncio loop for thread-safe operations
            self.asyncio_loop = asyncio.get_event_loop()
            self.log_info(f"Stored asyncio loop reference: {id(self.asyncio_loop)}")
            
            # Initialize async queues
            await self.queues.initialize()
            
            # Update queue size from configuration
            self.queues.max_size = self._config['max_queue_size']
            
            # Set up configured topics
            self._setup_configured_topics()
            
            # Start WebSocket server if enabled
            if self._config['websocket_server']['enabled']:
                self.websocket_server = WebSocketAgentServer(self)
                host = self._config['websocket_server']['host']
                port = self._config['websocket_server']['port']
                success = await self.websocket_server.start_server(host, port)
                if success:
                    self.log_info(f"WebSocket server enabled on {host}:{port}")
                else:
                    self.log_error("Failed to start WebSocket server")
            else:
                self.log_info("WebSocket server disabled")
            
            # Start outbound message processing timer (1kHz)
            self.create_timer(0.001, self._process_outbound_queue)
            
            # Start metrics logging timer (1Hz)
            self.create_timer(1.0, self._log_metrics)
            
            # WebSocket broadcasting is handled directly in ROS callbacks
            
            self._started = True
            self.log_info("ROS AI Bridge started successfully")
            
        except Exception as e:
            self.log_error(f"Failed to start bridge: {e}")
            raise
            
    def _setup_configured_topics(self):
        """Set up topic subscriptions and publishers from configuration"""
        self.log_info(f"Setting up topics: subscribed_topics type={type(self._config.get('subscribed_topics'))}, value={self._config.get('subscribed_topics')}")
        
        # Load rate limits from separate configuration section
        rate_limits = self._config.get('rate_limits', {})
        
        # Set up subscriptions
        subscribed_topics = self._config.get('subscribed_topics', [])
        self.log_info(f"Setting up {len(subscribed_topics)} subscriptions")
        for topic_config in subscribed_topics:
            base_topic = topic_config['topic']
            full_topic = self._construct_topic_name(base_topic)
            msg_type = topic_config['msg_type']
            self.log_info(f"Processing subscription config: {topic_config}")
            
            # Check if rate limiting is configured for this topic
            # First check inline max_fps for backwards compatibility
            max_fps = topic_config.get('max_fps', 0)
            # Then check the separate rate_limits section
            if max_fps == 0 and full_topic in rate_limits:
                max_fps = rate_limits[full_topic]
            # Apply global video rate limit to Image topics (both regular and compressed)
            if max_fps == 0 and "Image" in msg_type:
                max_fps = self._config.get('max_video_fps', 2.0)
                self.log_info(f"Applying global video rate limit to {full_topic}: {max_fps} fps")
            
            if max_fps > 0:
                self._topic_rate_limiters[full_topic] = FrameRateLimiter(max_fps)
                self.log_info(f"Rate limiting enabled for {full_topic}: max {max_fps} fps")
            
            # Create QoS profile
            qos_profile = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=topic_config.get('queue_depth', 10),
                reliability=QoSReliabilityPolicy.BEST_EFFORT
            )
            
            self.log_info(f"Subscribing to {full_topic} (base: {base_topic})")
            self.reconfigurer.add_topic_subscription(full_topic, msg_type, qos_profile)
            
        # Set up publishers
        published_topics = self._config.get('published_topics', [])
        self.log_info(f"Setting up {len(published_topics)} publishers")
        for topic_config in published_topics:
            base_topic = topic_config['topic']
            full_topic = self._construct_topic_name(base_topic)
            msg_type = topic_config['msg_type']
            self.log_info(f"Processing publisher config: {topic_config}")
            
            try:
                msg_class = self.reconfigurer._import_message_type(msg_type)
                if msg_class:
                    publisher = self.create_publisher(msg_class, full_topic, 10)
                    # Store with base topic as key for agent compatibility
                    self._topic_publishers[base_topic] = {
                        'publisher': publisher,
                        'msg_class': msg_class,
                        'msg_type': msg_type,
                        'full_topic': full_topic
                    }
                    self.log_info(f"Created publisher for {full_topic} (base: {base_topic}, type: {msg_type})")
            except Exception as e:
                import traceback
                self.log_error(f"Failed to create publisher {full_topic}: {e}")
                self.log_error(f"Full traceback: {traceback.format_exc()}")
                
    def _ros_callback(self, msg: Any, topic_name: str, msg_type: str):
        """ROS callback - smart frame forwarding based on voice/text triggers"""
        try:
            # Check if smart frame forwarding is enabled
            frame_forwarding_enabled = self._config.get('frame_forwarding', {}).get('enabled', True)
            
            # Handle image frames with smart caching
            if "Image" in msg_type and frame_forwarding_enabled:
                # Always store the latest frame (no rate limiting on storage)
                # Store with its message type for proper forwarding
                self._latest_frames[topic_name] = (msg, time.time(), msg_type)
                # Log frame caching (reduce spam by logging every 10th frame)
                if not hasattr(self, '_frame_cache_counter'):
                    self._frame_cache_counter = 0
                self._frame_cache_counter += 1
                if self._frame_cache_counter % 10 == 0:
                    self.log_debug(f"📸 Cached frame from {topic_name} (total cached topics: {len(self._latest_frames)})")
                
                # Check if we should use legacy rate limiting (max_video_fps > 0)
                if self._config.get('max_video_fps', 0) > 0:
                    # Legacy mode: use rate limiting
                    if topic_name in self._topic_rate_limiters:
                        limiter = self._topic_rate_limiters[topic_name]
                        if not limiter.should_pass():
                            return  # Drop due to rate limiting
                    # Fall through to normal forwarding
                else:
                    # Smart mode: don't forward automatically, wait for triggers
                    return  # Frame cached, waiting for trigger
            
            # For legacy rate limiting on non-smart mode
            elif topic_name in self._topic_rate_limiters and not frame_forwarding_enabled:
                limiter = self._topic_rate_limiters[topic_name]
                if not limiter.should_pass():
                    return  # Drop due to rate limiting
            
            # Create zero-copy envelope for non-image or legacy mode
            envelope = MessageEnvelope(
                msg_type="topic",
                topic_name=topic_name,
                raw_data=msg,  # Zero-copy: pass ROS message directly
                ros_msg_type=msg_type,
                timestamp=time.time(),
                metadata={'source_node': self.get_name()}
            )
            
            # Check if this is a voice/text trigger
            is_trigger = False
            if frame_forwarding_enabled:
                trigger_on_voice = self._config.get('frame_forwarding', {}).get('trigger_on_voice', True)
                trigger_on_text = self._config.get('frame_forwarding', {}).get('trigger_on_text', True)
                continuous_nth = self._config.get('frame_forwarding', {}).get('continuous_nth_frame', 5)
                
                # Check for voice trigger (support both old and new names for compatibility)
                if trigger_on_voice and ('prompt_voice' in topic_name or 'voice_chunks' in topic_name):
                    # Check if this is the first chunk (chunk_sequence == 0)
                    try:
                        if hasattr(msg, 'chunk_sequence'):
                            if msg.chunk_sequence == 0:
                                # First chunk - always trigger
                                is_trigger = True
                                self._voice_chunk_counter = 0
                                self.log_info(f"🎯 First voice chunk detected - forwarding cached frames")
                            else:
                                # Subsequent chunks - forward every Nth
                                self._voice_chunk_counter += 1
                                if self._voice_chunk_counter % continuous_nth == 0:
                                    is_trigger = True
                                    self.log_info(f"🎯 Voice chunk #{self._voice_chunk_counter} - forwarding frames (every {continuous_nth})")
                        else:
                            # No chunk_sequence field, treat as trigger
                            is_trigger = True
                            self.log_info(f"🎯 Voice detected (no sequence) - forwarding cached frames")
                    except Exception as e:
                        self.log_debug(f"Error checking chunk_sequence: {e}")
                        is_trigger = True  # Default to triggering on error
                        
                # Check for text trigger (support both old and new names for compatibility)
                elif trigger_on_text and ('prompt_text' in topic_name or 'text_input' in topic_name):
                    is_trigger = True
                    self.log_info(f"🎯 Text input detected - forwarding cached frames")
            
            # Put into inbound queue for direct agent interfaces
            success = self.queues.put_inbound_topic(envelope)
            if not success and "Image" not in msg_type:
                # Only warn for non-images to reduce spam
                self.log_warning(f"Inbound queue full, dropped message from {topic_name}")
            
            # Also broadcast to WebSocket agents directly (non-blocking)
            if self.websocket_server and self.asyncio_loop:
                # Schedule WebSocket broadcast in the asyncio thread (thread-safe)
                # Only log non-image broadcasts to reduce spam
                if "Image" not in envelope.ros_msg_type:
                    self.log_info(f"📡 Broadcasting message to WebSocket agents: {envelope.ros_msg_type}")
                self.asyncio_loop.call_soon_threadsafe(
                    asyncio.create_task, 
                    self.websocket_server.broadcast_to_agents(envelope)
                )
                
                # If this was a trigger, also forward cached frames
                if is_trigger:
                    if self.asyncio_loop:
                        # Schedule the coroutine to run in the asyncio loop
                        future = asyncio.run_coroutine_threadsafe(
                            self._forward_cached_frames(),
                            self.asyncio_loop
                        )
                        self.log_info(f"📸 Scheduled frame forwarding task")
                    else:
                        self.log_error("❌ No asyncio loop available for frame forwarding!")
            elif self.websocket_server:
                self.log_warning("WebSocket server available but no asyncio loop reference")
            else:
                self.log_warning("No WebSocket server - cannot broadcast")
            
            if not success and "Image" not in msg_type:
                # Only warn for non-images to reduce spam
                self.log_warning(f"Inbound queue full, dropped message from {topic_name}")
                
        except Exception as e:
            self.log_error(f"Error in ROS callback for {topic_name}: {e}")
    
    async def _forward_cached_frames(self):
        """Forward all cached frames when triggered by voice/text"""
        try:
            self.log_info(f"📸 Starting frame forwarding from {len(self._latest_frames)} cached topics")
            
            if not self._latest_frames:
                self.log_warning("⚠️ No cached frames to forward")
                return
                
            max_age_ms = self._config.get('frame_forwarding', {}).get('max_frame_age_ms', 1000)
            max_age_sec = max_age_ms / 1000.0
            current_time = time.time()
            
            for topic_name, frame_data in self._latest_frames.items():
                # Unpack with proper handling for both old and new format
                if len(frame_data) == 3:
                    frame_msg, timestamp, ros_msg_type = frame_data
                else:
                    # Fallback for old format (shouldn't happen but safe)
                    frame_msg, timestamp = frame_data
                    ros_msg_type = "sensor_msgs/CompressedImage"
                    
                age = current_time - timestamp
                
                # Only forward frames that are fresh enough
                if age < max_age_sec:
                    # Create envelope for the cached frame
                    envelope = MessageEnvelope(
                        msg_type="topic",
                        topic_name=topic_name,
                        raw_data=frame_msg,
                        ros_msg_type=ros_msg_type,
                        timestamp=timestamp,
                        metadata={
                            'source_node': self.get_name(),
                            'triggered_by': 'voice_detection',
                            'frame_age_ms': int(age * 1000)
                        }
                    )
                    
                    # Broadcast to WebSocket agents
                    if self.websocket_server:
                        self.log_info(f"📸 Broadcasting frame from {topic_name} to agents...")
                        result = await self.websocket_server.broadcast_to_agents(envelope)
                        self.log_info(f"📸 Forwarded cached frame from {topic_name} (age: {int(age*1000)}ms)")
                else:
                    self.log_debug(f"Skipped stale frame from {topic_name} (age: {int(age*1000)}ms > {max_age_ms}ms)")
                    
        except Exception as e:
            self.log_error(f"Error forwarding cached frames: {e}")
            
    def _process_outbound_queue(self):
        """Process outbound messages from agents and publish to ROS topics"""
        try:
            # Process up to 10 messages per timer cycle to avoid blocking
            for _ in range(10):
                msg_data = self.queues.get_outbound_topic(timeout=0.0)
                if msg_data is None:
                    break
                
                self.log_info(f"📤 Processing outbound message from queue for topic: {msg_data['topic']}")
                topic = msg_data['topic']
                ros_msg = msg_data['msg']
                
                # Find publisher for this topic
                if topic in self._topic_publishers:
                    try:
                        self._topic_publishers[topic]['publisher'].publish(ros_msg)
                        self.log_info(f"✅ Published message to ROS topic: /{topic}")
                    except Exception as e:
                        self.log_error(f"Failed to publish to {topic}: {e}")
                else:
                    self.log_warning(f"No publisher configured for topic {topic}")
                    
        except Exception as e:
            self.log_error(f"Error processing outbound queue: {e}")
            
    def _log_metrics(self):
        """Log bridge metrics periodically"""
        metrics = self.queues.get_metrics()
        if metrics['total_messages'] > 0:
            self.log_info(
                f"Bridge metrics - Inbound: {metrics['inbound_queue_size']}, "
                f"Outbound: {metrics['outbound_queue_size']}, "
                f"Dropped: {metrics['dropped_messages']}, "
                f"Total: {metrics['total_messages']}"
            )
            
            # Add WebSocket agent metrics
            if self.websocket_server:
                agent_status = self.websocket_server.get_agent_status()
                connected_count = len(agent_status['connected_agents'])
                if connected_count > 0:
                    self.log_info(f"WebSocket agents: {connected_count} connected")
            
            # Log rate limiter stats if any are active
            for topic, limiter in self._topic_rate_limiters.items():
                stats = limiter.get_stats()
                if stats['passed_count'] > 0 or stats['dropped_count'] > 0:
                    self.log_info(
                        f"Rate limiter [{topic}]: {stats['max_fps']} fps max, "
                        f"passed: {stats['passed_count']}, dropped: {stats['dropped_count']} "
                        f"({stats['drop_rate']*100:.1f}% drop rate)"
                    )
    
            
    def get_queues(self) -> MessageQueues:
        """Provide direct queue access for simple agents"""
        return self.queues
        
    def register_agent_interface(self, agent_id: str) -> AgentInterface:
        """Register agent and return dedicated interface with tracking"""
        interface = AgentInterface(agent_id, self.queues)
        self._agent_interfaces.append(interface)
        self.log_info(f"Registered agent interface: {agent_id}")
        return interface
        
    def get_agent_interfaces(self) -> List[AgentInterface]:
        """Get list of active agent interfaces"""
        return [iface for iface in self._agent_interfaces if iface.active]
        
    async def stop_bridge(self):
        """Clean shutdown with queue drainage"""
        if not self._started:
            return
            
        self.log_info("Stopping ROS AI Bridge...")
        
        try:
            # Signal shutdown
            self._shutdown_event.set()
            
            # Stop WebSocket server
            if self.websocket_server:
                await self.websocket_server.stop_server()
                self.websocket_server = None
            
            # Drain remaining inbound messages
            if self.queues.inbound_topics:
                drained = 0
                while not self.queues.inbound_topics.empty():
                    try:
                        await asyncio.wait_for(self.queues.inbound_topics.get(), timeout=0.1)
                        drained += 1
                    except asyncio.TimeoutError:
                        break
                if drained > 0:
                    self.log_info(f"Drained {drained} inbound messages")
                    
            # Close all agent interfaces
            for interface in self._agent_interfaces:
                await interface.close()
                
            # Destroy subscriptions
            for subscription in self._topic_subscriptions.values():
                subscription.destroy()
            self._topic_subscriptions.clear()
            
            self._started = False
            self.log_info("ROS AI Bridge stopped")
            
        except Exception as e:
            self.log_error(f"Error stopping bridge: {e}")


async def run_bridge_async(bridge: ROSAIBridge):
    """Run the bridge in async mode for testing/standalone use"""
    try:
        await bridge.start_bridge()
        
        # Keep running until shutdown
        while not bridge._shutdown_event.is_set():
            await asyncio.sleep(0.1)
            
    except KeyboardInterrupt:
        bridge.get_logger().info("Received shutdown signal")
    finally:
        await bridge.stop_bridge()


def main(args=None):
    """Main entry point for ROS AI Bridge node"""
    rclpy.init(args=args)
    
    try:
        bridge = ROSAIBridge()
        
        # For ROS2 spinning, we need to run async operations in a separate thread
        # The bridge operates in mixed mode: ROS2 callbacks + async agent interfaces
        
        # Start bridge async operations in background
        loop = asyncio.new_event_loop()
        def run_async():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(bridge.start_bridge())
            loop.run_forever()
            
        async_thread = threading.Thread(target=run_async, daemon=True)
        async_thread.start()
        
        # Run ROS2 spinning in main thread
        bridge.get_logger().info("Starting ROS2 node spinning...")
        rclpy.spin(bridge)
        
    except KeyboardInterrupt:
        bridge.get_logger().info("Received Ctrl+C, shutting down...")
    except Exception as e:
        bridge.get_logger().error(f"Bridge error: {e}")
    finally:
        # Cleanup
        bridge._shutdown_event.set()
        if 'loop' in locals():
            loop.call_soon_threadsafe(loop.stop)
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()