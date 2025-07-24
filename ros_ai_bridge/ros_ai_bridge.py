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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import yaml
import importlib

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


@dataclass
class MessageEnvelope:
    """Zero-copy envelope that preserves ROS message objects"""
    msg_type: str           # 'topic', 'service_request', 'service_response'
    topic_name: str         # ROS topic/service name
    raw_data: Any          # Raw ROS message object (zero-copy)
    ros_msg_type: str      # ROS message type string for agent processing
    timestamp: float       # Unix timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)


class MessageQueues:
    """Thread-safe message queues for bridging ROS2 and asyncio concurrency models"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        
        # ROS → Agent (asyncio queues for agent consumers)
        self.inbound_topics: asyncio.Queue = None
        self.inbound_service_requests: asyncio.Queue = None
        
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
        """Put message into inbound topic queue (called from ROS callback)"""
        try:
            if self.inbound_topics is None:
                return False
            self.inbound_topics.put_nowait(envelope)
            self.total_messages += 1
            return True
        except asyncio.QueueFull:
            self.dropped_messages += 1
            return False
            
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
            self.bridge.get_logger().warn(f"Topic {topic} already subscribed")
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
            self.bridge.get_logger().info(f"Added subscription to {topic} ({msg_type})")
            
        except Exception as e:
            import traceback
            self.bridge.get_logger().error(f"Failed to add subscription {topic}: {e}")
            self.bridge.get_logger().error(f"Full traceback: {traceback.format_exc()}")
            
    def remove_topic_subscription(self, topic: str):
        """Remove topic subscription at runtime"""
        if topic not in self.bridge._topic_subscriptions:
            self.bridge.get_logger().warn(f"Topic {topic} not subscribed")
            return
            
        try:
            self.bridge._topic_subscriptions[topic].destroy()
            del self.bridge._topic_subscriptions[topic]
            self.bridge.get_logger().info(f"Removed subscription to {topic}")
        except Exception as e:
            self.bridge.get_logger().error(f"Failed to remove subscription {topic}: {e}")
            
    def update_queue_size(self, new_size: int):
        """Adjust queue sizes (takes effect for new queues)"""
        self.bridge.queues.max_size = new_size
        self.bridge.get_logger().info(f"Updated queue max size to {new_size}")
        
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
                self.bridge.get_logger().error(f"Invalid message type format: {msg_type}")
                return None
        except Exception as e:
            self.bridge.get_logger().error(f"Failed to import message type {msg_type}: {e}")
            return None


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
        self.reconfigurer = BridgeReconfigurer(self)
        
        # Configuration
        self._declare_parameters()
        self._config = self._load_configuration()
        
        # State
        self._started = False
        self._shutdown_event = threading.Event()
        
        # Initialize logger level
        self.get_logger().info("ROS AI Bridge initialized")
        
    def _declare_parameters(self):
        """Declare ROS2 parameters with defaults"""
        # Queue configuration
        self.declare_parameter('max_queue_size', 100)
        self.declare_parameter('queue_timeout_ms', 1000)
        self.declare_parameter('drop_policy', 'oldest')
        
        # Topic configuration - will be loaded from YAML if provided
        self.declare_parameter('config_file', '')
        
    def _load_configuration(self) -> Dict:
        """Load bridge configuration from parameters and config file"""
        config = {
            'max_queue_size': self.get_parameter('max_queue_size').value,
            'queue_timeout_ms': self.get_parameter('queue_timeout_ms').value,
            'drop_policy': self.get_parameter('drop_policy').value,
            'subscribed_topics': [],
            'published_topics': [],
            'services': []
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
                                    self.get_logger().info(f"Processing {key}: type={type(value)}, value={value}")
                                    if isinstance(value, list):
                                        config[key] = value
                                    else:
                                        self.get_logger().error(f"Configuration key '{key}' should be a list, got {type(value)}")
                                else:
                                    config[key] = value
                            
                            self.get_logger().info(f"Loaded configuration from {config_file}")
                            self.get_logger().debug(f"Subscribed topics: {config['subscribed_topics']}")
                            self.get_logger().debug(f"Published topics: {config['published_topics']}")
            except Exception as e:
                self.get_logger().error(f"Failed to load config file {config_file}: {e}")
                
        return config
        
    async def start_bridge(self):
        """Start bridge operations - call after ROS2 node initialization"""
        if self._started:
            self.get_logger().warn("Bridge already started")
            return
            
        try:
            # Initialize async queues
            await self.queues.initialize()
            
            # Update queue size from configuration
            self.queues.max_size = self._config['max_queue_size']
            
            # Set up configured topics
            self._setup_configured_topics()
            
            # Start outbound message processing timer (1kHz)
            self.create_timer(0.001, self._process_outbound_queue)
            
            # Start metrics logging timer (1Hz)
            self.create_timer(1.0, self._log_metrics)
            
            self._started = True
            self.get_logger().info("ROS AI Bridge started successfully")
            
        except Exception as e:
            self.get_logger().error(f"Failed to start bridge: {e}")
            raise
            
    def _setup_configured_topics(self):
        """Set up topic subscriptions and publishers from configuration"""
        self.get_logger().info(f"Setting up topics: subscribed_topics type={type(self._config.get('subscribed_topics'))}, value={self._config.get('subscribed_topics')}")
        # Set up subscriptions
        for topic_config in self._config.get('subscribed_topics', []):
            topic = topic_config['topic']
            msg_type = topic_config['msg_type']
            
            # Create QoS profile
            qos_profile = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=topic_config.get('queue_depth', 10),
                reliability=QoSReliabilityPolicy.BEST_EFFORT
            )
            
            self.reconfigurer.add_topic_subscription(topic, msg_type, qos_profile)
            
        # Set up publishers
        for topic_config in self._config.get('published_topics', []):
            topic = topic_config['topic']
            msg_type = topic_config['msg_type']
            
            try:
                msg_class = self.reconfigurer._import_message_type(msg_type)
                if msg_class:
                    publisher = self.create_publisher(msg_class, topic, 10)
                    self._topic_publishers[topic] = {
                        'publisher': publisher,
                        'msg_class': msg_class,
                        'msg_type': msg_type
                    }
                    self.get_logger().info(f"Created publisher for {topic} ({msg_type})")
            except Exception as e:
                import traceback
                self.get_logger().error(f"Failed to create publisher {topic}: {e}")
                self.get_logger().error(f"Full traceback: {traceback.format_exc()}")
                
    def _ros_callback(self, msg: Any, topic_name: str, msg_type: str):
        """ROS callback - thread-safe queue put with zero-copy envelope creation"""
        try:
            # Create zero-copy envelope
            envelope = MessageEnvelope(
                msg_type="topic",
                topic_name=topic_name,
                raw_data=msg,  # Zero-copy: pass ROS message directly
                ros_msg_type=msg_type,
                timestamp=time.time(),
                metadata={'source_node': self.get_name()}
            )
            
            # Put into inbound queue (thread-safe)
            success = self.queues.put_inbound_topic(envelope)
            
            if not success:
                self.get_logger().warn(f"Inbound queue full, dropped message from {topic_name}")
                
        except Exception as e:
            self.get_logger().error(f"Error in ROS callback for {topic_name}: {e}")
            
    def _process_outbound_queue(self):
        """Process outbound messages from agents and publish to ROS topics"""
        try:
            # Process up to 10 messages per timer cycle to avoid blocking
            for _ in range(10):
                msg_data = self.queues.get_outbound_topic(timeout=0.0)
                if msg_data is None:
                    break
                    
                topic = msg_data['topic']
                ros_msg = msg_data['msg']
                
                # Find publisher for this topic
                if topic in self._topic_publishers:
                    try:
                        self._topic_publishers[topic]['publisher'].publish(ros_msg)
                    except Exception as e:
                        self.get_logger().error(f"Failed to publish to {topic}: {e}")
                else:
                    self.get_logger().warn(f"No publisher configured for topic {topic}")
                    
        except Exception as e:
            self.get_logger().error(f"Error processing outbound queue: {e}")
            
    def _log_metrics(self):
        """Log bridge metrics periodically"""
        metrics = self.queues.get_metrics()
        if metrics['total_messages'] > 0:
            self.get_logger().info(
                f"Bridge metrics - Inbound: {metrics['inbound_queue_size']}, "
                f"Outbound: {metrics['outbound_queue_size']}, "
                f"Dropped: {metrics['dropped_messages']}, "
                f"Total: {metrics['total_messages']}"
            )
            
    def get_queues(self) -> MessageQueues:
        """Provide direct queue access for simple agents"""
        return self.queues
        
    def register_agent_interface(self, agent_id: str) -> AgentInterface:
        """Register agent and return dedicated interface with tracking"""
        interface = AgentInterface(agent_id, self.queues)
        self._agent_interfaces.append(interface)
        self.get_logger().info(f"Registered agent interface: {agent_id}")
        return interface
        
    def get_agent_interfaces(self) -> List[AgentInterface]:
        """Get list of active agent interfaces"""
        return [iface for iface in self._agent_interfaces if iface.active]
        
    async def stop_bridge(self):
        """Clean shutdown with queue drainage"""
        if not self._started:
            return
            
        self.get_logger().info("Stopping ROS AI Bridge...")
        
        try:
            # Signal shutdown
            self._shutdown_event.set()
            
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
                    self.get_logger().info(f"Drained {drained} inbound messages")
                    
            # Close all agent interfaces
            for interface in self._agent_interfaces:
                await interface.close()
                
            # Destroy subscriptions
            for subscription in self._subscriptions.values():
                subscription.destroy()
            self._subscriptions.clear()
            
            self._started = False
            self.get_logger().info("ROS AI Bridge stopped")
            
        except Exception as e:
            self.get_logger().error(f"Error stopping bridge: {e}")


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