#!/usr/bin/env python3
"""
WebSocket Reflection Proxy for Cross-Agent Communication

A transparent proxy that sits between agents and the ROS bridge, enabling
direct agent-to-agent message reflection while maintaining full compatibility
with standard WebSocket bridges.

The proxy:
- Accepts connections from multiple agents
- Maintains single upstream connection to bridge
- Reflects messages between agents based on subscriptions
- Transparently forwards all messages to/from bridge

Author: Karim Virani
Date: September 2025
"""

import asyncio
import json
import logging
import signal
import sys
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, field

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
except ImportError:
    print("Error: websockets package not installed. Install with: pip install websockets")
    sys.exit(1)


@dataclass
class AgentInfo:
    """Information about a connected agent"""
    agent_id: str
    websocket: WebSocketServerProtocol
    subscriptions: Set[str] = field(default_factory=set)
    agent_role: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WebSocketReflectionProxy:
    """
    WebSocket proxy with reflection capability for cross-agent communication.

    This proxy acts as a multiplexer between multiple agents and a single
    bridge connection, with the added capability of reflecting messages
    directly between agents when they subscribe to each other's topics.
    """

    def __init__(self,
                 listen_host: str = "0.0.0.0",
                 listen_port: int = 8766,
                 bridge_host: str = "localhost",
                 bridge_port: int = 8765,
                 enable_reflection: bool = True,
                 reflection_whitelist: Optional[Set[str]] = None):
        """
        Initialize the reflection proxy.

        Args:
            listen_host: Host to listen on for agent connections
            listen_port: Port to listen on for agent connections
            bridge_host: Host of the upstream bridge
            bridge_port: Port of the upstream bridge
            enable_reflection: Whether to enable cross-agent reflection
            reflection_whitelist: Optional set of topics to reflect (None = all)
        """
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.bridge_url = f"ws://{bridge_host}:{bridge_port}"
        self.enable_reflection = enable_reflection
        self.reflection_whitelist = reflection_whitelist

        # Connected agents tracking
        self.agents: Dict[str, AgentInfo] = {}
        self.websocket_to_agent: Dict[int, str] = {}  # Map websocket id() to agent_id

        # Bridge connection
        self.bridge_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.bridge_connected = False
        self.bridge_task: Optional[asyncio.Task] = None

        # Combined subscriptions for bridge
        self.all_subscriptions: Set[str] = set()

        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Metrics
        self.metrics = {
            "agents_connected": 0,
            "messages_from_agents": 0,
            "messages_from_bridge": 0,
            "messages_reflected": 0,
            "messages_forwarded": 0,
        }

    async def start(self):
        """Start the proxy server and connect to bridge"""
        self.logger.info(f"🚀 Starting WebSocket Reflection Proxy")
        self.logger.info(f"   Listening on {self.listen_host}:{self.listen_port}")
        self.logger.info(f"   Upstream bridge: {self.bridge_url}")
        self.logger.info(f"   Reflection: {'Enabled' if self.enable_reflection else 'Disabled'}")

        # Add a small delay to ensure bridge WebSocket server is ready
        # This helps avoid the race condition where proxy starts before bridge
        self.logger.info("⏳ Waiting 2 seconds for bridge to be ready...")
        await asyncio.sleep(2.0)

        # Connect to upstream bridge (don't fail if bridge isn't ready yet)
        try:
            await self.connect_to_bridge()
        except Exception as e:
            self.logger.warning(f"⚠️ Initial bridge connection failed: {e}")
            self.logger.info("Will retry in background...")
            # Schedule reconnection
            asyncio.create_task(self.reconnect_to_bridge())

        # Start server for agents
        async with websockets.serve(
            self.handle_agent_connection,
            self.listen_host,
            self.listen_port
        ) as server:
            self.logger.info(f"✅ Proxy server started on port {self.listen_port}")
            await asyncio.Future()  # Run forever

    async def connect_to_bridge(self):
        """Establish connection to upstream bridge"""
        self.logger.info(f"🔗 Connecting to bridge at {self.bridge_url}")
        try:
            self.bridge_ws = await websockets.connect(self.bridge_url)
            self.logger.info(f"📡 WebSocket connection established with bridge")
            self.bridge_connected = True

            # Don't register the proxy itself - be transparent
            # Just start handling bridge messages
            # Agents will register themselves through us

            # Start task to handle bridge messages
            self.bridge_task = asyncio.create_task(self.handle_bridge_messages())

            # Re-forward any existing agent registrations to the bridge
            # This handles the case where agents connected while bridge was down
            await self.reforward_agent_registrations()

            self.logger.info("✅ Connected to upstream bridge")

        except Exception as e:
            self.logger.error(f"❌ Failed to connect to bridge: {e}")
            self.bridge_connected = False
            # Schedule reconnection
            asyncio.create_task(self.reconnect_to_bridge())

    async def update_bridge_subscriptions(self):
        """No longer needed - agents register directly through transparent proxy"""
        # This method is kept for compatibility but does nothing
        # The proxy is now fully transparent and doesn't register itself
        pass

    async def register_with_bridge(self, wait_for_response=True):
        """Deprecated - proxy no longer registers itself"""
        # Kept for backward compatibility but does nothing
        # The proxy is transparent - only agents register
        if False:  # Never execute, kept for reference
            # Old registration code that made proxy non-transparent
            try:
                response = await asyncio.wait_for(self.bridge_ws.recv(), timeout=5.0)
                data = json.loads(response)
                if data.get("status") == "success":
                    self.logger.info("✅ Proxy registered with bridge")
                else:
                    self.logger.warning(f"⚠️ Registration response: {data}")
            except asyncio.TimeoutError:
                self.logger.warning("⚠️ Registration response timeout")
        else:
            self.logger.info("📤 Sent registration update to bridge")

    def get_msg_type_for_topic(self, topic: str) -> str:
        """Get ROS message type for a given topic name"""
        # Common topic to message type mappings
        topic_types = {
            "prompt_voice": "by_your_command/AudioDataUtterance",
            "prompt_text": "std_msgs/String",
            "response_voice": "audio_common_msgs/AudioData",
            "response_text": "std_msgs/String",
            "response_cmd": "std_msgs/String",
            "prompt_transcript": "std_msgs/String",
            "conversation_id": "std_msgs/String",
            "interruption_signal": "std_msgs/Bool",
            "command_detected": "std_msgs/Bool",
            "voice_active": "std_msgs/Bool",
            "cmd_vel": "geometry_msgs/Twist"
        }

        # Check for image topics
        if "image" in topic.lower() and "compressed" in topic.lower():
            return "sensor_msgs/CompressedImage"
        elif "image" in topic.lower():
            return "sensor_msgs/Image"

        # Return known type or default to String
        return topic_types.get(topic, "std_msgs/String")

    async def reconnect_to_bridge(self):
        """Attempt to reconnect to bridge after failure"""
        await asyncio.sleep(5)  # Wait before reconnecting
        self.logger.info("🔄 Attempting to reconnect to bridge...")
        await self.connect_to_bridge()

    async def reforward_agent_registrations(self):
        """Re-forward existing agent registrations to the bridge after reconnection"""
        if not self.agents:
            return

        self.logger.info(f"📤 Re-forwarding {len(self.agents)} agent registrations to bridge")

        for agent_id, agent_info in self.agents.items():
            # Reconstruct the registration message
            registration = {
                "type": "register",
                "agent_id": agent_id,
                "capabilities": agent_info.metadata.get("capabilities", ["audio_processing", "realtime_api"]),
                "subscriptions": [{"topic": topic, "msg_type": self.get_msg_type_for_topic(topic)}
                                  for topic in agent_info.subscriptions]
            }

            try:
                if self.bridge_connected and self.bridge_ws:
                    await self.bridge_ws.send(json.dumps(registration))
                    self.logger.info(f"✅ Re-forwarded registration for agent {agent_id}")
                    self.metrics["messages_forwarded"] += 1
            except Exception as e:
                self.logger.error(f"Failed to re-forward registration for {agent_id}: {e}")

    async def handle_agent_connection(self, websocket: WebSocketServerProtocol, path: str = None):
        """Handle new agent connection"""
        agent_id = None
        # Handle both old and new websockets API (path is optional in newer versions)
        remote_addr = getattr(websocket, 'remote_address', 'unknown')
        self.logger.info(f"🤝 New agent connection from {remote_addr}")

        try:
            async for message in websocket:
                self.metrics["messages_from_agents"] += 1

                try:
                    data = json.loads(message)
                    message_type = data.get("type", "")
                    self.logger.debug(f"Received message type: {message_type}, data keys: {data.keys() if isinstance(data, dict) else 'not a dict'}")

                    # Handle registration
                    if message_type == "register":
                        agent_id = await self.handle_agent_registration(websocket, data)

                    # Handle outbound messages (agent → ros-internal)
                    elif message_type == "outbound_message" and agent_id:
                        await self.handle_outbound_message(agent_id, data)

                    # Forward everything to bridge
                    if self.bridge_connected and self.bridge_ws:
                        await self.bridge_ws.send(message)
                        self.metrics["messages_forwarded"] += 1

                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON from agent: {e}")
                except Exception as e:
                    import traceback
                    self.logger.error(f"Error handling agent message: {e}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")

        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"📴 Agent {agent_id or 'unknown'} disconnected")
        finally:
            if agent_id:
                await self.handle_agent_disconnection(agent_id)

    async def handle_agent_registration(self, websocket: WebSocketServerProtocol, data: Dict) -> str:
        """
        Handle agent registration message.

        Track agent subscriptions and metadata for reflection.
        """
        agent_id = data.get("agent_id", f"agent_{id(websocket)}")
        # Ensure subscriptions are strings only
        raw_subscriptions = data.get("subscriptions", [])
        subscriptions = set()
        for sub in raw_subscriptions:
            if isinstance(sub, str):
                subscriptions.add(sub)
            elif isinstance(sub, dict):
                # Handle case where subscription might be sent as {"topic": "topic_name"}
                if "topic" in sub:
                    subscriptions.add(sub["topic"])
                else:
                    self.logger.warning(f"Invalid subscription format from {agent_id}: {sub}")
            else:
                self.logger.warning(f"Invalid subscription type from {agent_id}: {type(sub)} - {sub}")

        metadata = data.get("metadata", {})
        agent_role = metadata.get("agent_role", "unknown")

        # Store agent info
        agent_info = AgentInfo(
            agent_id=agent_id,
            websocket=websocket,
            subscriptions=subscriptions,
            agent_role=agent_role,
            metadata=metadata
        )

        self.agents[agent_id] = agent_info
        self.websocket_to_agent[id(websocket)] = agent_id

        # Update combined subscriptions
        self.update_combined_subscriptions()

        self.metrics["agents_connected"] = len(self.agents)
        self.logger.info(f"✅ Registered agent: {agent_id} (role: {agent_role})")
        self.logger.info(f"   Subscriptions: {subscriptions}")

        # Send registration response back to agent
        response = {
            "type": "register_response",
            "status": "success",
            "agent_id": agent_id,
            "message": f"Registered with reflection proxy"
        }
        try:
            await websocket.send(json.dumps(response))
            self.logger.debug(f"Sent registration response to {agent_id}")
        except Exception as e:
            self.logger.error(f"Failed to send registration response to {agent_id}: {e}")

        return agent_id

    async def handle_agent_disconnection(self, agent_id: str):
        """Clean up after agent disconnection"""
        if agent_id in self.agents:
            agent_info = self.agents[agent_id]

            # Remove from tracking
            del self.agents[agent_id]
            ws_id = id(agent_info.websocket)
            if ws_id in self.websocket_to_agent:
                del self.websocket_to_agent[ws_id]

            # Update combined subscriptions
            self.update_combined_subscriptions()

            self.metrics["agents_connected"] = len(self.agents)
            self.logger.info(f"🧹 Cleaned up agent: {agent_id}")

    def update_combined_subscriptions(self):
        """Update the combined subscription list for the bridge"""
        old_subscriptions = self.all_subscriptions.copy()
        self.all_subscriptions = set()
        for agent_info in self.agents.values():
            self.all_subscriptions.update(agent_info.subscriptions)

        self.logger.debug(f"📋 Combined subscriptions: {self.all_subscriptions}")

        # Proxy no longer needs to re-register since it doesn't register itself
        # It's transparent - agents handle their own registrations
        # Just log the subscription change for debugging
        if self.all_subscriptions != old_subscriptions:
            self.logger.debug(f"Agent subscriptions updated: {len(self.all_subscriptions)} topics")

    async def handle_outbound_message(self, sender_id: str, data: Dict):
        """
        Handle outbound message from agent.

        Check if message should be reflected to other agents.
        """
        if not self.enable_reflection:
            return

        full_topic = data.get("topic", "")
        # Extract base topic name for matching
        # e.g., /grunt1/agent/response_text -> response_text
        topic = full_topic.split('/')[-1] if '/' in full_topic else full_topic

        # Ensure topic is a string, not a dict or other type
        if not isinstance(topic, str):
            self.logger.warning(f"Invalid topic type from {sender_id}: {type(topic)} - {topic}")
            return

        metadata = data.get("metadata", {})

        # Check if topic is in whitelist (if configured)
        if self.reflection_whitelist and topic not in self.reflection_whitelist:
            return

        # Check each agent's subscriptions
        reflected_to = []
        for agent_id, agent_info in self.agents.items():
            # Don't reflect to sender
            if agent_id == sender_id:
                continue

            # Check if agent subscribes to this topic
            if topic in agent_info.subscriptions:
                # Create reflected message (as if from bridge)
                reflected_msg = {
                    "type": "inbound_message",
                    "topic": topic,
                    "ros_msg_type": data.get("msg_type", ""),
                    "data": data.get("data", {}),
                    "metadata": metadata  # Preserve metadata!
                }

                try:
                    await agent_info.websocket.send(json.dumps(reflected_msg))
                    reflected_to.append(agent_id)
                    self.metrics["messages_reflected"] += 1
                except Exception as e:
                    self.logger.error(f"Failed to reflect to {agent_id}: {e}")

        if reflected_to:
            self.logger.debug(f"🔄 Reflected {topic} from {sender_id} to {reflected_to}")

    async def handle_bridge_messages(self):
        """
        Handle messages from bridge and forward to agents.

        This runs as a background task for the bridge connection.
        """
        try:
            async for message in self.bridge_ws:
                self.metrics["messages_from_bridge"] += 1

                try:
                    data = json.loads(message)

                    # Handle bridge message format with envelope
                    if data.get("type") == "message" and "envelope" in data:
                        envelope = data["envelope"]
                        full_topic = envelope.get("topic_name", "")
                        # Extract base topic name (last part after /)
                        # e.g., /grunt1/agent/prompt_voice -> prompt_voice
                        topic = full_topic.split('/')[-1] if '/' in full_topic else full_topic
                    else:
                        # Fallback to direct topic field
                        topic = data.get("topic", "")

                    # Ensure topic is a string
                    if not isinstance(topic, str):
                        self.logger.warning(f"Invalid topic type from bridge: {type(topic)} - {topic}")
                        continue

                    # Forward to agents that subscribe to this topic
                    forwarded_to = []
                    for agent_id, agent_info in self.agents.items():
                        if topic in agent_info.subscriptions:
                            try:
                                await agent_info.websocket.send(message)
                                forwarded_to.append(agent_id)
                            except Exception as e:
                                self.logger.error(f"Failed to forward to {agent_id}: {e}")

                    if forwarded_to:
                        self.logger.debug(f"📨 Forwarded {topic} (full: {full_topic if 'full_topic' in locals() else topic}) from bridge to {forwarded_to}")
                    else:
                        self.logger.debug(f"⚠️ No agents subscribed to {topic} (full: {full_topic if 'full_topic' in locals() else topic})")

                except json.JSONDecodeError:
                    # Forward non-JSON messages as-is
                    for agent_info in self.agents.values():
                        try:
                            await agent_info.websocket.send(message)
                        except Exception:
                            pass

        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("⚠️ Bridge connection closed")
            self.bridge_connected = False
            await self.reconnect_to_bridge()

    def get_metrics(self) -> Dict[str, Any]:
        """Get proxy metrics"""
        return self.metrics.copy()


async def main():
    """Main entry point"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create proxy with default configuration
    proxy = WebSocketReflectionProxy(
        listen_port=8766,
        bridge_host="localhost",
        bridge_port=8765,
        enable_reflection=True,
        reflection_whitelist=None  # Reflect all topics
    )

    # Handle shutdown gracefully
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig, lambda: asyncio.create_task(shutdown(proxy))
        )

    # Start proxy
    await proxy.start()


async def shutdown(proxy):
    """Graceful shutdown"""
    logging.info("🛑 Shutting down proxy...")
    if proxy.bridge_ws:
        await proxy.bridge_ws.close()
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())