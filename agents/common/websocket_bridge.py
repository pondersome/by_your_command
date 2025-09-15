#!/usr/bin/env python3
"""
WebSocket Bridge Interface for OpenAI Realtime Agent

Provides WebSocket client connection to ROS AI Bridge for distributed deployment.
Replaces direct bridge instantiation with network-based communication.

Author: Karim Virani
Version: 1.0
Date: July 2025
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
import websockets
from websockets import WebSocketClientProtocol
from datetime import datetime


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
        elif self.ros_msg_type == "audio_common_msgs/AudioData":
            return AudioDataProxy(data)
        elif self.ros_msg_type == "sensor_msgs/CompressedImage":
            return CompressedImageProxy(data)
        else:
            return DataProxy(data)


class AudioDataUtteranceProxy:
    """Proxy object for AudioDataUtterance from WebSocket"""
    
    def __init__(self, data: Dict):
        # AudioDataUtterance fields - audio data in multiple formats
        self.int16_data = data.get("int16_data", [])
        self.float32_data = data.get("float32_data", [])
        self.int32_data = data.get("int32_data", [])
        self.int8_data = data.get("int8_data", [])
        self.uint8_data = data.get("uint8_data", [])
        
        # Utterance metadata fields (only these exist in AudioDataUtterance.msg)
        self.utterance_id = data.get("utterance_id", 0)
        self.is_utterance_end = data.get("is_utterance_end", False)
        self.chunk_sequence = data.get("chunk_sequence", 0)


class StringProxy:
    """Proxy object for std_msgs/String from WebSocket"""
    
    def __init__(self, data: Dict):
        self.data = data.get("data", "")


class AudioDataProxy:
    """Proxy object for audio_common_msgs/AudioData from WebSocket"""
    
    def __init__(self, data: Dict):
        self.int16_data = data.get("int16_data", [])


class CompressedImageProxy:
    """Proxy object for sensor_msgs/CompressedImage from WebSocket"""
    
    def __init__(self, data: Dict):
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Header fields
            self.header = data.get("header", {})
            # Image format (e.g., "rgb8; jpeg compressed bgr8")
            self.format = data.get("format", "")
            # JPEG data - handle both base64 string (new) and list (legacy) formats
            data_field = data.get("data", [])
            
            if isinstance(data_field, str):
                # New base64 format - much more efficient!
                import base64
                self.data = base64.b64decode(data_field)
            elif isinstance(data_field, list):
                # Legacy JSON array format (fallback)
                self.data = bytes(data_field)
            else:
                self.data = data_field
        except Exception as e:
            logger.error(f"❌ CompressedImageProxy init failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise


class DataProxy:
    """Generic proxy object for unknown message types"""
    
    def __init__(self, data: Dict):
        self.__dict__.update(data)


class WebSocketBridgeInterface:
    """WebSocket client for bridge communication"""
    
    def __init__(self, config: Dict):
        self.config = config  # Store the full config for access in other methods
        self.host = config.get('bridge_connection', {}).get('host', 'localhost')
        self.port = config.get('bridge_connection', {}).get('port', 8765)
        self.agent_id = config.get('agent_id', 'openai_realtime')
        self.reconnect_interval = config.get('bridge_connection', {}).get('reconnect_interval', 5.0)
        self.max_reconnect_attempts = config.get('bridge_connection', {}).get('max_reconnect_attempts', 10)
        
        # Connection state
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.connected = False
        self.reconnect_attempts = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Metrics
        self.messages_received = 0
        self.messages_sent = 0
        self.connection_attempts = 0
        
    async def connect_with_retry(self) -> bool:
        """Connect to bridge with initial retry attempts"""
        for attempt in range(1, self.max_reconnect_attempts + 1):
            self.logger.info(f"Initial connection attempt {attempt}/{self.max_reconnect_attempts}")
            
            success = await self.connect()
            if success:
                return True
                
            if attempt < self.max_reconnect_attempts:
                self.logger.info(f"Connection failed, retrying in {self.reconnect_interval}s...")
                await asyncio.sleep(self.reconnect_interval)
            else:
                self.logger.error(f"Failed to connect after {self.max_reconnect_attempts} attempts")
                
        return False
        
    async def connect(self) -> bool:
        """Connect to bridge WebSocket server with retry logic"""
        self.connection_attempts += 1
        
        try:
            uri = f"ws://{self.host}:{self.port}"
            self.logger.info(f"Connecting to bridge at {uri} (attempt {self.connection_attempts})")
            
            self.websocket = await websockets.connect(
                uri,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            # Register with bridge
            success = await self._register_agent()
            if success:
                self.connected = True
                self.reconnect_attempts = 0
                
                # Start message listener
                asyncio.create_task(self._message_listener())
                
                self.logger.info(f"✅ Connected to bridge at {uri}")
                return True
            else:
                await self.websocket.close()
                self.websocket = None
                return False
                
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            self.websocket = None
            return False
            
    async def _register_agent(self) -> bool:
        """Register agent with bridge"""
        try:
            # Build subscription list based on configuration
            subscriptions = [
                {"topic": "prompt_voice", "msg_type": "by_your_command/AudioDataUtterance"},
                {"topic": "prompt_text", "msg_type": "std_msgs/String"},
                {"topic": "conversation_id", "msg_type": "std_msgs/String"}
            ]
            
            # Add camera subscription if video is enabled
            if self.config.get('enable_video', False):
                subscriptions.append({
                    "topic": "/grunt1/arm1/cam_live/color/image_raw/compressed",  # JPEG compressed - ready for Gemini
                    "msg_type": "sensor_msgs/CompressedImage"
                })
                self.logger.info("Video enabled - subscribing to compressed camera topic")
            
            registration = {
                "type": "register",
                "agent_id": self.agent_id,
                "capabilities": ["audio_processing", "realtime_api"],
                "subscriptions": subscriptions
            }
            
            await self.websocket.send(json.dumps(registration))
            self.logger.debug(f"Sent registration for agent: {self.agent_id}")
            
            # Wait for registration response
            response = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
            data = json.loads(response)
            
            if data.get("status") == "success":
                session_id = data.get("session_id", "unknown")
                self.logger.info(f"Agent registered successfully. Session: {session_id}")
                
                # Log namespace info if provided
                namespace_info = data.get("namespace_info", {})
                if namespace_info and (namespace_info.get("namespace") or namespace_info.get("prefix")):
                    self.logger.info(f"Bridge namespace: /{namespace_info.get('full_prefix', '')}")
                    
                return True
            else:
                error_msg = data.get("message", "Unknown error")
                self.logger.error(f"Registration failed: {error_msg}")
                return False
                
        except asyncio.TimeoutError:
            self.logger.error("Registration timed out")
            return False
        except Exception as e:
            self.logger.error(f"Registration error: {e}")
            return False
            
    async def _message_listener(self):
        """Listen for messages from bridge"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type", "")
                    
                    if message_type == "message":
                        # Queue message for agent processing
                        msg_type = data["envelope"].get('ros_msg_type', 'unknown')
                        await self.message_queue.put(data["envelope"])
                        self.messages_received += 1
                        if "Image" in msg_type:
                            # Reduce image message spam
                            self.logger.debug(f"📥 IMAGE queued: {msg_type}")
                        else:
                            self.logger.info(f"📥 Queued message: {msg_type} (queue size: {self.message_queue.qsize()})")
                        
                    elif message_type == "heartbeat":
                        # Respond to heartbeat
                        await self.websocket.send(json.dumps({"type": "heartbeat_response"}))
                        self.logger.debug("💓 Heartbeat responded")
                        
                    else:
                        self.logger.debug(f"Received message type: {message_type}")
                        
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON received: {e}")
                except Exception as e:
                    self.logger.error(f"Message processing error: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
            self.connected = False
        except Exception as e:
            self.logger.error(f"Message listener error: {e}")
            self.connected = False
            
        # Try to reconnect if connection was lost
        if not self.connected:
            await self._handle_disconnection()
            
    async def _handle_disconnection(self):
        """Handle connection loss and attempt reconnection"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            return
            
        self.reconnect_attempts += 1
        self.logger.info(f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts}")
        
        await asyncio.sleep(self.reconnect_interval)
        
        success = await self.connect()
        if not success:
            # Will try again due to recursive call in _message_listener
            pass
            
    async def get_inbound_message(self, timeout: float = 1.0) -> Optional[WebSocketMessageEnvelope]:
        """Get message from bridge (compatible with existing interface)"""
        if not self.connected:
            return None
            
        # Debug log to see queue size
        if self.message_queue.qsize() > 0:
            self.logger.debug(f"📋 Getting message from queue (size: {self.message_queue.qsize()})")
            
        try:
            envelope_data = await asyncio.wait_for(self.message_queue.get(), timeout)
            msg_type = envelope_data.get('ros_msg_type', 'unknown')
            if "Image" in msg_type:
                self.logger.debug(f"📤 Retrieved image: {msg_type}")
            else:
                self.logger.info(f"📤 Retrieved message: {msg_type}")
            
            # Debug: Check int16_data in the envelope  
            if envelope_data.get('ros_msg_type') == 'by_your_command/AudioDataUtterance':
                data_dict = envelope_data.get('data', {})
                int16_data = data_dict.get('int16_data', [])
                self.logger.info(f"🔍 WebSocket envelope int16_data: type={type(int16_data)}, length={len(int16_data) if hasattr(int16_data, '__len__') else 'N/A'}")
            
            return WebSocketMessageEnvelope(envelope_data)
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self.logger.error(f"Error getting inbound message: {e}")
            return None
            
    async def put_outbound_message(self, topic: str, msg_data: Dict, msg_type: str) -> bool:
        """Send message back to bridge"""
        if not self.connected or not self.websocket:
            self.logger.warning("Cannot send message - not connected to bridge")
            return False
            
        try:
            outbound = {
                "type": "outbound_message",
                "topic": topic,
                "msg_type": msg_type,
                "data": msg_data
            }
            
            outbound_json = json.dumps(outbound)
            self.logger.info(f"📤 Sending outbound message to bridge: {outbound_json[:100]}...")
            await self.websocket.send(outbound_json)
            self.messages_sent += 1
            self.logger.info(f"✅ Successfully sent message to topic: {topic}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send outbound message: {e}")
            return False
            
    async def close(self):
        """Close WebSocket connection"""
        self.connected = False
        
        if self.websocket:
            try:
                await self.websocket.close()
                self.logger.info("WebSocket connection closed")
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {e}")
        
        self.websocket = None
        
    def is_connected(self) -> bool:
        """Check if connected to bridge"""
        return self.connected and self.websocket is not None
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get connection metrics"""
        return {
            "connected": self.connected,
            "messages_received": self.messages_received,
            "messages_sent": self.messages_sent,
            "connection_attempts": self.connection_attempts,
            "reconnect_attempts": self.reconnect_attempts,
            "queue_size": self.message_queue.qsize()
        }
        
    def get_status_summary(self) -> str:
        """Get human-readable status summary"""
        status = "Connected" if self.connected else "Disconnected"
        return f"Bridge: {status} | RX: {self.messages_received} | TX: {self.messages_sent}"